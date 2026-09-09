import torch
from torch import nn
from torch.nn import functional as F
import random
import math
import re

# Special tokens used for sequence alignment
pad_token = "[PAD]"
eos_token = "[EOS]"

# A character-level tokenizer that maps discrete characters (digits, operators, and steps) to integer IDs
class character_level_tokenizer:
    def __init__(self):
        # The vocabulary includes digits 0-9, operators (+, -, =), intermediate algorithmic tokens (c for carry, b for borrow), and control tokens.
        self.vocab = [str(x) for x in range(10)] + ["+", "-", "=", "c", "b"] + [pad_token, eos_token]
        self.token_to_id = {v: k for k, v in enumerate(self.vocab)}
        self.id_to_token = {k: v for k, v in enumerate(self.vocab)}
        self.ntokens = len(self.vocab)
        # Precompile regex pattern to quickly strip out unsupported characters
        self.pattern = f"[^{re.escape(''.join(self.vocab))}]"

    def clean(self, text):
        """Removes any characters not explicitly defined in the vocabulary."""
        out = re.sub(self.pattern, "", text)
        return out

    def pre_tokenization(self, text):
        """Splits the text into individual characters."""
        return [c for c in text]

    def encode(self, text):
        """Converts a string of characters into a list of token IDs."""
        text_list = self.pre_tokenization(self.clean(text))        
        return [self.token_to_id[c] for c in text_list]

    def decode(self, token_list):
        """Converts a list of token IDs back into a string."""
        return "".join([self.id_to_token[i] for i in token_list])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2).float() / d_model) * (-math.log(1e4)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Custom Transformer encoder layer designed to expose attention weights for visualization
class CustomEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False, **kwargs):
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal
        )
        self.attn_weights = attn_weights
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super().__init__()
        self.input_emb = nn.Embedding(ntoken, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        # Use CustomEncoderLayer to allow access to attention maps
        encoder_layers = CustomEncoderLayer(ninp, nhead, nhid, 0.1)
        self.encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(ninp, ntoken)
        self.ninp = ninp
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    # Generates a causal mask to prevent attention to future tokens during training
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):        
        mask = self._generate_square_subsequent_mask(len(src)).to(src.device)
        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output_enc = self.encoder(src, mask=mask, is_causal=True) 
        output_dec = self.decoder(output_enc)
        # Retrieve attention maps from the custom layers
        attention_maps = [layer.attn_weights for layer in self.encoder.layers]
        return F.log_softmax(output_dec, dim=-1), output_enc, attention_maps

# Generates a single synthetic arithmetic problem
def sample_datapoint(number_digits=3, force_negative=False):
    # Randomly generate two operands
    a_list = [random.randint(0, 9) for _ in range(number_digits)]
    a_int = int("".join([str(x) for x in a_list]))
    
    b_list = [random.randint(0, 9) for _ in range(number_digits)]
    b_int = int("".join([str(x) for x in b_list]))

    # Determine operation based on force_negative flag
    if force_negative:
        op = "-"
        if a_int >= b_int:
            a_int, b_int = b_int, a_int
            if a_int == b_int:
                b_int += 1
    else:
        op = random.choice(["+", "-"])
        if op == "-":
            # Prevent negative numbers unless explicitly forced
            if a_int < b_int:
                a_int, b_int = b_int, a_int

    # Calculate actual ground truth
    if op == "+":
        sum_int = a_int + b_int
    else:
        sum_int = a_int - b_int

    prompt = f"{a_int}{op}{b_int}="
    ans_forward = str(sum_int)
    
    # Construct the right-to-left "scratchpad" reverse sequence
    out_rev = ""
    if op == "+":
        a_str = str(a_int)
        b_str = str(b_int)
        max_len = max(len(a_str), len(b_str))
        a_str = a_str.zfill(max_len)
        b_str = b_str.zfill(max_len)
        carry = 0
        
        # Iterate from right (units) to left (highest order digit)
        for i in range(max_len - 1, -1, -1):
            d1 = int(a_str[i])
            d2 = int(b_str[i])
            s = d1 + d2 + carry
            digit = s % 10
            carry = s // 10
            out_rev += str(digit)
            if carry > 0:
                out_rev += "c1" # Explicitly append a carry token 'c1' to train algorithmic reasoning
        if carry > 0:
            out_rev += str(carry)
    else:
        # Subtraction logic
        a_tmp, b_tmp = a_int, b_int
        if a_tmp < b_tmp:
            out_rev += "-"
            a_tmp, b_tmp = b_tmp, a_tmp
            
        a_str = str(a_tmp)
        b_str = str(b_tmp)
        max_len = max(len(a_str), len(b_str))
        a_str = a_str.zfill(max_len)
        b_str = b_str.zfill(max_len)
        borrow = 0
        
        for i in range(max_len - 1, -1, -1):
            d1 = int(a_str[i])
            d2 = int(b_str[i])
            s = d1 - d2 - borrow
            if s < 0:
                s += 10
                borrow = 1
                b_token = "b1" # Explicitly append a borrow token 'b1'
            else:
                borrow = 0
                b_token = ""
            out_rev += str(s) + b_token
            
    # Clean up any trailing zeros from the reverse string (which correspond to leading zeros in a normal string)
    if out_rev == "0" or out_rev == "-0":
        out_rev = "0"
    else:
        has_minus = False
        if out_rev.startswith("-"):
            has_minus = True
            out_rev = out_rev[1:]
        while len(out_rev) > 1 and out_rev.endswith("0"):
            out_rev = out_rev[:-1]
        if has_minus and out_rev != "0":
            out_rev = "-" + out_rev
            
    ans_reverse = out_rev
    return (prompt, ans_forward, ans_reverse)

# Pads a batch of token lists to the same maximum length
def pad(token_list, tokenizer, type_list="prompts"):
    assert type_list in ['prompts', 'answers']
    max_length = max([len(x) for x in token_list])
    out = []
    for x in token_list:
        # Prompts are left-padded so the '=' signs align at the end
        if type_list == "prompts":
            out.append([tokenizer.token_to_id[pad_token]] * (max_length - len(x)) + x)
        # Answers are right-padded with EOS and PAD tokens
        if type_list == "answers":
            out.append(x + [tokenizer.token_to_id[eos_token]] + [tokenizer.token_to_id[pad_token]] * (max_length - len(x)))
    return out, max_length

# Retrieves and formats a single batch of tensors (Input X and Target Y) for training
def get_batch(data, i, batch_size, tokenizer):
    prompts = [tokenizer.encode(data[idx][0]) for idx in range(i, min(i + batch_size, len(data)))]
    padded_prompts, length_prompts = pad(prompts, tokenizer, "prompts")
    answers = [tokenizer.encode(data[idx][1]) for idx in range(i, min(i + batch_size, len(data)))]
    padded_answers, length_answers = pad(answers, tokenizer, "answers")
    X = torch.stack([torch.tensor(x) for x in padded_prompts], 1)
    Y = torch.stack([torch.tensor(x) for x in padded_answers], 1)
    return X, Y, length_prompts, length_answers

# Autoregressively generates sequence tokens one-by-one
def generate(model, prompts, new_tokens=5, device='cuda'):
    input_tensor = prompts.to(device)
    for _ in range(new_tokens):
        output, _, _ = model(input_tensor)
        last_output = output[-1, :, :]
        token = torch.argmax(last_output, -1).view((1, -1))
        input_tensor = torch.cat((input_tensor, token), 0)
    return input_tensor
