import re
import random
import math
import torch
import torch.nn as nn

# special tokens for padding and end of sequence
pad_token = "[PAD]"
eos_token = "[EOS]"

class character_level_tokenizer:
    def __init__(self):
        # vocabulary: digits, arithmetic operators, and special tokens
        self.vocab = [str(x) for x in range(10)] + ["+", "-", "="] + [pad_token, eos_token]
        # token -> integer id
        self.token_to_id = {v: k for k, v in enumerate(self.vocab)}
        # integer id -> token
        self.id_to_token = {k: v for k, v in enumerate(self.vocab)}
        # total number of tokens
        self.ntokens = len(self.vocab)
        # remove characters that are not in the vocabulary
        self.pattern = f"[^{re.escape(''.join(self.vocab))}]"

    def clean(self, text):
        # remove unsupported characters
        out = re.sub(self.pattern, "", text)
        return out

    def pre_tokenization(self, text):
        # split the text into individual characters
        return [c for c in text]

    def encode(self, text):
        # convert each character into its token id
        text_list = self.pre_tokenization(self.clean(text))
        return [self.token_to_id[c] for c in text_list]

    def decode(self, token_list):
        # convert token ids back into a string
        return "".join([self.id_to_token[i] for i in token_list])

def sample_datapoint(number_digits=3, operation="+"):
    # generate the first operand
    a_list = [random.randint(0, 9) for _ in range(number_digits)]
    a_int = int("".join([str(x) for x in a_list]))

    # generate the second operand
    b_list = [random.randint(0, 9) for _ in range(number_digits)]
    b_int = int("".join([str(x) for x in b_list]))

    # calculate the result
    if operation == "+":
        result = a_int + b_int
    elif operation == "-":
        result = a_int - b_int
    else:
        raise ValueError("operation must be '+' or '-'")

    # return the prompt and answer
    return (
        str(a_int) + operation + str(b_int) + "=",
        str(result)
    )

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2).float() / d_model) * (-math.log(1e4)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # store positional encoding with the model
        self.register_buffer("pe", pe)

    def forward(self, x):
        # add positional encoding to the token embeddings
        x = x + self.pe[:x.size(0)]
        return x

def pad(token_list, tokenizer, type_list="prompts"):
    # check whether prompts or answers are being padded
    assert type_list in ["prompts", "answers"]
    # find the longest sequence in the batch
    max_length = max([len(x) for x in token_list])
    out = []
    for x in token_list:
        if type_list == "prompts":
            # left-pad prompts so the '=' signs are aligned
            out.append([tokenizer.token_to_id[pad_token]] * (max_length-len(x)) + x)
        if type_list == "answers":
            # append EOS and right-pad answers
            out.append(x + [tokenizer.token_to_id[eos_token]] + [tokenizer.token_to_id[pad_token]] * (max_length-len(x)))
    return out, max_length

def get_batch(data, i, batch_size, tokenizer, mode="forward"):
    # encode and pad the prompts
    prompts = [tokenizer.encode(data[idx][0]) for idx in range(i, min(i + batch_size, len(data)))]
    padded_prompts, length_prompts = pad(prompts, tokenizer, "prompts")

    # encode the answers based on the prediction mode
    answers = []
    for idx in range(i, min(i + batch_size, len(data))):
        answer = data[idx][1]

        # keep the original answer order (forward)
        if mode == "forward":
            target = answer

        # reverse the entire answer including the negative sign
        elif mode == "reverse_all":
            target = answer[::-1]

        # reverse only the digits and keep the negative sign at the front
        elif mode == "reverse_digits":
            if answer.startswith("-"):
                target = "-" + answer[:0:-1]
            else:
                target = answer[::-1]
                
        # show the error message if the mode is invalid
        else:
            raise ValueError("mode must be 'forward', 'reverse_all', or 'reverse_digits'")

        answers.append(tokenizer.encode(target))

    # pad the answers and append the EOS token
    padded_answers, length_answers = pad(answers, tokenizer, "answers")

    # convert the padded sequences into tensors
    X = torch.stack([torch.tensor(x) for x in padded_prompts], 1)
    Y = torch.stack([torch.tensor(x) for x in padded_answers], 1)

    return X, Y, length_prompts, length_answers