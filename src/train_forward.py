import torch
import torch.nn as nn
from utils import character_level_tokenizer, TransformerModel, pad
import pickle

# Extracts a batch of prompts and forward-answers, tokenizes them, and pads them
def get_batch_lm(data, i, batch_size, tokenizer):
    batch = data[i:i+batch_size]
    # Encode prompts
    prompts = [tokenizer.encode(p) for p, a, _ in batch]
    padded_prompts, length_prompts = pad(prompts, tokenizer, "prompts")
    # Encode answers (Forward targets) and append [EOS]
    answers = [tokenizer.encode(a) + [tokenizer.token_to_id['[EOS]']] for p, a, _ in batch]
    padded_answers, length_answers = pad(answers, tokenizer, "answers")
    
    # Transpose lists into standard PyTorch tensor shape (Sequence_Length, Batch_Size)
    X = torch.stack([torch.tensor(x) for x in padded_prompts], 1)
    Y = torch.stack([torch.tensor(x) for x in padded_answers], 1)
    return X, Y, length_prompts, length_answers

# Evaluates the model on validation data by performing full autoregressive generation and comparing the generated sequence exactly with the target sequence
def evaluate_lm(model, data, tokenizer, batch_size, device):
    model.eval()
    correct_digit = 0
    total_digit = 0
    correct_seq = 0
    total_seq = 0
    with torch.no_grad():
        # Evaluate on a subset (max 1000) for speed during training loop
        for i in range(0, min(1000, len(data)), batch_size): 
            batch = data[i:i+batch_size]
            prompts = [tokenizer.encode(p) for p, a, _ in batch]
            answers = [tokenizer.encode(a) + [tokenizer.token_to_id['[EOS]']] for p, a, _ in batch]
            max_prompt_len = max(len(p) for p in prompts)
            max_ans_len = max(len(a) for a in answers)
            
            # Pad the sequences
            padded_prompts = [[tokenizer.token_to_id['[PAD]']] * (max_prompt_len - len(p)) + p for p in prompts]
            padded_answers = [a + [tokenizer.token_to_id['[PAD]']] * (max_ans_len - len(a)) for a in answers]
            
            X = torch.tensor(padded_prompts, dtype=torch.long).transpose(0, 1).to(device)
            Y = torch.tensor(padded_answers, dtype=torch.long).transpose(0, 1).to(device)
            
            input_tensor = X
            # Autoregressively generate tokens for the maximum answer length
            for _ in range(max_ans_len):
                output, _, _ = model(input_tensor)
                token = torch.argmax(output[-1, :, :], -1).view(1, -1)
                input_tensor = torch.cat((input_tensor, token), 0)
                
            pred_answers = input_tensor[max_prompt_len:max_prompt_len+max_ans_len, :]
            
            # Mask out padding tokens to ensure they don't affect accuracy artificially
            mask = Y != tokenizer.token_to_id['[PAD]']
            equality = pred_answers == Y
            
            # Calculate metrics
            correct_digit += torch.logical_and(mask, equality).sum().item()
            total_digit += mask.sum().item()
            seq_correct = torch.all(torch.logical_or(~mask, equality), dim=0)
            correct_seq += seq_correct.sum().item()
            total_seq += Y.shape[1]

    return correct_digit / total_digit if total_digit > 0 else 0, correct_seq / total_seq if total_seq > 0 else 0

if __name__ == "__main__":
    # Load consolidated dataset
    with open("assets/dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    
    data_train = dataset["train"]
    data_val = dataset["val"]
    
    # Initialize the tokenizer
    tokenizer = character_level_tokenizer()
    print(f"ntokens in train_forward: {tokenizer.ntokens}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize a lightweight transformer model suitable for simple character sequences
    model = TransformerModel(ntoken=tokenizer.ntokens, ninp=128, nhead=16, nhid=64, nlayers=6).to(device)
    
    # CrossEntropyLoss automatically ignores padded indices during backpropagation
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id['[PAD]'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Training configuration
    batch_size = 100
    epochs = 30
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(data_train), batch_size):
            X, Y, length_prompts, length_answers = get_batch_lm(data_train, i, batch_size, tokenizer)
            X, Y = X.to(device), Y.to(device)
            # Concatenate prompt and target for autoregressive teaching (Teacher Forcing)
            input_tensor = torch.cat((X, Y), 0) 
            
            optimizer.zero_grad()
            output, _, _ = model(input_tensor) 
            
            # Shift the output to align predictions with the targets
            output_answers = output[length_prompts-1:-1, :, :].reshape(-1, tokenizer.ntokens)
            target_answers = Y.reshape(-1)
            
            loss = criterion(output_answers, target_answers)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        train_loss = total_loss / (len(data_train) / batch_size)
        val_digit_acc, val_seq_acc = evaluate_lm(model, data_val, tokenizer, batch_size, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Val Digit Acc: {val_digit_acc:.4f}, Val Seq Acc: {val_seq_acc:.4f}")

    # Save the trained baseline model
    torch.save(model.state_dict(), "assets/forward_model.pth")
    print("Saved forward_model.pth")
