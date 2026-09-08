import torch
import torch.nn as nn
from src.utils import character_level_tokenizer, TransformerModel, pad
import pickle

def get_batch_lm(data, i, batch_size, tokenizer):
    batch = data[i:i+batch_size]
    prompts = [tokenizer.encode(p) for p, a in batch]
    padded_prompts, length_prompts = pad(prompts, tokenizer, "prompts")
    answers = [tokenizer.encode(a[::-1]) + [tokenizer.token_to_id['[EOS]']] for p, a in batch]
    padded_answers, length_answers = pad(answers, tokenizer, "answers")
    X = torch.stack([torch.tensor(x) for x in padded_prompts], 1)
    Y = torch.stack([torch.tensor(x) for x in padded_answers], 1)
    return X, Y, length_prompts, length_answers

def evaluate_lm(model, data, tokenizer, batch_size, device):
    model.eval()
    correct_digit = 0
    total_digit = 0
    correct_seq = 0
    total_seq = 0
    with torch.no_grad():
        for i in range(0, min(1000, len(data)), batch_size): 
            batch = data[i:i+batch_size]
            prompts = [tokenizer.encode(p) for p, a in batch]
            answers = [tokenizer.encode(a[::-1]) + [tokenizer.token_to_id['[EOS]']] for p, a in batch]
            max_prompt_len = max(len(p) for p in prompts)
            max_ans_len = max(len(a) for a in answers)
            
            padded_prompts = [[tokenizer.token_to_id['[PAD]']] * (max_prompt_len - len(p)) + p for p in prompts]
            padded_answers = [a + [tokenizer.token_to_id['[PAD]']] * (max_ans_len - len(a)) for a in answers]
            
            X = torch.tensor(padded_prompts, dtype=torch.long).transpose(0, 1).to(device)
            Y = torch.tensor(padded_answers, dtype=torch.long).transpose(0, 1).to(device)
            
            input_tensor = X
            for _ in range(max_ans_len):
                output, _, _ = model(input_tensor)
                token = torch.argmax(output[-1, :, :], -1).view(1, -1)
                input_tensor = torch.cat((input_tensor, token), 0)
                
            pred_answers = input_tensor[max_prompt_len:max_prompt_len+max_ans_len, :]
            
            mask = Y != tokenizer.token_to_id['[PAD]']
            equality = pred_answers == Y
            
            correct_digit += torch.logical_and(mask, equality).sum().item()
            total_digit += mask.sum().item()
            
            seq_correct = torch.all(torch.logical_or(~mask, equality), dim=0)
            correct_seq += seq_correct.sum().item()
            total_seq += Y.shape[1]

    return correct_digit / total_digit if total_digit > 0 else 0, correct_seq / total_seq if total_seq > 0 else 0

if __name__ == "__main__":
    with open("assets/dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    
    data_train = dataset["train"]
    data_val = dataset["val"]
    
    tokenizer = character_level_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TransformerModel(ntoken=tokenizer.ntokens, ninp=128, nhead=16, nhid=64, nlayers=6).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id['[PAD]'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    batch_size = 100
    epochs = 15
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(data_train), batch_size):
            X, Y, length_prompts, length_answers = get_batch_lm(data_train, i, batch_size, tokenizer)
            X, Y = X.to(device), Y.to(device)
            input_tensor = torch.cat((X, Y), 0) 
            
            optimizer.zero_grad()
            output, _, _ = model(input_tensor) 
            
            output_answers = output[length_prompts-1:-1, :, :].reshape(-1, tokenizer.ntokens)
            target_answers = Y.reshape(-1)
            
            loss = criterion(output_answers, target_answers)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        train_loss = total_loss / (len(data_train) / batch_size)
        val_digit_acc, val_seq_acc = evaluate_lm(model, data_val, tokenizer, batch_size, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Val Digit Acc: {val_digit_acc:.4f}, Val Seq Acc: {val_seq_acc:.4f}")

    torch.save(model.state_dict(), "assets/reverse_model.pth")
    print("Saved reverse_model.pth")
