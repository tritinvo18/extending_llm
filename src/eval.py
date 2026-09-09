import torch
import torch.nn as nn
from utils import character_level_tokenizer, TransformerModel
import pickle
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# Evaluates either the forward or reverse model on the testing dataset
def evaluate_test(model, data, tokenizer, batch_size, device, mode="forward"):
    model.eval()
    results = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size): 
            batch = data[i:i+batch_size]
            prompts = [tokenizer.encode(p) for p, a_fw, a_rev in batch]
            max_prompt_len = max(len(p) for p in prompts)
            
            padded_prompts = [[tokenizer.token_to_id['[PAD]']] * (max_prompt_len - len(p)) + p for p in prompts]
            X = torch.tensor(padded_prompts, dtype=torch.long).transpose(0, 1).to(device)
            
            input_tensor = X
            # Generate up to 15 tokens to ensure intermediate algorithmic steps are fully captured
            for _ in range(15):
                output, _, _ = model(input_tensor)
                token = torch.argmax(output[-1, :, :], -1).view(1, -1)
                input_tensor = torch.cat((input_tensor, token), 0)
                
            pred_answers = input_tensor[max_prompt_len:, :]
            pred_answers = pred_answers.transpose(0, 1).cpu().tolist() 
            
            for b_idx, (p, a_fw, a_rev) in enumerate(batch):
                pred_tokens = pred_answers[b_idx]
                if tokenizer.token_to_id['[EOS]'] in pred_tokens:
                    eos_idx = pred_tokens.index(tokenizer.token_to_id['[EOS]'])
                    pred_tokens = pred_tokens[:eos_idx]
                
                pred_str = tokenizer.decode(pred_tokens)
                raw_pred = pred_str
                
                # Fetch appropriate true answer based on evaluation mode
                ans_str = a_fw if mode == 'forward' else a_rev
                
                op = "+" if "+" in p else "-"
                
                # Tag specific mathematical edge cases for deeper error analysis
                crossing_zero = (op == "-" and "-" in a_fw)
                ops_parts = p.replace("=","").split(op)
                asymmetric = len(ops_parts[0]) != len(ops_parts[1])
                
                results.append({
                    "prompt": p,
                    "true_ans": ans_str,
                    "pred_ans": pred_str,
                    "raw_pred": raw_pred,
                    "op": op,
                    "crossing_zero": crossing_zero,
                    "asymmetric": asymmetric,
                    "correct": pred_str == ans_str
                })
    return results

# Computes accuracy on a per-digit basis
def compute_digit_accuracy(results, is_reverse=False):
    pos_correct = defaultdict(int)
    pos_total = defaultdict(int)
    
    for r in results:
        t = r["true_ans"]
        p = r["pred_ans"]
        
        if is_reverse:
            # Clean intermediate steps and reverse back to standard forward numeric format
            t = t.replace("c1", "").replace("b1", "").replace("c", "").replace("b", "")[::-1]
            p = p.replace("c1", "").replace("b1", "").replace("c", "").replace("b", "")[::-1]
            
        # Pad strings to the same length with leading spaces to align positional digits correctly
        max_len = max(len(t), len(p))
        t = t.rjust(max_len)
        p = p.rjust(max_len)
        
        # Reverse them so index 0 is units, index 1 is tens, etc.
        t_rev = t[::-1]
        p_rev = p[::-1]
        
        for i in range(len(t_rev)):
            if t_rev[i].strip() == "": continue # Skip padding comparisons
            pos_total[i] += 1
            if i < len(p_rev) and t_rev[i] == p_rev[i]:
                pos_correct[i] += 1
                
    accs = []
    # Generate list of accuracies for as many digit positions as exist in the batch
    for i in range(max(pos_total.keys()) + 1 if pos_total else 0):
        if pos_total[i] > 0:
            accs.append(pos_correct[i] / pos_total[i])
        else:
            accs.append(0)
    return accs


if __name__ == "__main__":
    if not os.path.exists("plot"):
        os.makedirs("plot")
        
    with open("assets/dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
        
    data_test = dataset["test"]
    tokenizer = character_level_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 500

    print("Loading models for centralized evaluation...")
    model_fw = TransformerModel(ntoken=tokenizer.ntokens, ninp=128, nhead=16, nhid=64, nlayers=6).to(device)
    model_fw.load_state_dict(torch.load("assets/forward_model.pth", map_location=device, weights_only=True))

    model_rev = TransformerModel(ntoken=tokenizer.ntokens, ninp=128, nhead=16, nhid=64, nlayers=6).to(device)
    model_rev.load_state_dict(torch.load("assets/reverse_model.pth", map_location=device, weights_only=True))

    print("Evaluating Forward Baseline...")
    res_fw = evaluate_test(model_fw, data_test, tokenizer, batch_size, device, mode="forward")

    print("Evaluating Algorithmic Reverse Model...")
    res_rev = evaluate_test(model_rev, data_test, tokenizer, batch_size, device, mode="reverse")

    # Aggregate performance metrics
    fw_overall = sum([1 for r in res_fw if r["correct"]]) / len(res_fw)
    rev_overall = sum([1 for r in res_rev if r["correct"]]) / len(res_rev)
    print(f"\nFinal Test Set Overall Accuracy -> Forward: {fw_overall*100:.2f}%, Reverse: {rev_overall*100:.2f}%")
