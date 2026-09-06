import torch
import torch.nn as nn
from utils import character_level_tokenizer, TransformerModel
import pickle
import os
import matplotlib.pyplot as plt

def evaluate_test(model, data, tokenizer, batch_size, device, mode="forward"):
    model.eval()
    results = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size): 
            batch = data[i:i+batch_size]
            prompts = [tokenizer.encode(p) for p, a in batch]
            max_prompt_len = max(len(p) for p in prompts)
            
            padded_prompts = [[tokenizer.token_to_id['[PAD]']] * (max_prompt_len - len(p)) + p for p in prompts]
            X = torch.tensor(padded_prompts, dtype=torch.long).transpose(0, 1).to(device)
            
            input_tensor = X
            # max possible answer length is 5 (e.g. -1998, 1998 -> 5 chars max: "-", "1", "9", "9", "8", plus EOS)
            # we generate up to 6 tokens
            for _ in range(6):
                output, _, _ = model(input_tensor)
                token = torch.argmax(output[-1, :, :], -1).view(1, -1)
                input_tensor = torch.cat((input_tensor, token), 0)
                
            pred_answers = input_tensor[max_prompt_len:, :]
            pred_answers = pred_answers.transpose(0, 1).cpu().tolist() # batch x seq_len
            
            for b_idx, (p, a) in enumerate(batch):
                pred_tokens = pred_answers[b_idx]
                if tokenizer.token_to_id['[EOS]'] in pred_tokens:
                    eos_idx = pred_tokens.index(tokenizer.token_to_id['[EOS]'])
                    pred_tokens = pred_tokens[:eos_idx]
                
                pred_str = tokenizer.decode(pred_tokens)
                if mode == "reverse":
                    pred_str = pred_str[::-1]
                
                # Operation
                op = "+" if "+" in p else "-"
                
                results.append({
                    "prompt": p,
                    "true_ans": a,
                    "pred_ans": pred_str,
                    "op": op,
                    "correct": pred_str == a
                })
    return results

def compute_digit_accuracy(results):
    # Align from right to left
    # position 0: units, 1: tens, 2: hundreds, 3: thousands/sign
    pos_correct = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    pos_total = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    for r in results:
        t = r["true_ans"]
        p = r["pred_ans"]
        # pad p to match length of t with spaces if shorter
        p = p.rjust(len(t)) if len(p) < len(t) else p
        
        # compare right to left
        t_rev = t[::-1]
        p_rev = p[::-1]
        for i in range(len(t_rev)):
            pos_total[i] += 1
            if i < len(p_rev) and t_rev[i] == p_rev[i]:
                pos_correct[i] += 1
                
    accs = []
    for i in range(5):
        if pos_total[i] > 0:
            accs.append(pos_correct[i] / pos_total[i])
        else:
            accs.append(0)
    return accs

if __name__ == "__main__":
    if not os.path.exists("plot"):
        os.makedirs("plot")
        
    with open("dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
        
    data_test = dataset["test"]
    tokenizer = character_level_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Models
    model_fw = TransformerModel(ntoken=tokenizer.ntokens, ninp=128, nhead=16, nhid=64, nlayers=6).to(device)
    model_fw.load_state_dict(torch.load("forward_model.pth"))
    
    model_rev = TransformerModel(ntoken=tokenizer.ntokens, ninp=128, nhead=16, nhid=64, nlayers=6).to(device)
    model_rev.load_state_dict(torch.load("reverse_model.pth"))
    
    batch_size = 500
    
    print("Evaluating Forward Model...")
    res_fw = evaluate_test(model_fw, data_test, tokenizer, batch_size, device, mode="forward")
    
    print("Evaluating Reverse Model...")
    res_rev = evaluate_test(model_rev, data_test, tokenizer, batch_size, device, mode="reverse")
    
    # 1. Operation Robustness
    fw_add_acc = sum([1 for r in res_fw if r["op"] == "+" and r["correct"]]) / sum([1 for r in res_fw if r["op"] == "+"])
    fw_sub_acc = sum([1 for r in res_fw if r["op"] == "-" and r["correct"]]) / sum([1 for r in res_fw if r["op"] == "-"])
    
    rev_add_acc = sum([1 for r in res_rev if r["op"] == "+" and r["correct"]]) / sum([1 for r in res_rev if r["op"] == "+"])
    rev_sub_acc = sum([1 for r in res_rev if r["op"] == "-" and r["correct"]]) / sum([1 for r in res_rev if r["op"] == "-"])
    
    labels = ['Addition', 'Subtraction']
    fw_accs = [fw_add_acc, fw_sub_acc]
    rev_accs = [rev_add_acc, rev_sub_acc]
    
    x = [0, 1]
    width = 0.35
    
    fig, ax = plt.subplots()
    ax.bar([i - width/2 for i in x], fw_accs, width, label='Forward')
    ax.bar([i + width/2 for i in x], rev_accs, width, label='Reverse')
    ax.set_ylabel('Accuracy')
    ax.set_title('Operation Robustness (+ vs -)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig("plot/operation_robustness.png")
    plt.close()
    
    # 2. Digit-level performance
    fw_add_res = [r for r in res_fw if r["op"] == "+"]
    fw_sub_res = [r for r in res_fw if r["op"] == "-"]
    rev_add_res = [r for r in res_rev if r["op"] == "+"]
    rev_sub_res = [r for r in res_rev if r["op"] == "-"]
    
    fw_add_dig = compute_digit_accuracy(fw_add_res)
    rev_add_dig = compute_digit_accuracy(rev_add_res)
    fw_sub_dig = compute_digit_accuracy(fw_sub_res)
    rev_sub_dig = compute_digit_accuracy(rev_sub_res)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x_pos = [0, 1, 2, 3, 4]
    x_labels = ["Units", "Tens", "Hundreds", "Thousands", "Sign"]
    
    axes[0].plot(x_pos, fw_add_dig, marker='o', label="Forward")
    axes[0].plot(x_pos, rev_add_dig, marker='s', label="Reverse")
    axes[0].set_title("Digit Accuracy: Addition")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(x_labels)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    
    axes[1].plot(x_pos, fw_sub_dig, marker='o', label="Forward")
    axes[1].plot(x_pos, rev_sub_dig, marker='s', label="Reverse")
    axes[1].set_title("Digit Accuracy: Subtraction")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(x_labels)
    axes[1].legend()
    
    plt.savefig("plot/digit_accuracy.png")
    plt.close()

    print("Plots saved in plot/ directory.")
    fw_overall = sum([1 for r in res_fw if r["correct"]]) / len(res_fw)
    rev_overall = sum([1 for r in res_rev if r["correct"]]) / len(res_rev)
    print(f"Overall Accuracy - Forward: {fw_overall:.4f}, Reverse: {rev_overall:.4f}")
