import torch
import torch.nn as nn
from src.utils import character_level_tokenizer, TransformerModel
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
            for _ in range(6):
                output, _, _ = model(input_tensor)
                token = torch.argmax(output[-1, :, :], -1).view(1, -1)
                input_tensor = torch.cat((input_tensor, token), 0)
                
            pred_answers = input_tensor[max_prompt_len:, :]
            pred_answers = pred_answers.transpose(0, 1).cpu().tolist() 
            
            for b_idx, (p, a) in enumerate(batch):
                pred_tokens = pred_answers[b_idx]
                if tokenizer.token_to_id['[EOS]'] in pred_tokens:
                    eos_idx = pred_tokens.index(tokenizer.token_to_id['[EOS]'])
                    pred_tokens = pred_tokens[:eos_idx]
                
                pred_str = tokenizer.decode(pred_tokens)
                raw_pred = pred_str
                if mode == "reverse":
                    pred_str = pred_str[::-1]
                
                op = "+" if "+" in p else "-"
                
                # Check for crossing zero (e.g., negative answers in subtraction)
                crossing_zero = (op == "-" and "-" in a)
                
                # Check for asymmetric lengths (e.g. 100 - 9)
                ops_parts = p.replace("=","").split(op)
                asymmetric = len(ops_parts[0]) != len(ops_parts[1])
                
                results.append({
                    "prompt": p,
                    "true_ans": a,
                    "pred_ans": pred_str,
                    "raw_pred": raw_pred,
                    "op": op,
                    "crossing_zero": crossing_zero,
                    "asymmetric": asymmetric,
                    "correct": pred_str == a
                })
    return results

def compute_digit_accuracy(results):
    pos_correct = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    pos_total = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    for r in results:
        t = r["true_ans"]
        p = r["pred_ans"]
        p = p.rjust(len(t)) if len(p) < len(t) else p
        
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
        
    with open("assets/dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
        
    data_test = dataset["test"]
    tokenizer = character_level_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Models
    model_fw = TransformerModel(ntoken=tokenizer.ntokens, ninp=128, nhead=16, nhid=64, nlayers=6).to(device)
    model_fw.load_state_dict(torch.load("assets/forward_model.pth"))
    
    model_rev = TransformerModel(ntoken=tokenizer.ntokens, ninp=128, nhead=16, nhid=64, nlayers=6).to(device)
    model_rev.load_state_dict(torch.load("assets/reverse_model.pth"))
    
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

    # 2.5 Error Analysis (Crossing Zero & Asymmetric Lengths)
    import pandas as pd
    import seaborn as sns
    
    errors_fw = [r for r in res_fw if not r["correct"]]
    errors_rev = [r for r in res_rev if not r["correct"]]
    
    fw_neg_total = sum([1 for r in res_fw if r.get("crossing_zero", False)])
    fw_neg_errors = sum([1 for r in errors_fw if r.get("crossing_zero", False)])
    rev_neg_total = sum([1 for r in res_rev if r.get("crossing_zero", False)])
    rev_neg_errors = sum([1 for r in errors_rev if r.get("crossing_zero", False)])
    
    fw_neg_error_rate = fw_neg_errors / max(1, fw_neg_total)
    rev_neg_error_rate = rev_neg_errors / max(1, rev_neg_total)
    
    fw_asym_total = sum([1 for r in res_fw if r.get("asymmetric", False)])
    fw_asym_errors = sum([1 for r in errors_fw if r.get("asymmetric", False)])
    rev_asym_total = sum([1 for r in res_rev if r.get("asymmetric", False)])
    rev_asym_errors = sum([1 for r in errors_rev if r.get("asymmetric", False)])
    
    fw_asym_error_rate = fw_asym_errors / max(1, fw_asym_total)
    rev_asym_error_rate = rev_asym_errors / max(1, rev_asym_total)
    
    data = {
        "Model": ["Forward", "Forward", "Reverse", "Reverse"],
        "Error Type": ["Negative Output", "Asymmetric Lengths", "Negative Output", "Asymmetric Lengths"],
        "Error Rate": [fw_neg_error_rate, fw_asym_error_rate, rev_neg_error_rate, rev_asym_error_rate]
    }
    df_errors = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 4))
    sns.barplot(x="Error Type", y="Error Rate", hue="Model", data=df_errors)
    plt.title("Error Rate Patterns (OOD & Edge Cases)")
    plt.ylabel("Error Rate")
    plt.savefig("plot/error_rates.png")
    plt.close()
    
    # Print Error Samples
    fw_error_samples = pd.DataFrame(errors_fw).head(5)
    rev_error_samples = pd.DataFrame(errors_rev).head(5)
    print("\nForward Model - Sample Errors:")
    if not fw_error_samples.empty:
        print(fw_error_samples[["prompt", "true_ans", "pred_ans", "op", "crossing_zero", "asymmetric"]].to_string())
    print("\nReverse Model - Sample Errors:")
    if not rev_error_samples.empty:
        print(rev_error_samples[["prompt", "true_ans", "pred_ans", "op", "crossing_zero", "asymmetric"]].to_string())


    # 2.6 Target vs Predicted Scatter Plot
    def get_xy(results):
        x_correct, y_correct = [], []
        x_incorrect, y_incorrect = [], []
        for r in results:
            try:
                t = int(r["true_ans"])
                p = int(r["pred_ans"])
                if r["correct"]:
                    x_correct.append(t)
                    y_correct.append(p)
                else:
                    x_incorrect.append(t)
                    y_incorrect.append(p)
            except ValueError:
                pass
        return x_correct, y_correct, x_incorrect, y_incorrect

    fw_xc, fw_yc, fw_xi, fw_yi = get_xy(res_fw)
    rev_xc, rev_yc, rev_xi, rev_yi = get_xy(res_rev)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(fw_xc, fw_yc, color='blue', alpha=0.1, label='Correct')
    axes[0].scatter(fw_xi, fw_yi, color='red', alpha=0.5, label='Incorrect')
    axes[0].plot([-2000, 2000], [-2000, 2000], 'k--', alpha=0.5)
    axes[0].set_title('Forward Model: Target vs Predicted')
    axes[0].set_xlabel('Target Number')
    axes[0].set_ylabel('Predicted Number')
    axes[0].legend()

    axes[1].scatter(rev_xc, rev_yc, color='blue', alpha=0.1, label='Correct')
    axes[1].scatter(rev_xi, rev_yi, color='red', alpha=0.5, label='Incorrect')
    axes[1].plot([-2000, 2000], [-2000, 2000], 'k--', alpha=0.5)
    axes[1].set_title('Reverse Model: Target vs Predicted')
    axes[1].set_xlabel('Target Number')
    axes[1].set_ylabel('Predicted Number')
    axes[1].legend()

    plt.savefig("plot/target_vs_predicted.png")
    plt.close()

    # 2.7 Attention Maps
    def get_attention_and_tokens(model, p, mode):
        padded_prompts = [[tokenizer.token_to_id['[PAD]']] * (8 - len(p)) + tokenizer.encode(p)]
        X = torch.tensor(padded_prompts, dtype=torch.long).transpose(0, 1).to(device)
        
        input_tensor = X
        for _ in range(6):
            output, _, attns = model(input_tensor)
            token = torch.argmax(output[-1, :, :], -1).view(1, -1)
            input_tensor = torch.cat((input_tensor, token), 0)
            
        _, _, attns = model(input_tensor)
        
        tokens = input_tensor[:, 0].tolist()
        tokens_str = [tokenizer.id_to_token[t] for t in tokens]
        
        n_tokens = len(tokens_str)
        attn_avg = torch.zeros(n_tokens, n_tokens)
        for layer_attn in attns:
            attn_avg += layer_attn[0].mean(dim=0).cpu()
        attn_avg /= len(attns)
        
        return attn_avg.detach().numpy(), tokens_str

    fw_attn, fw_tokens = get_attention_and_tokens(model_fw, "119+809=", "forward")
    rev_attn, rev_tokens = get_attention_and_tokens(model_rev, "119+809=", "reverse")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(fw_attn, xticklabels=fw_tokens, yticklabels=fw_tokens, cmap="viridis", ax=axes[0])
    axes[0].set_title("Forward Model Attention (119+809=)")
    sns.heatmap(rev_attn, xticklabels=rev_tokens, yticklabels=rev_tokens, cmap="viridis", ax=axes[1])
    axes[1].set_title("Reverse Model Attention (119+809=)")
    plt.tight_layout()
    plt.savefig("plot/attention_maps.png")
    plt.close()

    # 2.8 Standard Inference Examples
    std_prompts = [
        "12+34=", "809+119=", "514-513=", "408-730=", "180-185=",
        "999+1=", "100-99=", "555+445=", "123-456=", "0+0=",
        "8-12=", "77+33=", "456+789=", "999-999=", "10+990="
    ]

    print("\nStandard Inference Examples:")
    print(f"{'Prompt':<15} | {'Expected':<15} | {'Forward Pred':<15} | {'Reverse Pred':<15}")
    print("-" * 68)

    for p in std_prompts:
        padded_prompts = [[tokenizer.token_to_id['[PAD]']] * (8 - len(p)) + tokenizer.encode(p)]
        X = torch.tensor(padded_prompts, dtype=torch.long).transpose(0, 1).to(device)
        
        # Forward Model
        input_tensor_fw = X
        for _ in range(6):
            output, _, _ = model_fw(input_tensor_fw)
            token = torch.argmax(output[-1, :, :], -1).view(1, -1)
            input_tensor_fw = torch.cat((input_tensor_fw, token), 0)
            
        pred_tokens_fw = input_tensor_fw[8:, 0].tolist()
        if tokenizer.token_to_id['[EOS]'] in pred_tokens_fw:
            pred_tokens_fw = pred_tokens_fw[:pred_tokens_fw.index(tokenizer.token_to_id['[EOS]'])]
        pred_str_fw = tokenizer.decode(pred_tokens_fw)
        
        # Reverse Model
        input_tensor_rev = X
        for _ in range(6):
            output, _, _ = model_rev(input_tensor_rev)
            token = torch.argmax(output[-1, :, :], -1).view(1, -1)
            input_tensor_rev = torch.cat((input_tensor_rev, token), 0)
            
        pred_tokens_rev = input_tensor_rev[8:, 0].tolist()
        if tokenizer.token_to_id['[EOS]'] in pred_tokens_rev:
            pred_tokens_rev = pred_tokens_rev[:pred_tokens_rev.index(tokenizer.token_to_id['[EOS]'])]
        pred_str_rev = tokenizer.decode(pred_tokens_rev)[::-1]
        
        true_ans = str(eval(p[:-1]))
        print(f"{p:<15} | {true_ans:<15} | {pred_str_fw:<15} | {pred_str_rev:<15}")

    # 3. Out of Distribution (Length Generalization) Testing
    ood_prompts = [
        ("1000+2000=", "3000"),
        ("9999+1=", "10000"),
        ("12345+54321=", "66666"),
        ("5000-1000=", "4000"),
        ("9999-9999=", "0"),
        ("1111+2222=", "3333"),
        ("8888-4444=", "4444"),
        ("10000+20000=", "30000"),
        ("7777+3333=", "11110"),
        ("50000-1=", "49999")
    ]
    
    print("\nOut of Distribution (Length Generalization) Testing:")
    print(f"{'Prompt':<15} | {'Expected':<15} | {'Forward Pred':<15} | {'Reverse Pred':<15}")
    print("-" * 68)
    
    for p, true_ans in ood_prompts:
        padded_prompts = [[tokenizer.token_to_id['[PAD]']] * (15 - len(p)) + tokenizer.encode(p)]
        X = torch.tensor(padded_prompts, dtype=torch.long).transpose(0, 1).to(device)
        
        # Forward Model
        input_tensor_fw = X
        for _ in range(6):
            output, _, _ = model_fw(input_tensor_fw)
            token = torch.argmax(output[-1, :, :], -1).view(1, -1)
            input_tensor_fw = torch.cat((input_tensor_fw, token), 0)
            
        pred_tokens_fw = input_tensor_fw[15:, 0].tolist()
        if tokenizer.token_to_id['[EOS]'] in pred_tokens_fw:
            pred_tokens_fw = pred_tokens_fw[:pred_tokens_fw.index(tokenizer.token_to_id['[EOS]'])]
        pred_str_fw = tokenizer.decode(pred_tokens_fw)
        
        # Reverse Model
        input_tensor_rev = X
        for _ in range(6):
            output, _, _ = model_rev(input_tensor_rev)
            token = torch.argmax(output[-1, :, :], -1).view(1, -1)
            input_tensor_rev = torch.cat((input_tensor_rev, token), 0)
            
        pred_tokens_rev = input_tensor_rev[15:, 0].tolist()
        if tokenizer.token_to_id['[EOS]'] in pred_tokens_rev:
            pred_tokens_rev = pred_tokens_rev[:pred_tokens_rev.index(tokenizer.token_to_id['[EOS]'])]
        pred_str_rev = tokenizer.decode(pred_tokens_rev)[::-1]
        
        print(f"{p:<15} | {true_ans:<15} | {pred_str_fw:<15} | {pred_str_rev:<15}")
