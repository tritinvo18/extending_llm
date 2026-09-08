import pickle
from src.utils import sample_datapoint

if __name__ == "__main__":
    number_digits = 3
    train_size = 50000
    val_size = 10000
    test_size = 10000

    data = set()
    print("Generating unique arithmetic samples...")
    while len(data) < (train_size + val_size + test_size):
        # Store just the equation, we can format target later
        sample = sample_datapoint(number_digits, mode="forward") # returns (prompt, ans)
        data.add(sample)
    
    data = list(data)
    
    data_train = data[:train_size]
    data_val = data[train_size: train_size+val_size]
    data_test = data[train_size+val_size : train_size+val_size+test_size]
    
    with open("assets/dataset.pkl", "wb") as f:
        pickle.dump({"train": data_train, "val": data_val, "test": data_test}, f)
        
    print(f"Dataset generated. Train: {len(data_train)}, Val: {len(data_val)}, Test: {len(data_test)}")
