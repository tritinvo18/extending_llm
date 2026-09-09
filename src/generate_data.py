import pickle
from utils import sample_datapoint
import os

if __name__ == "__main__":
    # Configuration parameters for dataset generation
    number_digits = 3
    train_size = 100000
    val_size = 20000
    test_size = 20000

    data = set()
    print("Generating unique arithmetic samples...")
    total_size = train_size + val_size + test_size
    
    # Toggle force_neg to ensure an even distribution of negative-result subtractions
    force_neg = False
    while len(data) < total_size:
        # Each sample is a tuple: (prompt, forward_target, reverse_target_with_steps)
        sample = sample_datapoint(number_digits, force_negative=force_neg)
        data.add(sample)
        force_neg = not force_neg
    
    # Convert set back to list to allow slicing
    data = list(data)
    
    # Split dataset precisely according to configuration sizes
    data_train = data[:train_size]
    data_val = data[train_size: train_size+val_size]
    data_test = data[train_size+val_size : train_size+val_size+test_size]
    
    # Ensure assets directory exists before saving
    if not os.path.exists("assets"):
        os.makedirs("assets")
        
    with open("assets/dataset.pkl", "wb") as f:
        # Save as a consolidated dictionary to easily retrieve splits later
        pickle.dump({"train": data_train, "val": data_val, "test": data_test}, f)
        
    print(f"Dataset generated. Train: {len(data_train)}, Val: {len(data_val)}, Test: {len(data_test)}")
