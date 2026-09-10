import random
import pickle
import os

from utils import sample_datapoint


# set a fixed seed for reproducible data generation
random.seed(42)

# maximum number of digits for each operand
number_digits = 3

# set the dataset sizes for each operation
train_size_per_operation = 50000
val_size_per_operation = 10000
test_size_per_operation = 10000

# calculate the total number of samples for each operation
total_size_per_operation = (
    train_size_per_operation
    + val_size_per_operation
    + test_size_per_operation
)


def generate_unique_data(operation, total_size, used_prompts):
    # generate unique samples for the given operation
    data = []

    while len(data) < total_size:
        sample = sample_datapoint(
            number_digits=number_digits,
            operation=operation
        )

        prompt, answer = sample

        # add the sample if the prompt has not been generated before
        if prompt not in used_prompts:
            data.append(sample)
            used_prompts.add(prompt)

    return data


# track all generated prompts to prevent duplicates
used_prompts = set()


# generate addition samples
addition_data = generate_unique_data(
    operation="+",
    total_size=total_size_per_operation,
    used_prompts=used_prompts
)

# generate subtraction samples
subtraction_data = generate_unique_data(
    operation="-",
    total_size=total_size_per_operation,
    used_prompts=used_prompts
)


# split addition data into train, validation, and test sets
addition_train = addition_data[:train_size_per_operation]

addition_val = addition_data[
    train_size_per_operation:
    train_size_per_operation + val_size_per_operation
]

addition_test = addition_data[
    train_size_per_operation + val_size_per_operation:
]


# split subtraction data into train, validation, and test sets
subtraction_train = subtraction_data[:train_size_per_operation]

subtraction_val = subtraction_data[
    train_size_per_operation:
    train_size_per_operation + val_size_per_operation
]

subtraction_test = subtraction_data[
    train_size_per_operation + val_size_per_operation:
]


# combine addition and subtraction data
data_train = addition_train + subtraction_train
data_val = addition_val + subtraction_val
data_test = addition_test + subtraction_test


# shuffle each split to mix addition and subtraction samples
random.shuffle(data_train)
random.shuffle(data_val)
random.shuffle(data_test)


# save the dataset with original answers
dataset = {
    "train": data_train,
    "val": data_val,
    "test": data_test
}

# create the assets folder if it does not exist
os.makedirs("assets", exist_ok=True)

with open("assets/dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)


# check the dataset sizes and operation balance
for split_name, data in dataset.items():
    addition_count = sum(1 for prompt, answer in data if "+" in prompt)
    subtraction_count = sum(1 for prompt, answer in data if "-" in prompt)

    total = len(data)

    print(f"\n{split_name.capitalize()}: {total:,}")
    print(f"Addition:    {addition_count:,} ({addition_count / total:.1%})")
    print(f"Subtraction: {subtraction_count:,} ({subtraction_count / total:.1%})")


# check for overlapping samples between splits
print("\nTrain overlap Val:", len(set(data_train) & set(data_val)))
print("Train overlap Test:", len(set(data_train) & set(data_test)))
print("Val overlap Test:", len(set(data_val) & set(data_test)))

print("\nDataset saved to assets/dataset.pkl")