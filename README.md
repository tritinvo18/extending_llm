# Extending addition LLM

In 7.3 Tutorial, you learned how to implement a large language model (LLM) that can perform addition, exploring how it predicts each digit sequentially from left to right. In this assessment, you are invited to take that knowledge a step further: you will extend the workshop code so that your model can learn subtraction, another fundamental arithmetic operation. But we won’t stop there. Arithmetic is often performed manually from right to left to handle carries and borrows efficiently, and this idea inspires our next challenge. You will implement a reverse-mode version of your model, where the model predicts the digits in the opposite order.

Finally, you will conduct a comparative analysis to understand how these two prediction paradigms (Forward and Reverse) affect the model’s learning. By examining digit-level performance, error patterns, and robustness across addition and subtraction, you will uncover where the model succeeds, where it struggles, and what these behaviours reveal about how LLMs learn algorithmic tasks.

## Task details

Work through the following tasks to complete this challenge:

### Task 1: Handling subtraction

The objective of this task is to modify our code to enable the model designed for performing addition to also execute subtraction operations. Specifically, the model should also handle operations of the format a−b=c, where a and b are non-negative integers with a maximum of three digits. The model must be able to output negative results (e.g., -123) when b>a. We will not consider operations of the format -2+3 or -2-3. Note that due to the higher complexity, more epochs may be needed to train your model.  It is therefore important to have a validation dataset to check convergence.

### Task 2: Reverse-prediction implementation

In the workshop, the LLM predicts tokens from left to right, whereas manual arithmetic is typically performed right to left in order to handle carries and borrows. The objective of this task is to implement and train a reverse-mode version of your Addition + Subtraction LLM to investigate whether reversing the output alignment improves algorithmic learning. For example:

- Forward mode (left to right): 12 + 19 = 31: The model predicts first 3, then 1.

- Reverse mode (right to left): The target sequence is reversed: 13. The model first predicts 1, then 3, corresponding to the operations performed from the rightmost digits first (2 + 9, then 1 + 1 + carry).

### Task 3: Evaluation and comparative analysis

Once your models are trained, conduct a formal evaluation to compare the Forward and Reverse prediction paradigms across both addition and subtraction. Use a single large (>=10k samples) held-out test set containing a balanced mix of addition and subtraction examples, and evaluate both models on the same test examples. The objective is to determine whether both operations can be learned robustly and how the prediction direction affects performance. Your analysis should go beyond reporting a single overall accuracy and investigate where and why errors occur. In particular, you should consider:

- Operation robustness: Compare how well the model learns addition vs. subtraction in both Forward and Reverse modes.

- Digit-level performance: Measure prediction accuracy for different digit positions in the result (e.g., units, tens, hundreds, and thousands) for both operations and modes.

- Direction effects: Analyse whether prediction accuracy changes when switching between Forward and Reverse modes.

- Error patterns: Identify situations where the model fails more frequently, such as specific digit positions, carry/borrow operations, or longer numerical strings.

As a general performance expectation, both successfully trained model should achieve overall accuracy 78% or higher on the held-out test set.

Overall  accuracy:

$$Acc_{overall} = \frac{1}{N} \sum_{i=1}^N l(\hat y_i = y_i)$$

where $N$ is the number of test examples. The function $l(.)$ is 1 when the condition is true and 0 otherwise. $y_i$ is the ground-truth result for test example $i$, and $\hat y_i$ is the predicted numeric result.

## Submission Instructions

### Report

You will submit a professional report in PDF format (max 2 pages) on this page. The report should be written with technical precision and clarity and must include:

- Method description: A short explanation of the conceptual changes required to accomplish Tasks 1 and 2. Do not include code blocks; the implementation will be evaluated in the notebook. Instead, focus on explaining the key modifications and why they were necessary.

- Results and analysis: Present and analyse the results obtained in Task 3, including, at a minimum, comparisons between the Forward and Reverse prediction modes and performance across different digit positions, operations, and carry/borrow cases. More thorough and insightful analysis will be rewarded.

### Project Code

You will submit a Zipped folder named project5_code.zip on this page. This folder must include:

- LLMForward.ipynb: A notebook containing the full implementation of the LLM in Forward mode, supporting both addition and subtraction.

- LLMReverse.ipynb A notebook containing the full implementation of the LLM in Reverse mode, supporting both addition and subtraction.

- main_report.ipynb: A notebook that reproduces all numerical results and plots reported for Task 3. Include all files and code required for reproducibility, including the held-out test set (for instance, a .pkl or .npz file), model setup, trained weights (.pth), and evaluation code. Do not include the training loop. This notebook will be run for grading; if any cell fails, the code evaluation will receive 0 marks.

Please adhere to the structure of the original LLM workshop code as closely as possible and ensure compatibility with the IFN680 computing environment.

## Supporting resouces

- 7.3 Tutorial Notebook
