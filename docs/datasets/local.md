# Using Local Datasets with K-Steering

In addition to Hugging Face datasets, K-Steering supports loading datasets directly from local files or in-memory data structures.

You can create datasets using:

- `TaskDataset.from_csv(...)`
- `TaskDataset.from_json(...)`
- `TaskDataset.from_dataframe(...)`

The required format remains the same:

> **One prompt column + one column per steering label**

---

# 📌 Required Dataset Structure

Your dataset must follow this structure:

| Question | Label_A     | Label_B     | Label_C     |
| -------- | ----------- | ----------- | ----------- |
| Prompt 1 | Response A1 | Response B1 | Response C1 |
| Prompt 2 | Response A2 | Response B2 | Response C2 |

### Requirements

- Exactly **one prompt column**
- One column per **steering label**
- Each row must align prompt with all label responses
- No missing values

---

# 🧾 Option 1: Load from CSV

## Example CSV File (`my_dataset.csv`)

```csv
Question,Expert,Casual
What is photosynthesis?,Photosynthesis is a biochemical process...,It's how plants make food from sunlight.
What is gravity?,Gravity is a fundamental interaction...,It's what keeps us stuck to the ground.
```

## Loading the CSV

```python
from k_steering.steering.dataset import DatasetSchema, TaskDataset

schema = DatasetSchema(
    prompt_column="Question",
    category_columns=["Expert", "Casual"],
)

dataset, eval_prompts = TaskDataset.from_csv(
    path="my_dataset.csv",
    schema=schema,
)
```

---

# 🧾 Option 2: Load from JSON

## Example JSON File (`my_dataset.json`)

```json
[
  {
    "Question": "What is photosynthesis?",
    "Expert": "Photosynthesis is a biochemical process...",
    "Casual": "It's how plants make food from sunlight."
  },
  {
    "Question": "What is gravity?",
    "Expert": "Gravity is a fundamental interaction...",
    "Casual": "It's what keeps us stuck to the ground."
  }
]
```

## Loading the JSON

```python
schema = DatasetSchema(
    prompt_column="Question",
    category_columns=["Expert", "Casual"],
)

dataset, eval_prompts = TaskDataset.from_json(
    path="my_dataset.json",
    schema=schema,
)
```

---

# 🧾 Option 3: Load from a Pandas DataFrame

If your data is already loaded into memory:

```python
import pandas as pd
from k_steering.steering.dataset import DatasetSchema, TaskDataset

df = pd.DataFrame({
    "Question": [
        "What is photosynthesis?",
        "What is gravity?"
    ],
    "Expert": [
        "Photosynthesis is a biochemical process...",
        "Gravity is a fundamental interaction..."
    ],
    "Casual": [
        "It's how plants make food from sunlight.",
        "It's what keeps us stuck to the ground."
    ]
})

schema = DatasetSchema(
    prompt_column="Question",
    category_columns=["Expert", "Casual"],
)

dataset, eval_prompts = TaskDataset.from_dataframe(
    df=df,
    schema=schema,
)
```

---

# 🧠 How Local Datasets Are Used

For each row:

1. Extract prompt from `prompt_column`
2. Extract responses from each `category_column`
3. Generate hidden states from:
   ```
   Prompt + Category Response
   ```
4. Train classifiers to distinguish category-specific activations

The loading source (CSV, JSON, DataFrame) does not change the steering pipeline — only how the data enters the system.

---

# 📊 Best Practices

### 1️⃣ Clean Column Names

Column names must exactly match those provided in `DatasetSchema`.

### 2️⃣ No Missing Values

Ensure all rows contain valid text for every label.

### 3️⃣ Behavioral Contrast

Categories should be meaningfully distinct.

Good:

- Correct vs Incorrect
- Empathetic vs Formal
- Moral vs Empirical framing

Bad:

- Minor wording differences
- Slight stylistic variation

### 4️⃣ Balanced Labels

Try to maintain similar number of examples per label.

---

# ✅ Pre-Training Checklist

- [ ] One prompt column defined
- [ ] One column per steering label
- [ ] No null values
- [ ] Balanced samples
- [ ] Clear behavioral contrast
- [ ] Schema matches column names exactly

---
