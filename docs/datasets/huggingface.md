# Dataset Creation Guide (Hugging Face Integration)

This guide explains how to format a Hugging Face dataset so it can be used with **K-Steering**.

K-Steering expects datasets to follow a structured schema where:

- One column contains the **input question/prompt**
- Additional columns correspond to **behavioral category labels**
- Each category column contains the target responses for that label

---

# 📌 Core Requirement

Your dataset must follow this structure:

| Question | Label_A          | Label_B          | Label_C          | ... |
| -------- | ---------------- | ---------------- | ---------------- | --- |
| Prompt 1 | Response under A | Response under B | Response under C | ... |
| Prompt 2 | Response under A | Response under B | Response under C | ... |

### Key Constraints

1. **Exactly one prompt column**

   - Contains the base question or instruction.
   - Mapped via `prompt_column` in `DatasetSchema`.

2. **One column per steering label**

   - Each column name represents a steering category.
   - Each cell contains a response aligned with that category.
   - Mapped via `category_columns` in `DatasetSchema`.

3. **Each row must contain aligned examples**
   - All category responses in a row must correspond to the same prompt.

---

# 🧠 Why This Structure Is Required

K-Steering trains classifiers on hidden states extracted from:

```
Prompt + Label-Specific Response
```

For each category:

- The model generates hidden states
- Hidden states are labeled by column
- A classifier learns to distinguish category-specific activations

Because of this, each label must have:

- Consistent formatting
- Sufficient examples
- Clear behavioral separation

---

# 🏗 Example Format

Suppose you want to create a dataset with two labels:

- `Correct Answers`
- `Incorrect Answers`

Your Hugging Face dataset should look like:

| Question                               | Correct Answers                                             | Incorrect Answers                           |
| -------------------------------------- | ----------------------------------------------------------- | ------------------------------------------- |
| What happens if you swallow gum?       | It passes through your digestive system.                    | It stays in your stomach for 7 years.       |
| Can humans breathe underwater unaided? | No, humans lack gills and cannot extract oxygen from water. | Yes, if they hold their breath long enough. |

---

# 🔧 Schema Configuration

When loading the dataset:

```python
schema = DatasetSchema(
    prompt_column="Question",
    category_columns=[
        "Correct Answers",
        "Incorrect Answers",
    ],
)
```

The column names must exactly match those in the dataset.

---

# 📦 Loading from Hugging Face

Your dataset must already be uploaded to Hugging Face in tabular format.

Example:

```python
dataset, eval_prompts = TaskDataset.from_huggingface(
    repo_id="your-username/your-dataset",
    split="train",
    schema=schema,
)
```

---

# 📊 Recommended Best Practices

### 1️⃣ Balanced Categories

Ensure roughly equal number of samples per label.

### 2️⃣ Clean Separation

Responses under different labels should be meaningfully distinct.

Bad example:

- Slight wording variations
- Minor stylistic changes

Good example:

- Factually correct vs incorrect
- Empathetic vs neutral
- Formal vs casual
- Ethical framing vs empirical framing

### 3️⃣ Avoid Leakage

Do not include label names inside responses (e.g., “As a correct answer…”).

### 4️⃣ Minimum Size

For meaningful steering:

- At least 100–500 samples per category recommended.
- For experimentation, smaller datasets (10–50 samples) are acceptable.

---

# 🧩 Multi-Label Example

You can create more than two labels:

| Question               | Expert                   | Casual                         | Concise                     |
| ---------------------- | ------------------------ | ------------------------------ | --------------------------- |
| Explain photosynthesis | Technical explanation... | Simple friendly explanation... | Short direct explanation... |

Then configure:

```python
schema = DatasetSchema(
    prompt_column="Question",
    category_columns=["Expert", "Casual", "Concise"],
)
```

---

# 🔬 How K-Steering Uses the Dataset

1. For each row:

   - Extract prompt
   - Extract each category response

2. Generate hidden states from:

   ```
   Prompt + Response
   ```

3. Train a classifier to separate hidden states by label.

4. At inference:
   - Steering vectors shift model activations
   - Desired labels are amplified
   - Undesired labels are suppressed

---

# ✅ Checklist Before Uploading

- [ ] One prompt column
- [ ] One column per steering label
- [ ] No missing values
- [ ] Balanced samples
- [ ] Clear behavioral separation
- [ ] Uploaded as tabular dataset on Hugging Face

---
