# 📘 Glossary & Argument Reference

This section documents the core concepts and configuration arguments used throughout the
`k-steering` package. It is intended to serve as a single reference point for understanding
_what each parameter means_, _where it is used_, and _how it affects steering behavior_.

---

# Core Concepts

## **Steering**

A mechanism for influencing a language model’s generation by modifying internal activations
at specific layers, without fine-tuning the base model.

## **Steering Classifier**

A lightweight model trained on hidden states to distinguish between behavioral attributes
(e.g., _Correct vs Incorrect_, _Empirical Grounding vs Straw Man Reframing_).

## **Steering Vector**

A direction in activation space derived from the steering classifier that is added to
model activations during inference.

## **K-Steering**

A framework for composing and applying steering vectors (potentially non-linearly) across different layers.

---

# Configuration Arguments

## `SteeringConfig`

Used to define how steering classifiers are trained, evaluated, and applied.

| Argument       | Type               | Description                                                             |
| -------------- | ------------------ | ----------------------------------------------------------------------- |
| `train_layer`  | `int`              | Layer index whose hidden states are used to train steering classifiers. |
| `steer_layers` | `list[int]`        | Layers where steering vectors are injected during inference.            |
| `eval_layer`   | `int` _(optional)_ | Layer used for evaluation or judging (e.g., `-1` for final layer).      |
| `pos`          | `int` _(optional)_ | Token position used for evaluation (`-1` = last token).                 |

**Example**

```python
SteeringConfig(
    train_layer=1,
    steer_layers=[1, 3],
    eval_layer=-1,
    pos=-1,
)
```

# Model Interface

## `KSteering`

Main entry point for training and applying steering.

| Argument          | Type             | Description                               |
| ----------------- | ---------------- | ----------------------------------------- |
| `model_name`      | `str`            | Hugging Face model identifier.            |
| `steering_config` | `SteeringConfig` | Configuration defining steering behavior. |

# Training Arguments

## `fit(...)`

Trains steering classifiers.
| Argument | Type | Description |
| -------------- | ------------- | ------------------------------------------------------------------ |
| `task` | `str` | Name of predefined behavioral task (e.g., `"debates"`, `"tones"`). |
| `dataset` | `TaskDataset` | Custom dataset for steering (optional). |
| `eval_prompts` | `list[str]` _(optional)_ | Prompts used for evaluation or alpha sweeps. |
| `max_samples` | `int` _(optional)_ | Maximum number of samples used for training. |

# Inference Arguments

## `get_steered_output(...)`

Generates model outputs with steering applied.

| Argument            | Type                           | Description                                                |
| ------------------- | ------------------------------ | ---------------------------------------------------------- |
| `prompts`           | `list[str]`                    | Input prompts.                                             |
| `target_labels`     | `list[str]`                    | Behaviors to encourage.                                    |
| `avoid_labels`      | `list[str]` _(optional)_       | Behaviors to suppress.                                     |
| `layer_strengths`   | `dict[int, float]`_(optional)_ | Layer-wise steering coefficients.                          |
| `max_new_tokens`    | `int` _(optional)_             | Maximum number of tokens to generate.                      |
| `generation_kwargs` | `dict` _(optional)_            | Standard generation parameters (temperature, top-p, etc.). |

# Dataset Abstractions

## `DatasetSchema`

Defines how dataset columns map to prompts and behavioral labels.
| Argument | Type | Description |
| ------------------ | ----------- | ------------------------------------------- |
| `prompt_column` | `str` | Column containing input prompts. |
| `category_columns` | `list[str]` | Columns representing behavioral categories. |

## `TaskDataset`

Utility for loading and standardizing datasets.
| Method | Description |
| ----------------------- | -------------------------------------------- |
| `from_huggingface(...)` | Load dataset from a Hugging Face repository. |
| `from_dataframe(...)` | Load dataset from an existing dataframe. |
| `from_csv(...)` | Load dataset from a local csv file. |
| `from_json(...)` | Load dataset from a local json file. |

# Evaluation & Optimization

## `sweep_alpha(...)`

Searches for optimal steering strengths using a judge.

| Argument         | Type        | Description                           |
| ---------------- | ----------- | ------------------------------------- |
| `task`           | `str`       | Task used for evaluation prompts.     |
| `judge`          | `BaseJudge` | Evaluation function (e.g., OODJudge). |
| `target_labels`  | `list[str]` | Labels to optimize for.               |
| `max_new_tokens` | `int`       | Generation length during evaluation.  |

# Judges

## `OODJudge`

Evaluates whether steering generalizes beyond training data (out-of-distribution).
