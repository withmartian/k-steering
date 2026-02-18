## Quick Start

Get K-Steering running in minutes!!

### Try it in Google Colab

You can explore K-Steering without any local setup using the Colab notebook below.

[👉 K-Steering Colab Notebook](https://colab.research.google.com/drive/1cj3G_gKZ1OSOwwzxPRGjusazF3MFb-yl#scrollTo=Vbm8dXXtNCeV).

_(Includes installation, training, and inference examples)_

## API Usage

See [Examples](/examples/) for Complete Scripts for Training Different Steering Models

## K-Steering (Non-Linear Steering)

This example shows how to use **K-Steering** to guide a language model’s behavior by training lightweight steering classifiers and applying them during inference.

---

### 1️⃣ Load Required Modules

```python
from k_steering.steering.config import SteeringConfig
from k_steering.steering.k_steer import KSteering
```

### 2️⃣ Select a Base Model

```python
# Hugging Face model to be steered
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
```

### 3️⃣ Configure Steering

Define which layers are used to train and apply steering.

```python
steering_config = SteeringConfig(
    train_layer=1,          # Layer used to train the steering classifier
    steer_layers=[1, 3],    # Layers where steering is applied
)
```

### 4️⃣ Task and Generation Settings

```python
TASK_NAME = "debates"       # e.g., "debates" or "tones"
MAX_NEW_TOKENS = 100        # Maximum number of tokens to generate
MAX_SAMPLES = 10            # Maximum number of samples for training

GENERATION_KWARGS = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "temperature": 1.0,
    "top_p": 0.9,
}
```

### 5️⃣ Initialize K-Steering

Wrap the base model with K-Steering.

```python
steer_model = KSteering(
    model_name=MODEL_NAME,
    steering_config=steering_config,
)
```

### 6️⃣ Train Steering Classifiers

Train steering classifiers on task-specific data. Remove `max_samples` to use the full dataset.

```python
steer_model.fit(
    task=TASK_NAME,
    max_samples=MAX_SAMPLES,
)
```

### 7️⃣ Generate Steered Outputs

```python
prompts = [
    "Are political ideologies evolving in response to global challenges?"
]

output = steer_model.get_steered_output(
    prompts,
    target_labels=["Empirical Grounding"],     # Behaviors to encourage
    avoid_labels=["Straw Man Reframing"],      # Behaviors to suppress
    generation_kwargs=GENERATION_KWARGS,
)

print(output)
```
