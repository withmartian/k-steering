# K-Steering

K-Steering is a steering framework for training and applying non-linear control mechanisms to large language models (LLMs), enabling you to steer model behavior **towards desired target attributes** and **away from undesired behaviors.**

The framework is based on the paper [Beyond Linear Steering: Unified Multi-Attribute Control for Language Models](https://arxiv.org/abs/2505.24535), which introduces Non-Linear K-Steering as a principled alternative to linear combinations of steering vectors for multi-attribute control.

![K-Steering Intro](docs/assets/k_steering_intro.png)
_Figure 1. An illustration of gradient-based K-Steering. For an activation vector A, we calculate a steering loss that
penalizes higher logits from a classifier on A for undesired labels and rewards higher logits for desired labels. By
backpropagating this loss through the classifier, we obtain the steered activations $A' = A − α∆L$_

In addition to K-Steering, the package also includes an implementation of [Contrastive Activation Addition (CAA)](https://arxiv.org/abs/2312.06681) for comparison and baseline steering experiments.

## Features

- Non-linear, multi-attribute steering via K-Steering
- Built-in support for Contrastive Activation Addition (CAA)
- Modular configuration for:
  - training layers
  - evaluation layers
  - token positions
  - steering application layers
- Predefined behavioral tasks for quick experimentation
- Designed for research, interpretability, and controlled generation workflows

## Quick Start

Get K-Steering running in minutes!!

### Try it in Google Colab

You can explore K-Steering without any local setup using the Colab notebook below.

[👉 K-Steering Colab Notebook](https://colab.research.google.com/drive/1cj3G_gKZ1OSOwwzxPRGjusazF3MFb-yl#scrollTo=Vbm8dXXtNCeV).

_(Includes installation, training, and inference examples)_

> The Colab notebook mirrors the examples below and is the recommended way to get started quickly.

### Prerequisites

- **Python 3.12 or higher**
- **[uv](https://docs.astral.sh/uv/)** - Fast Python package installer and resolver

To install `uv`, follow the instructions at https://docs.astral.sh/uv/getting-started/installation/

### Installation

For now, we recommend running K-Steering locally from the root directory:

```bash
uv sync # for Environment Setup
```

This will create the environment and install all required dependencies.

## API Usage

### K-Steering (Non-Linear Steering)

```python
from k_steering.steering.k_steer import KSteering
from k_steering.steering.config import SteeringConfig

# Initialize LLM for Steering
model_name = "unsloth/Llama-3.2-1B-Instruct"

# Define a Basic SteeringConfig
steering_config = SteeringConfig(
    eval_layer= -1, # Layer used for evaluation
    train_layer=1, # Layer on which the steering classifier is trained
    pos = -1, # Token position to consider
    steer_layers=[1,3]) # Layers where steering is applied

# Initialize KSteering
steer_model = KSteering(model_name, steering_config)

steer_model.fit(task = "debates", # Loading a Pre-defined task
                max_samples=10)  # Use a small subset for quick local runs

# Generate steered output
output = steer_model.get_steered_output(["Are political ideologies evolving in response to global challenges?"], target_labels=['Empirical Grounding'],
                           avoid_labels=['Straw Man Reframing'])

print(output[0])


```

### CAA Steering

K-Steering also includes an implementation of [Contrastive Activation Addition (CAA) paper](https://arxiv.org/abs/2312.06681) for linear steering baselines.

```python
from k_steering.steering.k_steer import CAASteering
from k_steering.steering.config import SteeringConfig

# Initialize LLM for Steering
model_name = "unsloth/Llama-3.2-1B-Instruct"

# Define a Basic SteeringConfig
steering_config = SteeringConfig(
    eval_layer= -1,
    train_layer=1,
    pos = -1,
    steer_layers=[1,3])

# Initialize CAASteering Class
steer_model = CAASteering(model_name, steering_config)

steer_model.fit(task = "debates",
                max_samples=10)

# Generate steered output
output = steer_model.get_steered_output(["Are political ideologies evolving in response to global challenges?"], target_labels=['Empirical Grounding'],
                           avoid_labels=['Straw Man Reframing'])

print(output[0])


```
