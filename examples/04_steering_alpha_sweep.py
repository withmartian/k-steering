"""
Example: Sweeping steering strength (alpha) using an OOD judge.

This script demonstrates how to:
1. Train K-Steering classifiers on a task
2. Use an out-of-distribution (OOD) judge to evaluate generations
3. Sweep steering strength (alpha) per layer
4. Apply the learned layer-wise strengths during inference
"""

import asyncio

from k_steering.evals.judges.ood import OODJudge
from k_steering.steering.config import SteeringConfig, TrainerConfig
from k_steering.steering.k_steer import KSteering

# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------

# Hugging Face model to be steered
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"

# ---------------------------------------------------------------------
# Steering & Trainer configuration
# ---------------------------------------------------------------------

# Configure training, evaluation, and steering layers
steering_config = SteeringConfig(
    train_layer=1,          # Layer used to train steering classifiers
    steer_layers=[1, 3],    # Layers where steering is applied
)

# Define a TrainerConfig for K-steering classifier configuration
trainer_config = TrainerConfig(
    clf_type = "mlp",       # Type of classifier, for e.g., "mlp" or "linear"
    hidden_dim = 128        # Hidden dimension of the MLP model
)


# ---------------------------------------------------------------------
# Task and generation settings
# ---------------------------------------------------------------------

# Name of the task used to load training data
# (e.g., "debates" or "tones")
TASK_NAME = "debates"

MAX_NEW_TOKENS = 100

# Maximum number of samples for training
MAX_SAMPLES = 10

GENERATION_KWARGS = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "temperature": 1.0,
    "top_p": 0.9,
}

# ---------------------------------------------------------------------
# Initialize K-Steering
# ---------------------------------------------------------------------

steer_model = KSteering(
    model_name=MODEL_NAME,
    steering_config=steering_config,
    trainer_config=trainer_config
)

# ---------------------------------------------------------------------
# Train steering classifiers
# ---------------------------------------------------------------------

# Fit classifiers on task-specific data
steer_model.fit(
    task=TASK_NAME,
    max_samples=MAX_SAMPLES,         # Limit samples for faster experimentation
)

# ---------------------------------------------------------------------
# Sweep steering strength (alpha) using a judge
# ---------------------------------------------------------------------

# OODJudge evaluates whether steering generalizes beyond training data
judge = OODJudge()

# Sweep alpha values for each steering layer and select the best strengths
layer_wise_alpha = asyncio.run(
    steer_model.sweep_alpha(
        task=TASK_NAME,
        judge=judge,
        target_labels=["Empirical Grounding"],
        max_new_tokens=MAX_NEW_TOKENS,
    )
)

print("Layer-wise steering strengths:")
print(layer_wise_alpha)

# ---------------------------------------------------------------------
# Inference with learned layer-wise strengths
# ---------------------------------------------------------------------

# Input prompt
prompts = [
    "Are political ideologies evolving in response to global challenges?"
]

# Generate steered output using learned alpha values
output = steer_model.get_steered_output(
    prompts,
    target_labels=["Empirical Grounding"],       # Behaviors to encourage
    avoid_labels=["Straw Man Reframing"],        # Behaviors to suppress
    generation_kwargs=GENERATION_KWARGS,
    layer_strengths=layer_wise_alpha,            # Apply learned alphas
)

print(output)
