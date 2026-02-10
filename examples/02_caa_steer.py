"""
Example: Using CAA-Steering to steer a language model's responses.

This script demonstrates how to:
1. Load a Hugging Face model
2. Configure CAA-Steering
3. Train steering classifiers on a task
4. Generate steered outputs at inference time
"""

from k_steering.steering.config import SteeringConfig
from k_steering.steering.caa import CAASteering

# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------

# Hugging Face model to be steered
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"

# ---------------------------------------------------------------------
# Steering configuration
# ---------------------------------------------------------------------

# Define how and where steering classifiers are trained and applied
steering_config = SteeringConfig(
    train_layer=1,          # Layer index used to train the steering vectors
    pos = -1,               # Token Position used to extract hidden activations
    steer_layers=[1, 3],    # Layers where the steering will be applied
)

# ---------------------------------------------------------------------
# Task and generation settings
# ---------------------------------------------------------------------

# Name of the task used to load training data
# (e.g., "debates" or "tones")
TASK_NAME = "debates"

# Maximum number of tokens to generate
MAX_NEW_TOKENS = 100

# Standard generation parameters passed to the model
GENERATION_KWARGS = {
    "temperature": 1.0,
    "top_p": 0.9,
}

# ---------------------------------------------------------------------
# Initialize CAA-Steering
# ---------------------------------------------------------------------

# Create a CAASteering wrapper around the base model
steer_model = CAASteering(
    model_name=MODEL_NAME,
    steering_config=steering_config,
)

# ---------------------------------------------------------------------
# Train steering classifiers
# ---------------------------------------------------------------------

# Fit steering classifiers using task-specific data
# max_samples controls how many examples are used for training, remove max_samples arg to load complete data
steer_model.fit(
    task=TASK_NAME,
    max_samples=10,
)

# ---------------------------------------------------------------------
# Inference with steering
# ---------------------------------------------------------------------

# Input prompts
prompts = [
    "Are political ideologies evolving in response to global challenges?"
]

# Generate steered output by encouraging and discouraging specific labels
output = steer_model.get_steered_output(
    prompts,
    target_labels=['Empirical Grounding'],     # Labels to steer *towards*
    avoid_labels=['Straw Man Reframing'],    # Labels to steer *away from*
    max_new_tokens=MAX_NEW_TOKENS,
    generation_kwargs=GENERATION_KWARGS,
)

print(output)