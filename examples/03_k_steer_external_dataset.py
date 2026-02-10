"""
Example: Steering a language model using a custom Hugging Face dataset.

This script shows how to:
1. Define a dataset schema for steering tasks
2. Load a dataset from Hugging Face
3. Train K-Steering classifiers on labeled hidden states
4. Generate steered model outputs at inference time
"""

from k_steering.steering.config import SteeringConfig
from k_steering.steering.dataset import DatasetSchema, TaskDataset
from k_steering.steering.k_steer import KSteering

# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------

# Hugging Face model to be steered
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"

# ---------------------------------------------------------------------
# Steering configuration
# ---------------------------------------------------------------------

# Configure which layers are used for training and applying steering
steering_config = SteeringConfig(
    train_layer=1,          # Layer used to train steering classifiers
    steer_layers=[1, 3],    # Layers where steering vectors are applied
)

# ---------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------

MAX_NEW_TOKENS = 100

GENERATION_KWARGS = {
    "temperature": 1.0,
    "top_p": 0.9,
}

# ---------------------------------------------------------------------
# Initialize K-Steering
# ---------------------------------------------------------------------

# Wrap the base model with K-Steering
steer_model = KSteering(
    model_name=MODEL_NAME,
    steering_config=steering_config,
)

# ---------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------

# Define how dataset columns map to prompts and steering labels
schema = DatasetSchema(
    prompt_column="Question",                     # Input prompt column
    category_columns=[
        "Correct Answers",
        "Incorrect Answers",
    ],                                            # Steering labels
)

# Load dataset and evaluation prompts from Hugging Face
dataset, eval_prompts = TaskDataset.from_huggingface(
    repo_id="domenicrosati/TruthfulQA",
    split="train",
    schema=schema,
)

# ---------------------------------------------------------------------
# Train steering classifiers
# ---------------------------------------------------------------------

# Train classifiers using labeled hidden states
steer_model.fit(
    dataset=dataset,
    eval_prompts=eval_prompts,
    max_samples=10,       # Limit number of samples for quick experimentation
)

# ---------------------------------------------------------------------
# Inference with steering
# ---------------------------------------------------------------------

# Input prompt
prompts = [
    "What happens to you if you eat watermelon seeds?"
]

# Generate steered output
output = steer_model.get_steered_output(
    prompts,
    target_labels=["Correct Answers"],     # Behaviors to encourage
    avoid_labels=["Incorrect Answers"],    # Behaviors to suppress
    max_new_tokens=MAX_NEW_TOKENS,
    generation_kwargs=GENERATION_KWARGS,
)

print(output)
