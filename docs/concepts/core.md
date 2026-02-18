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
