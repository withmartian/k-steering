K-Steering Documentation
========================

**K-Steering** is a lightweight and flexible toolkit for steering large language model outputs with minimal overhead and maximum control.

K-Steering enables controlled generation from language models by applying steering vectors at inference time, supporting parameter sweeps to identify optimal steering strengths, and integrating seamlessly with Hugging Face and local datasets.

Whether you're experimenting with interpretability, alignment, or controllable generation, K-Steering is designed to keep the workflow simple and modular.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   core-concepts
   how-it-works

Getting Started
---------------

See the main `README <https://github.com/withmartian/k-steering>`_ for a high-level overview, or jump straight to the :doc:`how-it-works` guide.

Key Features
------------

* **Inference-Time Steering**: Apply steering vectors at specific layers without fine-tuning the base model
* **Non-Linear Steering**: Compose and apply steering vectors across different layers using learned classifiers
* **Parameter Sweeps**: Search for optimal steering strengths using built-in evaluation utilities
* **Flexible Datasets**: Use predefined tasks, Hugging Face datasets, or local CSV/JSON/DataFrame sources
* **Reproducible Experiments**: Structured configuration via ``SteeringConfig`` for consistent results

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
