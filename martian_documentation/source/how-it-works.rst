How It Works
============

This page covers installation, a quickstart walkthrough, and guidance for running larger models.

Installation
------------

Clone the Repository
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/withmartian/k-steering.git

Prerequisites
~~~~~~~~~~~~~

* **Python 3.12 or higher**
* **uv** - Fast Python package installer and resolver

To install ``uv``, follow the instructions at https://docs.astral.sh/uv/getting-started/installation/

Install Dependencies
~~~~~~~~~~~~~~~~~~~~

For now, we recommend running K-Steering locally from the root directory:

.. code-block:: bash

   uv sync  # for Environment Setup

This will create the environment and install all required dependencies.

Quick Start
-----------

Try it in Google Colab
~~~~~~~~~~~~~~~~~~~~~~

You can explore K-Steering without any local setup using the `Colab notebook <https://colab.research.google.com/drive/1cj3G_gKZ1OSOwwzxPRGjusazF3MFb-yl#scrollTo=Vbm8dXXtNCeV>`_.

*(Includes installation, training, and inference examples)*

API Usage
~~~~~~~~~

See the `examples/ <https://github.com/withmartian/k-steering/tree/main/examples>`_ directory for complete scripts for training different steering models.

K-Steering (Non-Linear Steering)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example shows how to use **K-Steering** to guide a language model's behavior by training lightweight steering classifiers and applying them during inference.

1. Load Required Modules
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from k_steering.steering.config import SteeringConfig
   from k_steering.steering.k_steer import KSteering

2. Select a Base Model
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Hugging Face model to be steered
   MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"

3. Configure Steering
^^^^^^^^^^^^^^^^^^^^^^

Define which layers are used to train and apply steering.

.. code-block:: python

   steering_config = SteeringConfig(
       train_layer=1,          # Layer used to train the steering classifier
       steer_layers=[1, 3],    # Layers where steering is applied
   )

4. Task and Generation Settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   TASK_NAME = "debates"       # e.g., "debates" or "tones"
   MAX_NEW_TOKENS = 100        # Maximum number of tokens to generate
   MAX_SAMPLES = 10            # Maximum number of samples for training

   GENERATION_KWARGS = {
       "max_new_tokens": MAX_NEW_TOKENS,
       "temperature": 1.0,
       "top_p": 0.9,
   }

5. Initialize K-Steering
^^^^^^^^^^^^^^^^^^^^^^^^^^

Wrap the base model with K-Steering.

.. code-block:: python

   steer_model = KSteering(
       model_name=MODEL_NAME,
       steering_config=steering_config,
   )

6. Train Steering Classifiers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train steering classifiers on task-specific data. Remove ``max_samples`` to use the full dataset.

.. code-block:: python

   steer_model.fit(
       task=TASK_NAME,
       max_samples=MAX_SAMPLES,
   )

7. Generate Steered Outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

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

Large Model Setup
-----------------

The table below provides approximate GPU memory requirements for transformer models at different parameter scales, helping you determine which models can run on Free Tier Colab versus requiring larger compute setups.

.. list-table::
   :header-rows: 1
   :widths: 15 12 18 18 20 17

   * - Model Size
     - Params
     - FP16 VRAM (Inference)
     - 4-bit VRAM (Inference)
     - Recommended GPU
     - Colab Free Feasible?
   * - Tiny
     - 100M--300M
     - ~0.5--1 GB
     - ~0.3--0.5 GB
     - Any GPU
     - Yes
   * - Small
     - 500M--1B
     - ~2--3 GB
     - ~1--1.5 GB
     - T4 / L4
     - Yes
   * - Medium
     - 2B--3B
     - ~5--7 GB
     - ~2--3 GB
     - T4 (tight) / L4
     - No
   * - Upper-Mid
     - 7B
     - ~14--16 GB
     - ~4--6 GB
     - L4 / A100
     - No
   * - Large
     - 13B
     - ~26--28 GB
     - ~8--10 GB
     - A100 40GB
     - No
   * - Very Large
     - 30B
     - ~60+ GB
     - ~18--22 GB
     - Multi-GPU
     - No
   * - Frontier
     - 70B
     - ~140+ GB
     - ~35--40 GB
     - Multi A100/H100
     - No
