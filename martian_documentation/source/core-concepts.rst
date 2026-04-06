Core Concepts
=============

K-Steering provides a framework for influencing language model generation by modifying internal activations at specific layers, without fine-tuning the base model. Understanding the following concepts is essential for working with the library.

Steering
--------

A mechanism for influencing a language model's generation by modifying internal activations at specific layers, without fine-tuning the base model.

Steering Classifier
-------------------

A lightweight model trained on hidden states to distinguish between behavioral attributes (e.g., *Correct vs Incorrect*, *Empirical Grounding vs Straw Man Reframing*).

Steering Vector
---------------

A direction in activation space derived from the steering classifier that is added to model activations during inference.

K-Steering
----------

A framework for composing and applying steering vectors (potentially non-linearly) across different layers.

How It Works
------------

For both predefined and custom datasets:

1. A base question is selected.
2. The label-specific instruction is appended.
3. The model generates a response.
4. Hidden states are cached for steering or evaluation.

This design allows:

* Controlled behavioral induction
* Representation-level analysis
* Steering coefficient optimization
* Comparative evaluation across stylistic and argumentative axes

Datasets
--------

K-Steering supports multiple data sources: predefined tasks, Hugging Face datasets, and local files.

Predefined Datasets
~~~~~~~~~~~~~~~~~~~~

K-Steering includes two predefined steering datasets:

1. **Tones Dataset** -- Controls stylistic and communicative tone.
2. **Debates Dataset** -- Controls different debate styles.

For both datasets, each label is associated with a detailed instruction template. During hidden cache generation, these instructions are appended to the original question to induce the desired behavioral shift.

Steering Labels
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 25 30

   * - Dataset
     - Type
     - Label
   * - Tones
     - Expert
     - ``expert``
   * - Tones
     - Cautious
     - ``cautious``
   * - Tones
     - Empathetic
     - ``empathetic``
   * - Tones
     - Casual
     - ``casual``
   * - Tones
     - Concise
     - ``concise``
   * - Debates
     - Reductio ad Absurdum
     - ``Reductio ad Absurdum``
   * - Debates
     - Appeal to Precedent
     - ``Appeal to Precedent``
   * - Debates
     - Straw Man Reframing
     - ``Straw Man Reframing``
   * - Debates
     - Burden of Proof Shift
     - ``Burden of Proof Shift``
   * - Debates
     - Analogy Construction
     - ``Analogy Construction``
   * - Debates
     - Concession and Pivot
     - ``Concession and Pivot``
   * - Debates
     - Empirical Grounding
     - ``Empirical Grounding``
   * - Debates
     - Moral Framing
     - ``Moral Framing``
   * - Debates
     - Refutation by Distinction
     - ``Refutation by Distinction``
   * - Debates
     - Circular Anticipation
     - ``Circular Anticipation``

Tones Dataset
^^^^^^^^^^^^^

The **Tones Dataset** steers *how* a response is delivered. Each label corresponds to a distinct communicative style.

**Expert**
   Formal and academic tone with advanced terminology and domain-specific jargon. References to theories, standards, and research. Deep analytical reasoning with complex sentence structures. Simulates an authoritative subject-matter expert with technical depth and methodological precision.

**Cautious**
   Heavy use of hedging language with explicit acknowledgment of uncertainty. Multiple disclaimers and caveats. Presentation of competing perspectives with clear boundaries of knowledge. Models epistemic humility and uncertainty-aware reasoning.

**Empathetic**
   Emotionally validating language with a compassionate and supportive tone. Focus on human experience with emotional resonance over technical depth. Simulates affect-sensitive communication that prioritizes emotional understanding.

**Casual**
   Conversational tone with simple language and informal phrasing. Occasional humor with a friendly and relatable voice. Produces responses that feel natural and informal, like a conversation with a friend.

**Concise**
   Extremely brief responses with no introductions or elaboration. Short sentences and minimal wording. Bullet points where possible. Maximizes information density and minimizes verbosity.

Debates Dataset
^^^^^^^^^^^^^^^

The **Debates Dataset** steers *how arguments are constructed*. Each label corresponds to a specific rhetorical or argumentative strategy.

These labels are useful for studying structured reasoning patterns, modeling rhetorical strategies, evaluating persuasion styles, and analyzing argumentation dynamics in LLMs.

**Reductio ad Absurdum**
   Extends an opposing argument to its logical extreme to reveal contradictions or absurd outcomes. Core mechanism: "If we follow this logic, then..." to demonstrate unacceptable consequences.

**Appeal to Precedent**
   Grounds arguments in historical examples, case law, or established decisions. Core mechanism: past decisions and precedents justify present conclusions.

**Straw Man Reframing**
   Recharacterizes the opposing argument in simplified or exaggerated terms before refuting it. Core mechanism: "Essentially, what you're saying is..." then refute the reframed version.

**Burden of Proof Shift**
   Redirects responsibility for evidence onto the opponent. Core mechanism: claims stand unless definitively disproven.

**Analogy Construction**
   Builds an argument through comparison to a familiar scenario. Core mechanism: "This situation is similar to..." to guide audience through analogy.

**Concession and Pivot**
   Acknowledges a minor opposing point before shifting to a stronger counterargument. Core mechanism: "While it's true that... however..."

**Empirical Grounding**
   Bases arguments primarily on data, statistics, and verifiable research. Core mechanism: evidence-driven reasoning with methodological emphasis.

**Moral Framing**
   Positions the issue within ethical principles and value systems. Core mechanism: appeals to justice, fairness, obligation, or rights.

**Refutation by Distinction**
   Identifies critical contextual differences that invalidate comparisons. Core mechanism: "We must distinguish between..." to highlight meaningful differences.

**Circular Anticipation**
   Preemptively addresses potential counterarguments before they are raised. Core mechanism: "Some might argue..." followed by immediate rebuttal.

Custom Datasets
~~~~~~~~~~~~~~~~

K-Steering expects datasets to follow a structured schema where one column contains the input question/prompt, and additional columns correspond to behavioral category labels.

.. list-table::
   :header-rows: 1

   * - Question
     - Label_A
     - Label_B
     - Label_C
   * - Prompt 1
     - Response under A
     - Response under B
     - Response under C
   * - Prompt 2
     - Response under A
     - Response under B
     - Response under C

**Key Constraints:**

1. **Exactly one prompt column** -- Contains the base question or instruction. Mapped via ``prompt_column`` in ``DatasetSchema``.
2. **One column per steering label** -- Each column name represents a steering category. Mapped via ``category_columns`` in ``DatasetSchema``.
3. **Each row must contain aligned examples** -- All category responses in a row must correspond to the same prompt.

Loading datasets:

.. code-block:: python

   from k_steering.steering.dataset import DatasetSchema, TaskDataset

   schema = DatasetSchema(
       prompt_column="Question",
       category_columns=["Expert", "Casual"],
   )

   # From Hugging Face
   dataset, eval_prompts = TaskDataset.from_huggingface(
       repo_id="your-username/your-dataset",
       split="train",
       schema=schema,
   )

   # From CSV
   dataset, eval_prompts = TaskDataset.from_csv(path="my_dataset.csv", schema=schema)

   # From JSON
   dataset, eval_prompts = TaskDataset.from_json(path="my_dataset.json", schema=schema)

   # From DataFrame
   dataset, eval_prompts = TaskDataset.from_dataframe(df=df, schema=schema)

API Reference
-------------

SteeringConfig
~~~~~~~~~~~~~~

Used to define how steering classifiers are trained, evaluated, and applied.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Argument
     - Type
     - Description
   * - ``train_layer``
     - ``int``
     - Layer index whose hidden states are used to train steering classifiers.
   * - ``steer_layers``
     - ``list[int]``
     - Layers where steering vectors are injected during inference.
   * - ``eval_layer``
     - ``int`` *(optional)*
     - Layer used for evaluation or judging (e.g., ``-1`` for final layer).
   * - ``pos``
     - ``int`` *(optional)*
     - Token position used for evaluation (``-1`` = last token).

KSteering
~~~~~~~~~

Main entry point for training and applying steering.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Argument
     - Type
     - Description
   * - ``model_name``
     - ``str``
     - Hugging Face model identifier.
   * - ``steering_config``
     - ``SteeringConfig``
     - Configuration defining steering behavior.

fit(...)
~~~~~~~~

Trains steering classifiers.

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Argument
     - Type
     - Description
   * - ``task``
     - ``str``
     - Name of predefined behavioral task (e.g., ``"debates"``, ``"tones"``).
   * - ``dataset``
     - ``TaskDataset``
     - Custom dataset for steering (optional).
   * - ``eval_prompts``
     - ``list[str]`` *(optional)*
     - Prompts used for evaluation or alpha sweeps.
   * - ``max_samples``
     - ``int`` *(optional)*
     - Maximum number of samples used for training.

get_steered_output(...)
~~~~~~~~~~~~~~~~~~~~~~~~

Generates model outputs with steering applied.

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Argument
     - Type
     - Description
   * - ``prompts``
     - ``list[str]``
     - Input prompts.
   * - ``target_labels``
     - ``list[str]``
     - Behaviors to encourage.
   * - ``avoid_labels``
     - ``list[str]`` *(optional)*
     - Behaviors to suppress.
   * - ``layer_strengths``
     - ``dict[int, float]`` *(optional)*
     - Layer-wise steering coefficients.
   * - ``max_new_tokens``
     - ``int`` *(optional)*
     - Maximum number of tokens to generate.
   * - ``generation_kwargs``
     - ``dict`` *(optional)*
     - Standard generation parameters (temperature, top-p, etc.).

sweep_alpha(...)
~~~~~~~~~~~~~~~~

Searches for optimal steering strengths using a judge.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Argument
     - Type
     - Description
   * - ``task``
     - ``str``
     - Task used for evaluation prompts.
   * - ``judge``
     - ``BaseJudge``
     - Evaluation function (e.g., ``OODJudge``).
   * - ``target_labels``
     - ``list[str]``
     - Labels to optimize for.
   * - ``max_new_tokens``
     - ``int``
     - Generation length during evaluation.
