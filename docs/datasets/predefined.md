# Predefined Dataset Labels Overview

K-Steering includes two predefined steering datasets:

1. **Tones Dataset** – Controls stylistic and communicative tone.
2. **Debates Dataset** – Controls different debate styles.

For both datasets, each label is associated with a detailed instruction template.  
During hidden cache generation, these instructions are appended to the original question to induce the desired behavioral shift.

Together, they enable fine-grained exploration of compositional behavioral control in language models.

---

# Steering Labels

| Dataset | Type                      | Label                       |
| ------- | ------------------------- | --------------------------- |
| Tones   | Expert                    | `expert`                    |
| Tones   | Cautious                  | `cautious`                  |
| Tones   | Empathetic                | `empathetic`                |
| Tones   | Casual                    | `casual`                    |
| Tones   | Concise                   | `concise`                   |
| Debates | Reductio ad Absurdum      | `Reductio ad Absurdum`      |
| Debates | Appeal to Precedent       | `Appeal to Precedent`       |
| Debates | Straw Man Reframing       | `Straw Man Reframing`       |
| Debates | Burden of Proof Shift     | `Burden of Proof Shift`     |
| Debates | Analogy Construction      | `Analogy Construction`      |
| Debates | Concession and Pivot      | `Concession and Pivot`      |
| Debates | Empirical Grounding       | `Empirical Grounding`       |
| Debates | Moral Framing             | `Moral Framing`             |
| Debates | Refutation by Distinction | `Refutation by Distinction` |
| Debates | Circular Anticipation     | `Circular Anticipation`     |

# 1️⃣ Tones Dataset

The **Tones Dataset** steers _how_ a response is delivered.  
Each label corresponds to a distinct communicative style.

---

## 🔹 Expert

**Style Characteristics:**

- Formal and academic tone
- Advanced terminology and domain-specific jargon
- References to theories, standards, and research
- Deep analytical reasoning
- Complex sentence structures

**Behavioral Goal:**  
Simulate an authoritative subject-matter expert with technical depth and methodological precision.

---

## 🔹 Cautious

**Style Characteristics:**

- Heavy use of hedging language
- Explicit acknowledgment of uncertainty
- Multiple disclaimers and caveats
- Presentation of competing perspectives
- Clear boundaries of knowledge

**Behavioral Goal:**  
Model epistemic humility and uncertainty-aware reasoning.

---

## 🔹 Empathetic

**Style Characteristics:**

- Emotionally validating language
- Compassionate and supportive tone
- Focus on human experience
- Emotional resonance over technical depth

**Behavioral Goal:**  
Simulate affect-sensitive communication that prioritizes emotional understanding.

---

## 🔹 Casual

**Style Characteristics:**

- Conversational tone
- Simple language
- Informal phrasing
- Occasional humor
- Friendly and relatable voice

**Behavioral Goal:**  
Produce responses that feel natural and informal, like a conversation with a friend.

---

## 🔹 Concise

**Style Characteristics:**

- Extremely brief responses
- No introductions or elaboration
- Short sentences
- Minimal wording
- Bullet points where possible

**Behavioral Goal:**  
Maximize information density and minimize verbosity.

---

# 2️⃣ Debates Dataset

The **Debates Dataset** steers _how arguments are constructed_.  
Each label corresponds to a specific rhetorical or argumentative strategy.

These labels are useful for:

- Studying structured reasoning patterns
- Modeling rhetorical strategies
- Evaluating persuasion styles
- Analyzing argumentation dynamics in LLMs

---

## 🔹 Reductio ad Absurdum

Extends an opposing argument to its logical extreme to reveal contradictions or absurd outcomes.

**Core Mechanism:**  
“If we follow this logic, then…” → demonstrate unacceptable consequences.

---

## 🔹 Appeal to Precedent

Grounds arguments in historical examples, case law, or established decisions.

**Core Mechanism:**  
Past decisions and precedents justify present conclusions.

---

## 🔹 Straw Man Reframing

Recharacterizes the opposing argument in simplified or exaggerated terms before refuting it.

**Core Mechanism:**  
“Essentially, what you're saying is…” → refute the reframed version.

---

## 🔹 Burden of Proof Shift

Redirects responsibility for evidence onto the opponent.

**Core Mechanism:**  
Claims stand unless definitively disproven.

---

## 🔹 Analogy Construction

Builds an argument through comparison to a familiar scenario.

**Core Mechanism:**  
“This situation is similar to…” → guide audience through analogy.

---

## 🔹 Concession and Pivot

Acknowledges a minor opposing point before shifting to a stronger counterargument.

**Core Mechanism:**  
“While it's true that… however…”

---

## 🔹 Empirical Grounding

Bases arguments primarily on data, statistics, and verifiable research.

**Core Mechanism:**  
Evidence-driven reasoning with methodological emphasis.

---

## 🔹 Moral Framing

Positions the issue within ethical principles and value systems.

**Core Mechanism:**  
Appeals to justice, fairness, obligation, or rights.

---

## 🔹 Refutation by Distinction

Identifies critical contextual differences that invalidate comparisons.

**Core Mechanism:**  
“We must distinguish between…” → highlight meaningful differences.

---

## 🔹 Circular Anticipation

Preemptively addresses potential counterarguments before they are raised.

**Core Mechanism:**  
“Some might argue…” → immediate rebuttal.

---

# 🔬 How These Labels Are Used

For both datasets:

1. A base question is selected.
2. The label-specific instruction is appended.
3. The model generates a response.
4. Hidden states are cached for steering or evaluation.

This design allows:

- Controlled behavioral induction
- Representation-level analysis
- Steering coefficient optimization
- Comparative evaluation across stylistic and argumentative axes

---
