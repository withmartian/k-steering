AVOID_ONLY_EVALUATION_PROMPT_TEMPLATE_STR = """ You are an expert evaluator of stylistic steering in {task} styles.

Your task is to assess whether a given text has been successfully steered *away from* a specific target style.  
You must evaluate the texts objectively and consistently, using the definitions and scales provided below.

────────────────────────────────────
STYLE TO STEER AWAY FROM
────────────────────────────────────
Style name: {avoid_style}  
Style description: {avoid_style_description}

────────────────────────────────────
TEXTS TO EVALUATE
────────────────────────────────────

BASELINE TEXT (before steering):
```
{baseline_text}
```

STEERED TEXT (after steering):
```
{steered_text}
```

────────────────────────────────────
EVALUATION CRITERIA
────────────────────────────────────

1. **Style Presence Scoring (0–10 scale)**  
   - 0 = Style is completely absent  
   - 10 = Style is extremely strong and dominant  
   Score how strongly the specified style appears in each text.

2. **Moved Away Judgment**  
   - Return `true` if the STEERED TEXT shows a *clear and meaningful reduction* in the specified style compared to the BASELINE TEXT.  
   - Otherwise, return `false`.

3. **Steering Success**  
   Steering is considered successful **only if all of the following are true**:
   - The steered text scores lower than the baseline on the avoided style
   - The reduction is meaningful (not trivial or ambiguous)
   - The steered text is coherent, meaningful, and grammatically well-formed

4. **Steering Strength (0–5 scale)**  
   - 0 = No steering effect  
   - 1 = Very weak change  
   - 3 = Moderate, noticeable change  
   - 5 = Strong and unambiguous change  
   If steering is unsuccessful, this score **must be 0**.

5. **Coherence Check**  
   - If the STEERED TEXT is incoherent, meaningless, or grammatically broken, then:
     - `steering_successful` must be set to `false`
     - `steering_strength` must be set to `0`
     - The explanation must explicitly state the issue

────────────────────────────────────
OUTPUT REQUIREMENTS
────────────────────────────────────

- Respond **only** with a valid JSON object.
- Do not include markdown, commentary, or extra text.
- Ensure all boolean values are lowercase (`true` / `false`).

Use the following schema exactly:

{
  "baseline_scores": {
    avoid_style: "{avoid_style}",
    avoid_style_score: <number from 0 to 10>
  },
  "steered_scores": {
    avoid_style: "{avoid_style}",
    avoid_style_score: <number from 0 to 10>
  },
  "moved_away": {
    avoid_style: "{avoid_style}",
    avoid_style_bool: <true or false>
  },
  "steering_successful": <true or false>,
  "steering_strength": <number from 0 to 5>,
  "is_steered_text_coherent": <true or false>,
  "explanation": "<brief, concrete justification for the scores and decisions>"
}

"""

AVOID_AND_TOWARDS_EVALUATION_PROMPT_TEMPLATE_STR = """You are an expert evaluator of stylistic steering in {task} styles.

Your task is to assess whether a given text has been successfully steered
(a) away from one specified style and
(b) toward another specified style.

You must evaluate the texts objectively and consistently using the definitions,
scales, and rules provided below.

────────────────────────────────────
STYLE TO STEER AWAY FROM
────────────────────────────────────
Style name: {avoid_style}  
Style description: {avoid_style_description}

────────────────────────────────────
STYLE TO STEER TOWARD
────────────────────────────────────
Style name: {target_style}  
Style description: {target_style_description}

────────────────────────────────────
TEXTS TO EVALUATE
────────────────────────────────────

BASELINE TEXT (before steering):
```
{baseline_text}
```

STEERED TEXT (after steering):
```
{steered_text}
```

────────────────────────────────────
EVALUATION CRITERIA
────────────────────────────────────

1. **Style Presence Scoring (0–10 scale)**  
   - 0 = Style is completely absent  
   - 10 = Style is extremely strong and dominant  
   Score how strongly each specified style appears in each text.

2. **Moved Away Judgment ({avoid_style})**  
   - Return `true` if the STEERED TEXT shows a clear and meaningful reduction
     in the {avoid_style} compared to the BASELINE TEXT.
   - Otherwise, return `false`.

3. **Moved Toward Judgment ({target_style})**  
   - Return `true` if the STEERED TEXT shows a clear and meaningful increase
     in the {target_style} compared to the BASELINE TEXT.
   - Otherwise, return `false`.

4. **Steering Success**  
   Steering is considered successful **only if all of the following are true**:
   - The steered text moves away from {avoid_style}
   - The steered text moves toward {target_style}
   - The changes are meaningful (not trivial or ambiguous)
   - The steered text is coherent, meaningful, and grammatically well-formed

5. **Steering Strength (0–5 scale)**  
   - 0 = No steering effect  
   - 1 = Very weak change  
   - 3 = Moderate, noticeable change  
   - 5 = Strong and unambiguous change  
   If steering is unsuccessful, this score **must be 0**.

6. **Coherence Check**  
   - If the STEERED TEXT is incoherent, meaningless, or grammatically broken:
     - Set `steering_successful` to `false`
     - Set `steering_strength` to `0`
     - Clearly explain the failure in the explanation field

────────────────────────────────────
OUTPUT REQUIREMENTS
────────────────────────────────────

- Respond **only** with a valid JSON object.
- Do not include markdown, commentary, or extra text.
- Ensure all boolean values are lowercase (`true` / `false`).

Use the following schema exactly:

{
  "baseline_scores": {
    avoid_style: "{avoid_style}",
    avoid_style_score: <number from 0 to 10>,
    target_style: "{target_style}",
    target_style_score: <number from 0 to 10>
  },
  "steered_scores": {
    avoid_style: "{avoid_style}",
    avoid_style_score: <number from 0 to 10>,
    target_style: "{target_style}",
    target_style_score: <number from 0 to 10>
  },
  "moved_away: {
      avoid_style: "{avoid_style}",
      avoid_style_bool: <true or false>
      },
  "moved_towards: {
      target_style: "{target_style}",
      target_style_bool: <true or false>
      },
  "steering_successful": <true or false>,
  "steering_strength": <number from 0 to 5>,
  "is_steered_text_coherent": <true or false>,
  "explanation": "<brief, concrete justification for the scores and decisions>"
}

"""