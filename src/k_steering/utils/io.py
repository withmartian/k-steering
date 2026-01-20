import os
from openai import OpenAI
from anthropic import Anthropic
import math
from typing import Optional


def openai_api_call(
    prompt: str,
    *,
    mode: str = "json",  # "json" or "logprob"
    response_format=None,
    system_prompt: str = "You are a helpful assistant. Always respond with valid JSON. No explanations.",
    model: str = "gpt-4.1-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    top_logprobs: int = 20,
    seed: int = 0,
) -> Optional[object]:

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # -------------------------------
    # LOGPROB SCORING MODE
    # -------------------------------
    if mode == "logprob":
        messages = [{"role": "user", "content": prompt}]

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=top_logprobs,
            seed=seed,
        )

        try:
            top = completion.choices[0].logprobs.content[0].top_logprobs
        except (IndexError, AttributeError):
            return None

        result = {}
        for el in top:
            try:
                result[int(el.token)] = math.exp(el.logprob)
            except ValueError:
                continue

        total = sum(result.values())
        if total < 0.25:
            return None

        score = sum(k * v for k, v in result.items()) / total
        return score

    # -------------------------------
    # JSON SCHEMA MODE (original)
    # -------------------------------
    if response_format is None:
        raise ValueError("response_format must be provided when mode='json'")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": response_format.__name__,
                "schema": response_format.model_json_schema(),
            },
        },
    )

    parsed = response.choices[0].message.content
    return response_format.model_validate_json(parsed)


def anthropic_api_call(prompt, response_format, system_prompt="You are a helpful assistant. Always respond with valid JSON. No explanations.",
                       model="claude-sonnet-4-5",
                       temperature=0.9, max_tokens:int= 1024):

    # Set your OpenAI API key
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Prepare your messages in OpenAI's chat template format
    messages = [
        # {"role": "system", "content": persona_prompt},
        {"role": "user", "content": prompt}
    ]

    # Call the Chat API
    response = client.beta.messages.parse(
        model=model,
        max_tokens=max_tokens,
        betas=["structured-outputs-2025-11-13"],
        system=system_prompt,
        messages=messages,
        temperature=temperature,
        output_format=response_format,
    )
    parsed = response.parsed_output
    return response_format.model_validate(parsed)