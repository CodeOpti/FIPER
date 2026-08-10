"""Official-provider model adapters used by the generation pipeline.

API credentials are read from environment variables and are never stored in
the repository.  The module intentionally contains no proxy-provider branch.
"""

from __future__ import annotations

import os
import statistics
import time
from typing import Any


def _read_api_keys(variable_name: str) -> list[str]:
    """Read a comma-separated API-key list from an environment variable."""

    raw_value = os.getenv(variable_name, "")
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _select_key(keys: list[str], key_index: int, variable_name: str) -> str:
    if not keys:
        raise RuntimeError(
            f"Set {variable_name} before making an official API request."
        )
    return keys[key_index % len(keys)]


def _build_messages(
    system_prompt: str,
    question_text: str,
    messages: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    if messages:
        return messages
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question_text},
    ]


def _report_response(
    model_name: str,
    output_prompt: bool,
    response_texts: list[str],
    mean_logprobs: list[float] | None = None,
) -> None:
    if not output_prompt:
        return
    print(f"model={model_name}")
    print(f"responses={response_texts}")
    if mean_logprobs:
        print(f"mean_logprobs={mean_logprobs}")


def ChatGPT_official_function(
    key_index: int = 0,
    model_name: str = "gpt-4o",
    system_prompt: str = "You are a helpful programming assistant.",
    question_text: str = "Provide a concise response.",
    messages: list[dict[str, str]] | None = None,
    num_candidates: int = 1,
    temperature: float = 0.01,
    max_length: int = 1024,
    return_logprobs: bool = False,
    output_prompt: bool = False,
) -> tuple[list[str], list[float]]:
    """Generate candidates through the official OpenAI-compatible endpoint."""

    from openai import OpenAI

    keys = _read_api_keys("OPENAI_API_KEYS")
    client = OpenAI(api_key=_select_key(keys, key_index, "OPENAI_API_KEYS"))
    response = client.chat.completions.create(
        model=model_name,
        messages=_build_messages(system_prompt, question_text, messages),
        n=num_candidates,
        temperature=temperature,
        max_tokens=max_length,
        logprobs=return_logprobs,
    )
    response_texts = [choice.message.content.strip() for choice in response.choices]
    mean_logprobs: list[float] = []
    if return_logprobs:
        for choice in response.choices:
            content = getattr(getattr(choice, "logprobs", None), "content", None) or []
            values = [item.logprob for item in content if item.logprob is not None]
            mean_logprobs.append(round(statistics.mean(values), 6) if values else 0.0)
        order = sorted(range(len(mean_logprobs)), key=mean_logprobs.__getitem__, reverse=True)
        response_texts = [response_texts[index] for index in order]
        mean_logprobs = [mean_logprobs[index] for index in order]
    _report_response(model_name, output_prompt, response_texts, mean_logprobs)
    return response_texts, mean_logprobs


def Gemini_official_function(
    key_index: int = 0,
    model_name: str = "gemini-2.5-flash",
    system_prompt: str = "You are a helpful programming assistant.",
    question_text: str = "Provide a concise response.",
    messages: list[dict[str, str]] | None = None,
    num_candidates: int = 1,
    temperature: float = 0.01,
    max_length: int | str = "unlimited",
    thinking_budget: int = -404,
    return_logprobs: bool = False,
    output_prompt: bool = False,
) -> tuple[list[str], list[float]]:
    """Generate candidates through the official Google Gemini API."""

    from google import genai
    from google.genai import types

    keys = _read_api_keys("GOOGLE_API_KEYS")
    client = genai.Client(api_key=_select_key(keys, key_index, "GOOGLE_API_KEYS"))
    config_kwargs: dict[str, Any] = {
        "system_instruction": system_prompt,
        "temperature": temperature,
        "candidate_count": num_candidates,
    }
    if isinstance(max_length, int):
        config_kwargs["max_output_tokens"] = max_length
    if thinking_budget >= 0:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget
        )
    response = client.models.generate_content(
        model=model_name,
        config=types.GenerateContentConfig(**config_kwargs),
        contents=question_text if not messages else messages,
    )
    response_texts = [
        candidate.content.parts[0].text.strip()
        for candidate in response.candidates[:num_candidates]
        if candidate.content and candidate.content.parts
    ]
    if len(response_texts) != num_candidates:
        raise RuntimeError(
            f"Gemini returned {len(response_texts)} candidates; expected {num_candidates}."
        )
    _report_response(model_name, output_prompt, response_texts)
    return response_texts, []


def DeepSeek_official_function(
    model_name: str = "deepseek-chat",
    key_index: int = 0,
    think: bool = False,
    system_prompt: str = "You are a helpful programming assistant.",
    question_text: str = "Provide a concise response.",
    messages: list[dict[str, str]] | None = None,
    temperature: float = 0.01,
    max_length: int = 1024,
    output_prompt: bool = False,
) -> tuple[list[str], dict[str, Any], str, float | str]:
    """Generate through the official DeepSeek API only."""

    from openai import OpenAI

    keys = _read_api_keys("DEEPSEEK_API_KEYS")
    client = OpenAI(
        api_key=_select_key(keys, key_index, "DEEPSEEK_API_KEYS"),
        base_url="https://api.deepseek.com",
    )
    if think or model_name == "deepseek-reasoner":
        model_name = "deepseek-reasoner"
    else:
        model_name = "deepseek-chat"
    request: dict[str, Any] = {
        "model": model_name,
        "messages": _build_messages(system_prompt, question_text, messages),
        "max_tokens": max_length,
    }
    if model_name == "deepseek-chat":
        request.update({"temperature": temperature, "logprobs": True})
    response = client.chat.completions.create(**request)
    choice = response.choices[0]
    response_texts = [choice.message.content.strip()]
    reasoning_text = getattr(choice.message, "reasoning_content", "") or ""
    mean_logprob: float | str = ""
    content = getattr(getattr(choice, "logprobs", None), "content", None) or []
    values = [item.logprob for item in content if item.logprob is not None]
    if values:
        mean_logprob = round(statistics.mean(values), 6)
    response_dict = response.model_dump() if hasattr(response, "model_dump") else response.to_dict()
    _report_response(model_name, output_prompt, response_texts)
    time.sleep(0.1)
    return response_texts, response_dict, reasoning_text, mean_logprob


def CodeLlama_serverstandard_inference_function(
    model: Any,
    system_prompt: str = "You are a helpful programming assistant.",
    question_text: str = "Provide a concise response.",
    num_candidates: int = 1,
    temperature: float = 0.7,
    max_length: int = 1024,
    output_prompt: bool = False,
    tokenizer: Any | None = None,
) -> list[str]:
    """Run a locally loaded CodeLlama model without an external proxy."""

    if tokenizer is None:
        raise ValueError("A tokenizer is required for local CodeLlama inference.")
    import torch

    prompt = f"{system_prompt}\n\n{question_text}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_length,
            num_return_sequences=num_candidates,
        )
    responses = [
        tokenizer.decode(output[inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        for output in outputs
    ]
    _report_response("local-codellama", output_prompt, responses)
    return responses


def load_local_codellama(model_name: str) -> tuple[Any, Any]:
    """Load a local CodeLlama checkpoint without contacting a remote provider."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    return tokenizer, model


if __name__ == "__main__":
    print("single_generation.py exposes official model adapters; no API call was made.")
