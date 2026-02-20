from __future__ import annotations
from typing import Any, Dict, List, Tuple
from openai import OpenAI

Messages = List[Dict[str, str]]

def _extract_reasoning_metrics(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Провайдеры возвращают usage по-разному.
    Достаём всё, что похоже на reasoning/think токены, и базовые токены.
    """
    usage = raw.get("usage") or {}
    metrics = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": None,
    }

    # Частые варианты у провайдеров: completion_tokens_details / output_tokens_details
    details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    if isinstance(details, dict):
        for k in ("reasoning_tokens", "thinking_tokens", "reasoning"):
            if k in details:
                metrics["reasoning_tokens"] = details.get(k)

    # Иногда reasoning могут класть прямо в usage
    for k in ("reasoning_tokens", "thinking_tokens"):
        if metrics["reasoning_tokens"] is None and k in usage:
            metrics["reasoning_tokens"] = usage.get(k)

    return metrics

def generate_thinking_response(
    client: OpenAI,
    model: str,
    messages: Messages,
) -> Tuple[str, Dict[str, Any]]:
    """
    Запрос к 'думающей' модели (Claude через OpenAI-compatible endpoint).
    Возвращает (answer_text, metrics_dict).
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    # OpenAI SDK objects -> dict (на новых версиях есть model_dump)
    raw = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)

    answer = ""
    if resp.choices and resp.choices[0].message:
        answer = resp.choices[0].message.content or ""

    metrics = _extract_reasoning_metrics(raw)
    metrics["model"] = raw.get("model")
    metrics["id"] = raw.get("id")
    metrics["finish_reason"] = (raw.get("choices") or [{}])[0].get("finish_reason")

    return answer, metrics
