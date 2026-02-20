from __future__ import annotations
import json
import time
from typing import Dict, List

import httpx
from openai import OpenAI

from load_env import get_settings
from test_agent import generate_thinking_response

from load_env import get_settings
from test_agent import generate_thinking_response

from pathlib import Path
import os

print("ENV exists:", (Path(__file__).resolve().parent / ".env").exists())
print("OPENAI_API_KEY loaded:", bool(os.getenv("OPENAI_API_KEY")))
print("OPENAI_BASE_URL:", os.getenv("OPENAI_BASE_URL"))



Message = Dict[str, str]

EXIT_COMMANDS = {"exit", "quit", "q", "выход"}

def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

def load_history(path: str) -> List[Message]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except FileNotFoundError:
        return []
    except Exception:
        return []
    return []

def save_history(path: str, history: List[Message]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def choose_mode() -> str:
    print("\nВыберите режим:")
    print("1 — Думающая модель (по умолчанию)")
    print("2 — Обычная модель")
    choice = input("Введите 1 или 2 (Enter = 1): ").strip()
    return "normal" if choice == "2" else "thinking"

def generate_normal_response(client: OpenAI, model: str, messages: List[Message]) -> str:
    resp = client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content or ""

def main() -> None:
    s = get_settings()

    if not s.api_key or not s.base_url:
        print("Ошибка: заполни OPENAI_API_KEY и OPENAI_BASE_URL в .env")
        return

    log("Старт ассистента")
    log(f"Base URL: {s.base_url}")
    log(f"History: {s.history_path}")
    log(f"Timeout: {s.timeout_seconds}s")
    log(f"Normal model: {s.normal_model}")
    log(f"Thinking model: {s.thinking_model}")

    http_client = httpx.Client(timeout=s.timeout_seconds)
    client = OpenAI(api_key=s.api_key, base_url=s.base_url, http_client=http_client)

    history = load_history(s.history_path)
    if not history or history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": "Вы полезный ассистент. Отвечайте на русском языке."})

    mode = choose_mode()
    log(f"Выбран режим: {mode}")

    while True:
        user_text = input("\nВы: ").strip()
        if not user_text:
            continue
        if user_text.lower() in EXIT_COMMANDS:
            save_history(s.history_path, history)
            print("\nИстория сохранена. Пока!")
            break

        history.append({"role": "user", "content": user_text})

        try:
            if mode == "thinking":
                answer, metrics = generate_thinking_response(client, s.thinking_model, history)
                history.append({"role": "assistant", "content": answer})
                save_history(s.history_path, history)

                print("\n--- Reasoning / Usage ---")
                print(f"model: {metrics.get('model')}")
                print(f"id: {metrics.get('id')}")
                print(f"prompt_tokens: {metrics.get('prompt_tokens')}")
                print(f"completion_tokens: {metrics.get('completion_tokens')}")
                print(f"total_tokens: {metrics.get('total_tokens')}")
                print(f"reasoning_tokens: {metrics.get('reasoning_tokens') if metrics.get('reasoning_tokens') is not None else 'n/a'}")
                print(f"finish_reason: {metrics.get('finish_reason')}")
                print("-------------------------")

                print("\nАссистент:", answer)

            else:
                answer = generate_normal_response(client, s.normal_model, history)
                history.append({"role": "assistant", "content": answer})
                save_history(s.history_path, history)
                print("\nАссистент:", answer)

        except httpx.TimeoutException:
            log("TIMEOUT: запрос к модели превысил лимит времени")
            print("Ошибка: таймаут запроса. Попробуй ещё раз.")
        except Exception as e:
            log(f"ERROR: {type(e).__name__}: {e}")
            print(f"Ошибка запроса: {e}")

if __name__ == "__main__":
    main()
