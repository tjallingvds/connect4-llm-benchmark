"""Pluggable LLM connectors for the benchmark.

Two transports are supported:
  - "openai_compatible": works with OpenAI, DeepSeek, OpenRouter, Together, Groq,
    Mistral, Fireworks, local vLLM, Ollama (with OpenAI shim), etc. Just point at
    the right base_url.
  - "anthropic": native Anthropic Messages API.

Both expose a single function (board, player, valid_moves) -> {move, rationale}
that drops straight into the benchmark runner.
"""
import json
import re

from core import format_board_for_llm


PROMPT_TEMPLATE = """You are playing Connect 4. You are Player 1 (marked as '1'), playing against Player 2 (marked as '2').

CRITICAL: Column numbering is 0-6 from LEFT TO RIGHT.

BOARD STATE (top to bottom, 0s are empty):
{board}

VALID MOVES: {valid_moves}

YOUR TASK: Select the single best move that either:
1. Wins the game immediately (gets 4 in a row)
2. Blocks opponent from winning on their next turn
3. Creates multiple winning threats
4. Maximizes your winning chances

Think strategically, then respond with your move choice and concise reasoning.

RESPONSE FORMAT: JSON with 'move' (integer 0-6) and 'rationale' (one sentence explaining your choice)."""

SYSTEM_PROMPT = (
    "You are an expert Connect 4 player. Always respond with valid JSON containing "
    "'move' (integer 0-6) and 'rationale' (string one sentence only)."
)


def _build_prompt(board, valid_moves):
    return PROMPT_TEMPLATE.format(
        board=format_board_for_llm(board),
        valid_moves=valid_moves,
    )


def _parse_json_response(text):
    """Best-effort JSON extraction. Some models wrap JSON in prose or code fences."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))
    obj_match = re.search(r"\{[^{}]*\"move\"[^{}]*\}", text, re.DOTALL)
    if obj_match:
        return json.loads(obj_match.group(0))
    raise ValueError(f"could not parse JSON from response: {text[:200]}")


def make_openai_compatible_mover(api_key, model, base_url=None,
                                 temperature=0.1, max_tokens=200,
                                 max_retries=3, on_failure=None):
    """Build a move-getter that talks to any OpenAI-compatible /chat/completions endpoint."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    def get_move(board, player, valid_moves):
        prompt = _build_prompt(board, valid_moves)
        last_err = None
        for _ in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content.strip()
                result = _parse_json_response(content)
                move = int(result["move"])
                rationale = str(result.get("rationale", ""))
                if move in valid_moves:
                    return {"move": move, "rationale": rationale}
                last_err = f"model returned out-of-range move {move}"
            except Exception as e:
                last_err = repr(e)
                # Some providers reject response_format — retry without it.
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    content = resp.choices[0].message.content.strip()
                    result = _parse_json_response(content)
                    move = int(result["move"])
                    rationale = str(result.get("rationale", ""))
                    if move in valid_moves:
                        return {"move": move, "rationale": rationale}
                    last_err = f"model returned out-of-range move {move}"
                except Exception as e2:
                    last_err = repr(e2)

        if on_failure:
            on_failure(last_err)
        return {"move": valid_moves[0], "rationale": f"FALLBACK after error: {last_err}"}

    return get_move


def make_anthropic_mover(api_key, model, temperature=0.1, max_tokens=400,
                         max_retries=3, on_failure=None):
    """Build a move-getter for the native Anthropic Messages API."""
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)

    def get_move(board, player, valid_moves):
        prompt = _build_prompt(board, valid_moves)
        last_err = None
        for _ in range(max_retries):
            try:
                msg = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system="You are an expert Connect 4 player. Respond with strict JSON: {\"move\": <int 0-6>, \"rationale\": \"<one sentence>\"}.",
                    messages=[{"role": "user", "content": prompt}],
                )
                content = msg.content[0].text.strip()
                result = _parse_json_response(content)
                move = int(result["move"])
                rationale = str(result.get("rationale", ""))
                if move in valid_moves:
                    return {"move": move, "rationale": rationale}
                last_err = f"model returned out-of-range move {move}"
            except Exception as e:
                last_err = repr(e)

        if on_failure:
            on_failure(last_err)
        return {"move": valid_moves[0], "rationale": f"FALLBACK after error: {last_err}"}

    return get_move


PROVIDER_PRESETS = {
    "OpenAI": {
        "transport": "openai_compatible",
        "base_url": None,
        "default_model": "gpt-4o",
        "key_env": "OPENAI_API_KEY",
    },
    "Anthropic": {
        "transport": "anthropic",
        "base_url": None,
        "default_model": "claude-sonnet-4-5-20250929",
        "key_env": "ANTHROPIC_API_KEY",
    },
    "DeepSeek": {
        "transport": "openai_compatible",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "Groq": {
        "transport": "openai_compatible",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
    },
    "OpenRouter": {
        "transport": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "openai/gpt-4o-mini",
        "key_env": "OPENROUTER_API_KEY",
    },
    "Together": {
        "transport": "openai_compatible",
        "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "key_env": "TOGETHER_API_KEY",
    },
    "Custom (OpenAI-compatible)": {
        "transport": "openai_compatible",
        "base_url": "",
        "default_model": "",
        "key_env": "",
    },
}


def build_mover(provider, api_key, model, base_url=None, on_failure=None):
    preset = PROVIDER_PRESETS.get(provider)
    if not preset:
        raise ValueError(f"unknown provider: {provider}")
    if preset["transport"] == "anthropic":
        return make_anthropic_mover(api_key, model, on_failure=on_failure)
    return make_openai_compatible_mover(
        api_key=api_key, model=model,
        base_url=base_url or preset.get("base_url") or None,
        on_failure=on_failure,
    )
