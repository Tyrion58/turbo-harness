"""Unified API client supporting OpenAI-compatible (litellm) and Vertex AI backends.

Set API_PROVIDER="vertex" to route calls through AnthropicVertex.
Default ("openai") uses litellm, preserving existing behavior.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

API_PROVIDER = os.environ.get("API_PROVIDER", "openai").lower()

VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID", "")
VERTEX_REGION = os.environ.get("VERTEX_REGION", "")
VERTEX_MODEL = os.environ.get("VERTEX_MODEL", "claude-opus-4-6")


@dataclass
class CompletionMessage:
    content: str
    role: str = "assistant"


@dataclass
class CompletionChoice:
    message: CompletionMessage
    index: int = 0


@dataclass
class CompletionResponse:
    choices: List[CompletionChoice] = field(default_factory=list)
    model: str = ""
    _hidden_params: dict = field(default_factory=lambda: {"response_cost": 0.0})


_vertex_client = None


def _get_vertex_client():
    global _vertex_client
    if _vertex_client is None:
        from anthropic import AnthropicVertex

        if not VERTEX_PROJECT_ID or not VERTEX_REGION:
            raise ValueError(
                "VERTEX_PROJECT_ID and VERTEX_REGION must be set when API_PROVIDER=vertex"
            )
        _vertex_client = AnthropicVertex(
            project_id=VERTEX_PROJECT_ID,
            region=VERTEX_REGION,
        )
    return _vertex_client


def _extract_system_and_messages(
    messages: List[Dict[str, str]],
) -> tuple[Optional[str], List[Dict[str, str]]]:
    system_parts = []
    non_system = []
    for msg in messages:
        if msg["role"] == "system":
            system_parts.append(msg["content"])
        else:
            non_system.append(msg)
    system = "\n\n".join(system_parts) if system_parts else None
    return system, non_system


def _vertex_completion(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 4096,
    **kwargs,
) -> CompletionResponse:
    client = _get_vertex_client()
    system, non_system = _extract_system_and_messages(messages)

    create_kwargs = {
        "model": model or VERTEX_MODEL,
        "max_tokens": max_tokens,
        "messages": non_system,
        "temperature": temperature,
    }
    if system:
        create_kwargs["system"] = system

    response = client.messages.create(**create_kwargs)

    return CompletionResponse(
        choices=[
            CompletionChoice(
                message=CompletionMessage(content=response.content[0].text)
            )
        ],
        model=response.model,
    )


def completion(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    timeout: float = 120.0,
    base_url: Optional[str] = None,
    max_tokens: int = 4096,
    **kwargs,
) -> CompletionResponse:
    """Call an LLM with automatic provider routing.

    When API_PROVIDER="vertex", calls AnthropicVertex.
    Otherwise falls through to litellm (OpenAI-compatible).

    Returns a CompletionResponse with .choices[0].message.content interface.
    """
    if API_PROVIDER == "vertex":
        return _vertex_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    from litellm import completion as litellm_completion

    response = litellm_completion(
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=timeout,
        base_url=base_url,
        **kwargs,
    )
    return response
