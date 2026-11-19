"""Request fields sent to OpenAI chat models."""

from types import SimpleNamespace
from typing import Any

import pytest

from promptmachine_eval.runner import PromptTester, openai_chat_params


def test_gpt_6_luna_takes_a_completion_cap_and_effort_none() -> None:
    assert openai_chat_params("gpt-6-luna", 0.0, 10) == {
        "max_completion_tokens": 10,
        "reasoning_effort": "none",
        "temperature": 0.0,
    }


def test_gpt_6_2_sol_gets_no_temperature_or_max_tokens() -> None:
    assert openai_chat_params("gpt-6.2-sol", 0.7, 512) == {"max_completion_tokens": 512}


def test_gpt_4o_mini_keeps_its_fields() -> None:
    assert openai_chat_params("gpt-4o-mini", 0.7, 64) == {"temperature": 0.7, "max_tokens": 64}


def test_a_shut_down_id_names_its_replacement() -> None:
    with pytest.raises(ValueError, match="use gpt-6-luna instead"):
        openai_chat_params("gpt-3.5-turbo", 0.7, 64)


class _Completions:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        message = SimpleNamespace(content="ok")
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


async def test_new_fields_go_in_the_request_body_for_older_sdks() -> None:
    completions = _Completions()
    tester = PromptTester(openai_api_key="test-key")
    tester._clients["openai"] = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    await tester._call_openai("gpt-6-luna", "hi", None, 0.2, 32)

    assert completions.kwargs["extra_body"] == {"max_completion_tokens": 32, "reasoning_effort": "none"}
    assert completions.kwargs["temperature"] == 0.2
    assert "max_completion_tokens" not in completions.kwargs
    assert "reasoning_effort" not in completions.kwargs
