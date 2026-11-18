"""Request fields sent to OpenAI chat models."""

import pytest

from promptmachine_eval.runner import openai_chat_params


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
