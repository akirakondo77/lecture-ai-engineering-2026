from types import SimpleNamespace
from unittest.mock import Mock

from app import ask


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        text="answer",
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=34,
        ),
    )
    client = Mock()
    client.models.generate_content.return_value = response
    monkeypatch.setattr(ask.genai, "Client", Mock(return_value=client))
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    assert ask.ask("question") == "answer"
    assert capsys.readouterr().out == "[トークン数] 入力: 12, 出力: 34\n"


def test_ask_reports_missing_usage_metadata(monkeypatch, capsys):
    response = SimpleNamespace(text="answer", usage_metadata=None)
    client = Mock()
    client.models.generate_content.return_value = response
    monkeypatch.setattr(ask.genai, "Client", Mock(return_value=client))
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    assert ask.ask("question") == "answer"
    assert capsys.readouterr().out == "[トークン数] 使用量情報を取得できませんでした\n"
