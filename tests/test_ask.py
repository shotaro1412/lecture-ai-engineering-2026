"""実SDKで履歴を検証する。通信だけ置き換え、APIキーや無料枠を使わない。"""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from google import genai
from google.genai import errors, types
from app import ask


def reply(text="テスト回答", usage=True):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=12, candidates_token_count=7) if usage else None,
    )


@pytest.fixture
def sdk(monkeypatch):
    client = genai.Client(api_key="unit-test-placeholder")
    calls, responses = [], []
    def generate(**kwargs):
        calls.append(deepcopy(kwargs))
        item = responses.pop(0) if responses else reply()
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setattr(client.models, "generate_content", generate)
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "unit-test-placeholder")
    return SimpleNamespace(client=client, calls=calls, responses=responses)


def inputs(monkeypatch, *items):
    iterator = iter(items)
    monkeypatch.setattr("builtins.input", lambda prompt: next(iterator))


def test_history_reaches_second_request(sdk, monkeypatch):
    sdk.responses.extend([reply("最初の回答"), reply("続き")])
    inputs(monkeypatch, "私はshotaです", "私の名前は？", "/exit")
    ask.chat()
    history = sdk.calls[1]["contents"]
    assert [c.role for c in history] == ["user", "model", "user"]
    assert [c.parts[0].text for c in history] == ["私はshotaです", "最初の回答", "私の名前は？"]


def test_reset_discards_previous_turns(sdk, monkeypatch):
    inputs(monkeypatch, "前の質問", "/reset", "新しい質問", "/exit")
    ask.chat()
    assert len(sdk.calls[1]["contents"]) == 1
    assert sdk.calls[1]["contents"][0].parts[0].text == "新しい質問"


def test_initial_question_and_blank_input(sdk, monkeypatch):
    inputs(monkeypatch, "  ", "/exit")
    ask.chat("最初の質問")
    assert len(sdk.calls) == 1


@pytest.mark.parametrize("termination", [EOFError, KeyboardInterrupt])
def test_keyboard_or_eof_ends_chat(sdk, monkeypatch, termination):
    def stop(prompt):
        raise termination
    monkeypatch.setattr("builtins.input", stop)
    ask.chat()
    assert not sdk.calls


def test_failed_turn_does_not_pollute_history(sdk, monkeypatch, capsys):
    sdk.responses.extend([reply("前の回答"), errors.APIError(429, {"message": "quota"}), reply()])
    inputs(monkeypatch, "成功", "失敗", "次の質問", "/exit")
    ask.chat()
    assert [c.parts[0].text for c in sdk.calls[2]["contents"]] == ["成功", "前の回答", "次の質問"]
    assert "レート制限" in capsys.readouterr().err


def test_single_question_and_tokens(sdk, capsys):
    assert ask.ask("LoRAとは？") == "テスト回答"
    assert sdk.calls[0]["contents"] == "LoRAとは？"
    assert "入力: 12 / 出力: 7" in capsys.readouterr().err


def test_missing_usage_is_unknown(capsys):
    ask.print_usage(reply(usage=False))
    assert "入力: 不明 / 出力: 不明" in capsys.readouterr().err


def test_zero_usage_is_zero(capsys):
    ask.print_usage(types.GenerateContentResponse(usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=0, candidates_token_count=0)))
    assert "入力: 0 / 出力: 0" in capsys.readouterr().err


def test_no_text_is_explained():
    assert "テキストの回答がありません" in ask.response_text(types.GenerateContentResponse())


def test_cli_default(sdk, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["app.ask"])
    ask.main()
    assert sdk.calls[0]["contents"] == ask.DEFAULT_QUESTION
    assert "テスト回答" in capsys.readouterr().out


def test_cli_error_exit(sdk, monkeypatch):
    sdk.responses.append(errors.APIError(503, {"message": "busy"}))
    monkeypatch.setattr("sys.argv", ["app.ask", "質問"])
    with pytest.raises(SystemExit) as error:
        ask.main()
    assert error.value.code == 1
