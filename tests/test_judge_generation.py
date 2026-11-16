"""第3回の保存承認が評価用の回答生成でも守られることを検証する。"""
import json

from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.models.test import TestModel

from app import agent as learning_agent, judge, tools


def test_generate_answers_denies_save_without_writing(monkeypatch, tmp_path):
    target = Agent(output_type=[str, DeferredToolRequests])
    target.tool_plain(tools.save_learning_log, requires_approval=True)
    model = TestModel(custom_output_text="保存していません。")
    monkeypatch.setattr(learning_agent, "agent", target)
    monkeypatch.setattr(learning_agent, "build_model", lambda: model)
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path / "logs")
    trace = tmp_path / "trace.json"
    cases = judge.generate_answers([{"id": "01", "question": "合成メモを保存して"}], wait=0,
                                   deny_approvals=True, trace_path=trace)
    assert isinstance(cases[0]["answer"], str)
    assert not (tmp_path / "logs").exists()
    saved = json.loads(trace.read_text())
    assert saved["deny_approvals"] is True
    calls = [e for e in saved["cases"][0]["events"] if e["kind"] == "call"]
    assert any(e["name"] == "save_learning_log" for e in calls)
    returns = [e for e in saved["cases"][0]["events"] if e["kind"] == "return"]
    assert returns and not any('"saved": true' in e["content"] for e in returns)


def test_generate_answers_returns_actual_text_and_records_usage(monkeypatch, tmp_path):
    target = Agent()
    monkeypatch.setattr(learning_agent, "agent", target)
    monkeypatch.setattr(learning_agent, "build_model", lambda: TestModel(custom_output_text="確認済み\n第3回"))
    trace = tmp_path / "trace.json"
    result = judge.generate_answers([{"id": "01", "question": "説明して"}], wait=0, trace_path=trace)
    assert result[0]["answer"] == "確認済み\n第3回"
    assert json.loads(trace.read_text())["cases"][0]["usage"]["requests"] == 1
