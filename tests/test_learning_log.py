import json
from pathlib import Path

import pytest
from pydantic_ai import DeferredToolRequests, DeferredToolResults
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import ToolReturnPart

from app import tools
from app.agent import agent, collect_approvals, run_with_approval


@pytest.fixture
def log_dir(monkeypatch, tmp_path):
    target = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", target)
    return target


def pending_save():
    model = TestModel(call_tools=["save_learning_log"], custom_output_text="終了")
    result = agent.run_sync("学習記録を保存して", model=model)
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1
    return model, result


def test_no_write_before_approval(log_dir):
    pending_save()
    assert not log_dir.exists()


def test_approved_writes_exact_arguments_once(log_dir):
    model, pending = pending_save()
    call = pending.output.approvals[0]
    result = agent.run_sync(model=model, message_history=pending.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals={call.tool_call_id: True}))
    files = list(log_dir.glob("*.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text())
    assert record["title"] == call.args_as_dict()["title"]
    assert record["content"] == call.args_as_dict()["content"]
    assert not isinstance(result.output, DeferredToolRequests)
    returns = [p for m in result.all_messages() for p in m.parts
               if isinstance(p, ToolReturnPart) and p.tool_name == "save_learning_log"]
    assert len(returns) == 1 and json.loads(returns[0].content)["saved"] is True


def test_denied_never_writes(log_dir):
    model, pending = pending_save()
    call = pending.output.approvals[0]
    agent.run_sync(model=model, message_history=pending.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals={call.tool_call_id: False}))
    assert not log_dir.exists()


@pytest.mark.parametrize("answer,approved", [("y", True),("YES", True),("n",False),("",False)])
def test_cli_decisions(log_dir, answer, approved, capsys):
    _, pending = pending_save()
    call = pending.output.approvals[0]
    decisions = collect_approvals(pending.output, lambda prompt: answer)
    assert decisions.approvals[call.tool_call_id] is approved
    assert json.dumps(call.args_as_dict(), ensure_ascii=False, indent=2) in capsys.readouterr().out
    assert not log_dir.exists()


@pytest.mark.parametrize("interrupt",[EOFError,KeyboardInterrupt])
def test_input_end_denies(log_dir, interrupt):
    _, pending = pending_save()
    def stop(prompt): raise interrupt
    decisions = collect_approvals(pending.output, stop)
    assert not any(decisions.approvals.values())
    assert not log_dir.exists()


def test_invalid_input_does_not_approve(log_dir):
    _, pending = pending_save()
    answers = iter(["maybe", "n"])
    decisions = collect_approvals(pending.output, lambda prompt: next(answers))
    assert not any(decisions.approvals.values())


def test_cli_resume_and_cumulative_usage(log_dir):
    model = TestModel(call_tools=["save_learning_log"], custom_output_text="終了")
    result = run_with_approval(agent,"学習記録を保存して",model,lambda prompt:"y")
    assert result.output == "終了"
    assert result.usage.requests >= 2
    assert len(list(log_dir.glob('*.json'))) == 1


def test_generated_filename_stays_inside_directory(log_dir):
    result=json.loads(tools.save_learning_log("../../outside", "確認用の本文"))
    assert result['saved'] is True
    assert len(list(log_dir.glob('*.json'))) == 1
    assert not (log_dir.parent/'outside').exists()


def test_existing_records_are_not_overwritten(log_dir):
    tools.save_learning_log("見出し", "1件目")
    tools.save_learning_log("見出し", "2件目")
    files=list(log_dir.glob('*.json'))
    assert len(files)==2
    assert {json.loads(p.read_text())['content'] for p in files}=={'1件目','2件目'}


def test_symlink_directory_is_rejected(log_dir,tmp_path):
    other=tmp_path/'other';other.mkdir()
    log_dir.symlink_to(other,target_is_directory=True)
    assert json.loads(tools.save_learning_log('見出し','本文'))['saved'] is False
    assert not list(other.iterdir())


def test_blank_and_oversize_records_rejected(log_dir):
    for title,content in [('', '本文'),('見出し',' '),('a'*201,'本文'),('見出し','a'*10001)]:
        assert json.loads(tools.save_learning_log(title,content))['saved'] is False
    assert not log_dir.exists()


def test_quiz_uses_glossary_definition():
    quiz=json.loads(tools.make_quiz(2))
    assert quiz['answer'] in [t['definition'] for t in tools.load_glossary() if t['lecture']==2]
    assert '意味は' in quiz['question']


def test_quiz_handles_unknown_lecture():
    assert 'error' in json.loads(tools.make_quiz(0))
