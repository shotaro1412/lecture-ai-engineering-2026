"""第5回: 同一条件でRAG取得方法を比較し、回答・trace・時間・使用量を保存する。

uv run python scripts/compare_rag_neighbors.py --group homework
uv run python scripts/compare_rag_neighbors.py --group exercise
APIキーはGEMINI_API_KEYから取得。結果はlectures/05/verification/へ保存する。
"""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from google.genai.types import HttpRetryOptions
from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.usage import UsageLimits

from app import agent as learning_agent, retrieval, tools
from app.config import MODEL, RAG_SOURCE, RAG_DB, get_api_key

ROOT = Path(__file__).resolve().parent.parent
CASES = [
    {"id": "single", "query": "LoRA QLoRA 違い",
     "question": "第2回の説明では、LoRAは何を固定し何を学習しますか？ QLoRAは何を加えますか？",
     "expected_pages": ["lec02-main-p052"], "expected": "元の重みを固定して小さな差分を学習し、QLoRAでは元の重みを4bitに圧縮する。"},
    {"id": "memory", "query": "コンパクション セッションをまたぐメモリ 日々のメモ 長期メモリ",
     "question": "コンパクションとセッションをまたぐメモリの違いは？ 日々のメモは全部長期メモリになりますか？",
     "expected_pages": ["lec03-main-p013", "lec03-main-p039", "lec03-main-p040"], "expected": "コンパクションは同じセッションの履歴の要約。外部のメモリは次のセッションで読み戻す。長期メモリには選んだものだけ移す。"},
    {"id": "absent", "query": "今回作ったRAG 実測平均応答時間 正答率",
     "question": "今回作ったRAGの実測平均応答時間と正答率は、それぞれいくつですか？",
     "expected_pages": [], "expected": "講義資料には今回の実測ログがないので、数値を捏造せず確認できないと答える。"},
    {"id": "adjacent", "query": "LoRA QLoRA 違い",
     "question": "LoRAとQLoRAの違いに加えて、その説明の直後のチャットボットのページで、形式だけ守らせるなら何が先か、FTが効く振る舞いの例は何かを説明してください。",
     "expected_pages": ["lec02-main-p052", "lec02-main-p053"], "expected": "LoRA/QLoRAを説明し、形式だけならAPIの構造化出力が先、FTは口調や根拠の付け方などの振る舞いに効くとp.53を根拠に述べる。"},
]


def model():
    # この端末では既定httpx2経路がタイムアウトしたため、SDK対応のhttpx経路を使う。
    # pydantic-ai 2.57.0では使用可能だがv3での削除予定に注意する。
    return GoogleModel(MODEL, provider=GoogleProvider(
        api_key=get_api_key(), http_client=httpx.AsyncClient(timeout=90),
        retry_options=HttpRetryOptions(attempts=3, initial_delay=2, max_delay=8),
    ))


def record_result(result, elapsed):
    trace = []
    for message in result.all_messages():
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                trace.append({"kind": "call", "name": part.tool_name, "args": part.args_as_dict()})
            elif isinstance(part, ToolReturnPart):
                trace.append({"kind": "return", "name": part.tool_name, "content": str(part.content)})
    output = io.StringIO()
    with redirect_stdout(output):
        learning_agent.print_trace(result, elapsed)
    return {"answer": result.output, "seconds": round(elapsed, 3), "usage": asdict(result.usage),
            "trace": trace, "trace_text": output.getvalue()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['exercise', 'homework'], required=True)
    args = parser.parse_args()
    dest = ROOT / 'lectures/05/verification' / (args.group + '.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    saved = json.loads(dest.read_text()) if dest.exists() else {
        "model": MODEL, "temperature": 0, "started_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(RAG_SOURCE.read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256(RAG_DB.read_bytes()).hexdigest(),
        "cases": CASES, "runs": [],
    }
    if (saved['model'] != MODEL or saved['cases'] != CASES
            or saved['source_sha256'] != hashlib.sha256(RAG_SOURCE.read_bytes()).hexdigest()
            or saved['index_sha256'] != hashlib.sha256(RAG_DB.read_bytes()).hexdigest()):
        raise ValueError('既存結果のモデル・資料と一致しません。別の保存先を使ってください。')
    def store(row):
        saved['runs'].append(row)
        dest.write_text(json.dumps(saved, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        print(row['id'], 'OK' if 'answer' in row else row.get('error'), flush=True)
    done = {r['id'] for r in saved['runs'] if 'answer' in r}
    if args.group == 'exercise':
        # 演習の通常検索は外側の環境変数に左右されないよう固定する。
        retrieval.RAG_INCLUDE_NEIGHBORS = False
        q = CASES[0]['question']
        modes = [
            ('full-context', learning_agent.FULL_CONTEXT_INSTRUCTIONS, []),
            ('vector', learning_agent.INSTRUCTIONS, [tools.search_materials]),
            ('tree', learning_agent.TREE_INSTRUCTIONS, [tools.browse_tree]),
            ('graph', learning_agent.GRAPH_INSTRUCTIONS, [tools.search_graph]),
        ]
        for name, instructions, functions in modes:
            if name in done:
                continue
            try:
                target = Agent(instructions=instructions, tools=functions, model_settings={'temperature': 0})
                started = perf_counter()
                prompt = learning_agent.FULL_CONTEXT_PROMPT.format(materials=retrieval.get_all_materials(), question=q) if name == 'full-context' else q
                result = target.run_sync(prompt, model=model(), usage_limits=UsageLimits(request_limit=16, tool_calls_limit=24))
                store({'id': name, 'question': q, **record_result(result, perf_counter()-started)})
            except Exception as error:
                store({'id': name, 'error': type(error).__name__ + ': ' + str(error).replace(get_api_key(), '[REDACTED]')[:500]})
    else:
        for number, case in enumerate(CASES):
            variants = [False, True] if number % 2 == 0 else [True, False]
            # 両方式で検索語・top-k・検索ヒットが同じことを確かめる。
            pair_seeds = None
            for expanded in variants:
                rid = case['id'] + ('-neighbors' if expanded else '-baseline')
                if rid in done:
                    continue
                try:
                    started = perf_counter()
                    hits = retrieval.search(case['query'], include_neighbors=False)
                    seed_ids = [p['page_id'] for p in hits]
                    if pair_seeds is not None and seed_ids != pair_seeds:
                        raise ValueError('比較中に検索ヒットが変化しました。')
                    pair_seeds = seed_ids
                    pages = retrieval.expand_neighbors(hits, retrieval.load_pages()) if expanded else hits
                    retrieval_seconds = perf_counter()-started
                    materials = json.dumps({'results': pages}, ensure_ascii=False)
                    prompt = learning_agent.FULL_CONTEXT_PROMPT.format(materials=materials, question=case['question'])
                    target = Agent(instructions=learning_agent.FULL_CONTEXT_INSTRUCTIONS, model_settings={'temperature': 0})
                    result = target.run_sync(prompt, model=model(), usage_limits=UsageLimits(request_limit=2))
                    row = {'id': rid, 'question': case['question'], 'query': case['query'], 'seed_ids': seed_ids,
                           'retrieved_ids': [p['page_id'] for p in pages], 'page_count': len(pages),
                           'retrieval_seconds': round(retrieval_seconds, 3),
                           'retrieval_trace': {'kind': 'retrieval', 'include_neighbors': expanded, 'results': pages},
                           **record_result(result, perf_counter()-started)}
                    store(row)
                except Exception as error:
                    store({'id': rid, 'error': type(error).__name__ + ': ' + str(error).replace(get_api_key(), '[REDACTED]')[:500]})


if __name__ == '__main__':
    main()
