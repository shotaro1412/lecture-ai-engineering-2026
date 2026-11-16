"""第4回評価CLI。通信はSDKが対応するhttpxクライアントで行う。

この環境では既定のhttpx2の接続がタイムアウトするため、公式providerの
http_client引数でhttpxを明示する。モデル・プロンプト・採点処理はapp.judgeと同じ。
使い方: uv run python scripts/run_day04_eval.py --cases ... --labels ...
APIキーは通常どおり環境変数GEMINI_API_KEYから読み込む。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from google.genai.types import HttpRetryOptions
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from app import agent, judge
from app.config import MODEL, get_api_key


def build_model(name: str) -> GoogleModel:
    return GoogleModel(name, provider=GoogleProvider(
        api_key=get_api_key(), http_client=httpx.AsyncClient(timeout=60.0),
        retry_options=HttpRetryOptions(attempts=3, initial_delay=2, max_delay=8),
    ))


if __name__ == "__main__":
    agent.build_model = lambda: build_model(MODEL)
    judge.build_llm_model = build_model
    judge.main()
