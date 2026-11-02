"""第1回: 単発の質問、または会話履歴を保持する対話。

uv run python -m app.ask "LoRAとは何ですか？"
uv run python -m app.ask --chat
uv run python -m app.ask --chat "LoRAとは何ですか？"
対話中は /exit で終了、/reset で会話履歴を消去する。
"""

import argparse
import sys

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def print_usage(response) -> None:
    """取得できない値をゼロと混同しない。回答と分けて標準エラーに出す。"""
    usage = response.usage_metadata
    prompt = usage.prompt_token_count if usage else None
    output = usage.candidates_token_count if usage else None
    print(
        f"[tokens] 入力: {prompt if prompt is not None else '不明'} / "
        f"出力: {output if output is not None else '不明'}",
        file=sys.stderr,
    )


def response_text(response) -> str:
    print_usage(response)
    return response.text or "（テキストの回答がありません。安全性フィルタなどを確認してください。）"


def ask(question: str) -> str:
    """従来の単発呼び出しを維持する。"""
    with genai.Client(api_key=get_api_key()) as client:
        response = client.models.generate_content(model=MODEL, contents=question)
        return response_text(response)


def chat(initial_question: str | None = None) -> None:
    """同じ Chat を再利用し、SDK にユーザー・モデル双方の履歴を管理させる。"""
    with genai.Client(api_key=get_api_key()) as client:
        session = client.chats.create(model=MODEL)
        print("対話モード: /exit で終了、/reset で履歴を消去。")
        question = initial_question
        while True:
            if question is None:
                try:
                    question = input("あなた> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n対話を終了しました。")
                    return
            question = question.strip()
            if question == "/exit":
                return
            if question == "/reset":
                session = client.chats.create(model=MODEL)
                print("会話履歴を消去しました。")
            elif question:
                try:
                    print(f"AI> {response_text(session.send_message(question))}")
                except errors.APIError as e:
                    print(api_error_message(e.code, e.message), file=sys.stderr)
            question = None


def main() -> None:
    parser = argparse.ArgumentParser(description="LLMに質問する（--chat で会話を継続）")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--chat", action="store_true", help="会話履歴を保つ対話モード")
    args = parser.parse_args()
    print(f"[model] {MODEL}")
    try:
        if args.chat:
            chat(args.question)
        else:
            question = args.question or DEFAULT_QUESTION
            print(f"[質問] {question}\n")
            print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message), file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\n対話を終了しました。")


if __name__ == "__main__":
    main()
