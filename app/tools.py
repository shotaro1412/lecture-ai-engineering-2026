"""第3回: エージェントに持たせるツール。

ここに書いた関数の docstring と型ヒントが、そのままモデルへの「ツールの説明文」になる。
説明文を書き換えると、モデルがツールを使うかどうか・どんな引数で呼ぶかが変わる。
"""

import json
from functools import cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_RESULTS = 3


@cache
def load_glossary() -> list[dict]:
    with open(DATA_DIR / "glossary.json", encoding="utf-8") as f:
        return json.load(f)["terms"]


def _normalize(text: str) -> str:
    return text.casefold().replace(" ", "").replace("　", "")


def _names(entry: dict) -> list[str]:
    return [_normalize(entry["term"]), _normalize(entry["id"])] + [
        _normalize(a) for a in entry["aliases"]
    ]


def search_glossary(query: str) -> str:
    """講座の用語集から用語を検索し、定義と、どの回で扱うかを返す。

    AIエンジニアリングの用語や、この講座（AIE2026）に固有の言葉について
    答える前に使うこと。query には調べたい用語を1つだけ入れる（例: "LoRA"）。
    """
    q = _normalize(query)
    if not q:
        return "検索語が空です。調べたい用語を1つ指定してください。"
    glossary = load_glossary()

    exact = [e for e in glossary if q in _names(e)]
    # 「r」のような短い別名が何にでも当たらないよう、名前側が短いものは除く
    partial = [
        e for e in glossary
        if e not in exact and any(q in n or (len(n) >= 3 and n in q) for n in _names(e))
    ]
    in_definition = [
        e for e in glossary
        if e not in exact and e not in partial and q in _normalize(e["definition"])
    ]
    hits = (exact + partial + in_definition)[:MAX_RESULTS]

    if not hits:
        return f"用語集に「{query}」は見つかりませんでした。"
    return json.dumps(
        [
            {
                "term": e["term"],
                "lecture": e["lecture"],
                "definition": e["definition"],
                "related": e["related"],
            }
            for e in hits
        ],
        ensure_ascii=False,
    )


def get_lecture_summary(lecture: int) -> str:
    """指定した回（1始まりの整数）の講義の要点を返す。

    「第3回では何を学ぶ？」のように、回の内容や関係を聞かれたときに使う。
    """
    path = DATA_DIR / "lectures" / f"{lecture:02d}.md"
    if not path.exists():
        available = sorted(p.stem for p in (DATA_DIR / "lectures").glob("*.md"))
        return f"第{lecture}回の要点はまだありません。用意されている回: {', '.join(available)}"
    return path.read_text(encoding="utf-8")


def make_quiz(lecture: int) -> str:
    """指定した回の用語から1つ選び、確認問題と正解を返す。

    「第2回の確認問題を1問出して」のように復習問題を求められたときに使う。
    lecture は1始まりの回番号。該当する用語がなければ、その旨を返す。
    """
    import random

    terms = [term for term in load_glossary() if term["lecture"] == lecture]
    if not terms:
        return json.dumps({"error": f"第{lecture}回の用語はありません。"}, ensure_ascii=False)
    term = random.choice(terms)
    return json.dumps({"question": f"「{term['term']}」の意味は？", "answer": term["definition"]}, ensure_ascii=False)


LEARNING_LOG_DIR = Path(__file__).resolve().parent.parent / "learning_log"


def save_learning_log(title: str, content: str) -> str:
    """利用者が保存を頼んだ学習メモを、承認後にlearning_log/へ新規保存する。

    title は短い見出し、content は保存する本文。パスは指定できない。
    単なる用語の質問には使わず、学習記録の保存を明示的に依頼されたときだけ使う。
    登録側のrequires_approval=Trueにより、この関数を呼ぶ前に保存内容を確認する。
    """
    from datetime import datetime, timezone
    import os
    from uuid import uuid4

    if not title.strip() or not content.strip():
        return json.dumps({"saved": False, "error": "見出しと本文を空にすることはできません。"}, ensure_ascii=False)
    if len(title) > 200 or len(content) > 10000:
        return json.dumps({"saved": False, "error": "見出しは200文字、本文は10000文字以内にしてください。"}, ensure_ascii=False)
    try:
        # モデルの引数をファイル名にしない。既存ファイルの上書きも行わない。
        if LEARNING_LOG_DIR.is_symlink():
            return json.dumps({"saved": False, "error": "保存先がシンボリックリンクのため保存しません。"}, ensure_ascii=False)
        LEARNING_LOG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        filename = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid4().hex}.json"
        record = {"title": title, "content": content, "created_at": datetime.now(timezone.utc).isoformat()}
        path = LEARNING_LOG_DIR / filename
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    except OSError as error:
        return json.dumps({"saved": False, "error": f"保存できませんでした（{type(error).__name__}）。"}, ensure_ascii=False)
    return json.dumps({"saved": True, "path": f"learning_log/{filename}"}, ensure_ascii=False)
