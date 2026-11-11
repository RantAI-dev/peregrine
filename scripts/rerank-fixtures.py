#!/usr/bin/env python3
"""Generate the S4.6b rerank fixtures from llama.cpp.

The fixtures under `crates/peregrine-models/tests/fixtures/bert/rerank/` hold, for a
fixed set of (query, document) pairs:

  texts.tsv            one `query<TAB>document` line per pair, backslash-escaped
  ids.txt              llama.cpp's own pair encoding, a Python list per line
  scores-<file>.f32    llama.cpp's /v1/rerank score per line, little-endian f32

The ids come from `scripts/tokenizer-pair-oracle.cpp`, which calls llama.cpp's
`format_prompt_rerank` rather than a re-implementation. The scores come from a
`llama-server --reranking` built at v0.9.0 (`cd959ba`), one request per query.

    python3 scripts/rerank-fixtures.py \\
        --pairs /tmp/pairoracle \\
        --base-url http://127.0.0.1:18111 \\
        --out crates/peregrine-models/tests/fixtures/bert/rerank \\
        --scores scores-llama-q8_0.f32 --model bge-reranker-v2-m3-Q8_0.gguf

The pair set is deterministic: 40 queries of eight documents each, mixed lengths and
eight languages, since the model is multilingual.
"""

import argparse
import json
import os
import struct
import sys
import tempfile
import urllib.error
import urllib.request

# Eight languages, five queries each: the exact queries the fixtures use.
QUERIES = [
    # English
    "What is the capital of France?",
    "How does a database index speed up a query?",
    "Which cooking method keeps vegetables crisp?",
    "What caused the 2008 financial crisis?",
    "How do vaccines train the immune system?",
    # Chinese
    "法国的首都是哪里？",
    "数据库索引如何加快查询速度？",
    "哪种烹饪方法能保持蔬菜脆嫩？",
    "2008年金融危机的原因是什么？",
    "疫苗如何训练免疫系统？",
    # French
    "Quelle est la capitale de la France ?",
    "Comment un index de base de données accélère-t-il une requête ?",
    "Quelle cuisson garde les légumes croquants ?",
    "Quelle est la cause de la crise financière de 2008 ?",
    "Comment les vaccins entraînent-ils le système immunitaire ?",
    # German
    "Was ist die Hauptstadt von Frankreich?",
    "Wie beschleunigt ein Datenbankindex eine Abfrage?",
    "Welche Gar methode hält Gemüse knackig?",
    "Was verursachte die Finanzkrise 2008?",
    "Wie trainieren Impfstoffe das Immunsystem?",
    # Spanish
    "¿Cuál es la capital de Francia?",
    "¿Cómo acelera un índice de base de datos una consulta?",
    "¿Qué cocción mantiene crujientes las verduras?",
    "¿Qué causó la crisis financiera de 2008?",
    "¿Cómo entrenan las vacunas el sistema inmunitario?",
    # Russian
    "Какая столица Франции?",
    "Как индекс базы данных ускоряет запрос?",
    "Какой способ приготовления сохраняет овощи хрустящими?",
    "Что вызвало финансовый кризис 2008 года?",
    "Как вакцины тренируют иммунную систему?",
    # Arabic
    "ما هي عاصمة فرنسا؟",
    "كيف يسرع فهرس قاعدة البيانات الاستعلام؟",
    "أي طريقة طهي تحافظ على قرمشة الخضار؟",
    "ما سبب الأزمة المالية عام 2008؟",
    "كيف تدرب اللقاحات جهاز المناعة؟",
    # Japanese
    "フランスの首都はどこですか？",
    "データベースのインデックスはなぜ検索を速くするのですか？",
    "野菜の歯ごたえを保つ調理法はどれですか？",
    "2008年の金融危機の原因は何ですか？",
    "ワクチンはどのように免疫系を訓練しますか？",
]

# The answer and a related-but-wrong answer, per language, in the order of QUERIES.
ANSWERS = [
    "Paris is the capital of France.",
    "数据库索引通过减少需要扫描的行数来加快查询速度。",
    "Paris est la capitale de la France.",
    "Paris ist die Hauptstadt von Frankreich.",
    "París es la capital de Francia.",
    "Париж — столица Франции.",
    "باريس هي عاصمة فرنسا.",
    "フランスの首都はパリです。",
]
WRONG = [
    "Lyon is the third largest city in France and a centre of gastronomy.",
    "索引是一种数据结构，它保存了排好序的键和指向行的指针。",
    "Lyon est la troisième ville de France et un centre gastronomique.",
    "Lyon ist die drittgrößte Stadt Frankreichs und ein Zentrum der Gastronomie.",
    "Lyon es la tercera ciudad de Francia y un centro gastronómico.",
    "Лион — третий по величине город Франции.",
    "ليون هي ثالث أكبر مدينة في فرنسا.",
    "リヨンはフランスで三番目に大きな都市です。",
]

# Five medium English passages, one per subject; a query's documents 2..5 are four of
# them, rotated so every query sees a different mix.
PASSAGES = [
    "A database index is a data structure that improves the speed of data retrieval operations on a table at the cost of additional writes and storage space to maintain the index structure.",
    "The capital city of France is Paris, which lies on the Seine and has been the country's political centre since the tenth century, long before the modern republic.",
    "Boiling vegetables in a large volume of water dissolves water-soluble vitamins; steaming or stir-frying over high heat keeps them crisp and retains more of their colour and nutrients.",
    "The 2008 financial crisis followed the collapse of the United States housing bubble, the failure of Lehman Brothers, and a wave of mortgage-backed securities whose risk had been mispriced across the global banking system.",
    "Vaccines introduce a harmless fragment or inactivated form of a pathogen so that the adaptive immune system can build memory B and T cells without the risks of the disease itself.",
]


def escape(text):
    out = []
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        else:
            out.append(ch)
    return "".join(out)


def corpus(n):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(
        here, "..", "crates", "peregrine-models", "tests", "fixtures", "bert", "texts.txt"
    )
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[n % len(lines)]


def pairs(qi):
    """The eight documents of query `qi`, in fixture order."""
    lang = qi // 5
    docs = [ANSWERS[lang], WRONG[lang]]
    docs += [PASSAGES[(qi + k) % len(PASSAGES)] for k in range(4)]
    docs.append(corpus(7 + qi * 3))
    docs.append(corpus(41 + qi * 5))
    return [(QUERIES[qi], d) for d in docs]


def ids_from_oracle(oracle, model, pairs):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        for q, d in pairs:
            f.write(escape(q) + "\t" + escape(d) + "\n")
        path = f.name
    try:
        text = subprocess_output([oracle, model, path])
    finally:
        os.unlink(path)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def subprocess_output(cmd):
    import subprocess

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("failed: %s\n%s" % (" ".join(cmd), r.stderr))
    return r.stdout


def llama_scores(base_url, pairs, per_pair=False):
    """The score per pair, in fixture order.

    By default one request carries a query's eight documents. With `per_pair` each pair
    is its own request and a pair the model's context cannot hold is stored as NaN, so a
    file with a 512-token limit still records the pairs that fit.
    """
    if per_pair:
        scores = []
        for q, d in pairs:
            body = json.dumps({"query": q, "documents": [d], "top_n": 1}).encode()
            req = urllib.request.Request(
                base_url.rstrip("/") + "/v1/rerank",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    scores.append(json.load(r)["results"][0]["relevance_score"])
            except urllib.error.HTTPError:
                scores.append(float("nan"))
        return scores
    scores = [None] * len(pairs)
    by_query = {}
    for i, (q, d) in enumerate(pairs):
        by_query.setdefault(q, []).append((i, d))
    for q, items in by_query.items():
        body = json.dumps(
            {"query": q, "documents": [d for _, d in items], "top_n": len(items)}
        ).encode()
        req = urllib.request.Request(
            base_url.rstrip("/") + "/v1/rerank",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.load(r)
        for result in data["results"]:
            scores[items[result["index"]][0]] = result["relevance_score"]
    if any(s is None for s in scores):
        sys.exit("llama.cpp returned no score for some pairs")
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="the pair oracle binary")
    ap.add_argument("--base-url", help="a llama-server --reranking")
    ap.add_argument("--model", help="the GGUF file the ids and scores are for")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default="ids.txt", help="the ids file, inside --out")
    ap.add_argument(
        "--per-pair",
        action="store_true",
        help="one request per pair; a pair past the model's context is recorded as NaN",
    )
    ap.add_argument("--scores", default="scores-llama-q8_0.f32")
    ap.add_argument("--queries", default="0:40", help="the query range, START:END")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    lo, hi = (int(x) for x in args.queries.split(":"))
    all_pairs = [p for qi in range(lo, hi) for p in pairs(qi)]

    with open(os.path.join(args.out, "texts.tsv"), "w", encoding="utf-8") as f:
        for q, d in all_pairs:
            f.write(escape(q) + "\t" + escape(d) + "\n")

    if not args.model:
        sys.exit("--model is required for the ids")
    ids = ids_from_oracle(args.pairs, args.model, all_pairs)
    with open(os.path.join(args.out, args.ids), "w", encoding="utf-8") as f:
        for row in ids:
            f.write(json.dumps(row) + "\n")

    if args.base_url:
        scores = llama_scores(args.base_url, all_pairs, args.per_pair)
        with open(os.path.join(args.out, args.scores), "wb") as f:
            for s in scores:
                f.write(struct.pack("<f", s))
        finite = [s for s in scores if s == s]
        print(
            "%d pairs, %d ids; scores %s: %d finite, min %.4f max %.4f"
            % (
                len(all_pairs),
                sum(len(r) for r in ids),
                args.scores,
                len(finite),
                min(finite),
                max(finite),
            )
        )
    else:
        print("%d pairs, %d ids" % (len(all_pairs), sum(len(r) for r in ids)))


if __name__ == "__main__":
    main()
