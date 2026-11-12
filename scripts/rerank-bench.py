#!/usr/bin/env python3
"""Rerank throughput: pairs per second, one query against N documents.

Both engines are asked the same request through their HTTP endpoint, so the number
includes the tokenizer and the response, and one engine runs at a time.

    # a llama-server on the file
    llama-server -m bge-reranker-v2-m3-Q8_0.gguf --reranking -t 4 -c 8192 -ub 2048 \\
        --port 18111 --host 127.0.0.1
    python3 scripts/rerank-bench.py --base-url http://127.0.0.1:18111 \\
        --name llama.cpp --docs 32 --tokens 128

    # then Peregrine, on its own, after stopping the first
    peregrine serve -m bge-reranker-v2-m3-Q8_0.gguf --threads 4 --ctx 8192 --port 18112
    python3 scripts/rerank-bench.py --base-url http://127.0.0.1:18112 \\
        --name peregrine --docs 32 --tokens 128

Prints one line: `name docs tokens pairs/s median of reps (min, max)`.
"""

import argparse
import json
import statistics
import time
import urllib.request

QUERY = "Which of these passages answers the question about the capital of France?"

# A paragraph of ordinary English prose; a document repeats it until it is long enough.
PARAGRAPH = (
    "The committee met on Tuesday to discuss the proposal, and after a long debate it "
    "agreed that the new index would be built over the winter months while the servers "
    "were quiet. Several members asked whether the change could be measured before it "
    "was deployed, and the chair promised a report at the next meeting. "
)


def document(tokens):
    # About four characters per token for English WordPiece; the exact count is read
    # back from the response's usage, so this only has to be close.
    target = tokens * 4
    text = ""
    while len(text) < target:
        text += PARAGRAPH
    return text[:target].rsplit(" ", 1)[0]


def run(base_url, docs, tokens, reps):
    texts = [document(tokens) for _ in range(docs)]
    body = json.dumps({"query": QUERY, "documents": texts, "top_n": docs}).encode()
    times = []
    prompt_tokens = 0
    for _ in range(reps):
        req = urllib.request.Request(
            base_url.rstrip("/") + "/v1/rerank",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=1800) as r:
            data = json.load(r)
        times.append(time.perf_counter() - t0)
        if len(data["results"]) != docs:
            raise SystemExit("the engine returned %d results for %d documents" % (len(data["results"]), docs))
        prompt_tokens = data["usage"]["prompt_tokens"]
    return times, prompt_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--docs", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    times, prompt_tokens = run(args.base_url, args.docs, args.tokens, args.reps)
    med = statistics.median(times)
    print(
        "%-10s docs %3d tokens %4d  %.2f pairs/s  (median of %d: %.2f-%.2f s, %d prompt tokens)"
        % (
            args.name,
            args.docs,
            prompt_tokens // args.docs,
            args.docs / med,
            args.reps,
            min(times),
            max(times),
            prompt_tokens,
        )
    )


if __name__ == "__main__":
    main()
