#!/usr/bin/env python3
"""Generate the encoder-tokenizer goldens from llama.cpp.

The WordPiece and SentencePiece-unigram tokenizers are checked against llama.cpp
v0.9.0 (commit cd959ba) on real model files: the ids in the fixtures must be what
`llama-tokenize` prints, byte for byte, including on input nobody would type twice.

    # a llama.cpp checkout at cd959ba with `build/bin/llama-tokenize` built
    python3 scripts/gen-tokenizer-golden.py \\
        --llama-tokenize /var/tmp/llama/build/bin/llama-tokenize \\
        --oracle /var/tmp/oracle \\
        --model all-MiniLM-L6-v2-Q2_K.gguf \\
        --name minilm \\
        --out crates/peregrine-core/tests/fixtures/tokenize-golden-minilm.tsv

`--oracle` is `scripts/tokenizer-oracle.cpp`, which adds the detokenized text (the
third column) from llama.cpp's own `llama_detokenize`; the ids it prints are
cross-checked against `llama-tokenize` for every case.

The golden columns are: bare ids (llama.cpp `--no-bos`), ids with the file's special
tokens (`--ids`), the escaped input, and llama.cpp's detokenization of the bare ids.
"""

import argparse
import os
import subprocess
import sys
import tempfile

# The fixture strings. `\t`, `\n`, `\r`, `\\` and `\xNN` are the escapes llama.cpp's
# `-p` understands; everything else is UTF-8 as written.
CASES = [
    # The strings the generator tokenizers' fixtures already use, so the three
    # tokenizers can be compared on the same input.
    "Hello world",
    "  leading and trailing  ",
    "a  b   c    d",
    "line one\\nline two\\n\\nline four",
    "Tab\\there\\tand\\ttab",
    "SKU-12345 / batch#7  qty:  48",
    "3.14159265358979 and 1,000,000 and 007",
    "def fibonacci(n):\\n    a, b = 0, 1\\n    for _ in range(n):\\n        a, b = b, a + b\\n    return a",
    "a,b;c... (d) [e] {f} <g> \"h\" 'i' !?@#$%^&*",
    "it's 2026! Don't we've I'll you'd",
    "emoji 🙂 and 🚀 👩\u200d💻",
    "你好，世界！这是一个测试。",
    "吾輩は猫である。名前はまだ無い。",
    "café naïve — “quoted” über Ångström",
    "Привет, мир! Γειά σου Κόσμε",
    "<s>",
    "</s>",
    # Case: both cases, and words the vocabulary only has in one of them.
    "Hello WORLD camelCase PascalCase SCREAMING_SNAKE",
    "The Quick Brown Fox Jumps Over The Lazy Dog",
    # Accents: precomposed, decomposed (combining marks), and a ring above.
    "café naïve résumé coöperate Ångström",
    "e\u0301 vs é and a\u0308 and n\u0303 and o\u0308",
    "Straße ﬁle ẞ İstanbul ılık I I",
    "Tiếng Việt có dấu, Nguyễn Trãi",
    # CJK: kana, kanji, hangul, ideographs outside the BMP.
    "ひらがなとカタカナと漢字が混ざった文です。",
    "안녕하세요 세계! 한국어 테스트입니다.",
    "中文简体、繁體與日本語の漢字",
    "𠮷野家 𠀋 と 𩸽",
    # Emoji: single, ZWJ sequence, flag, skin tone, keycap.
    "family 👨\u200d👩\u200d👧\u200d👦 and flag 🇮🇩 and skin 👋🏽 and keycap 1️⃣",
    # Punctuation runs, symbols, and markdown-ish text.
    "!!!???...---===___+++///\\\\\\",
    "©®™ ± § ¶ • … € £ ¥ ° µ ×÷≈≠",
    "**bold** _italic_ `code` #heading - list 1. item",
    # Whitespace: tabs, newlines, non-breaking and full-width spaces, only-space.
    "a\\tb\\r\\nc\\n\\nd",
    " \\t \\n ",
    "a\u00a0b\u2009c\u3000d u\u0308",
    "Ｆｕｌｌ　ｗｉｄｔｈ ｔｅｘｔ",
    # Control characters, including NUL and DEL: WordPiece drops them, unigram
    # maps NUL through the charsmap.
    "a\\x01b\\x02c\\x7fd",
    "ctrl:\\x00 eot",
    # Words longer than any token, runs of one character, and a sentence with no
    # spaces at all.
    "supercalifragilisticexpialidocious pneumonoultramicroscopicsilicovolcanoconiosis "
    "antidisestablishmentarianism floccinaucinihilipilification",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "thequickbrownfoxjumpsoverthelazydogandkeptrunninguntiltheendofthelinewithoutaspace",
    # Literal special-token text: llama.cpp pulls these out of the raw input before
    # tokenizing, and the reranker path does not.
    "[CLS] hello [SEP] [MASK] [PAD] [UNK] <s> </s> <unk>",
    "x[CLS]y[SEP]z[MASK][MASK]",
    "[MASK]",
    # URLs, code, identifiers.
    "https://example.com/a/b?c=d&e=f mailto:user@example.com",
    "snake_case.kebab-case/v1.2.3 struct Foo { x: i32 }",
    # Languages that stress whitespace and combining behaviour.
    "नमस्ते दुनिया สวัสดีชาวโลก",
    "مرحبا بالعالم שלום עולם",
    # A paragraph of prose, and a CJK one, at the length an embedding input has.
    "Retrieval-augmented generation answers a question by first finding passages that "
    "mention the entities in it. The retriever embeds the question and every passage, "
    "ranks passages by cosine similarity, and hands the top few to the generator. Most "
    "of the quality comes from the retriever: a passage that never reaches the prompt "
    "cannot be used, and a passage that reaches it out of order is a distraction. "
    "Rerankers exist because a bi-encoder compresses a passage into one vector, which "
    "loses the fine distinctions a cross-encoder can see: a reranker reads the query "
    "and the passage together and scores their interaction. On a CPU box the cost of "
    "that second pass is what decides whether reranking is affordable, and the cost is "
    "dominated by the tokenizer and the matrix multiplies, in that order for short "
    "texts on a small model.",
    "检索增强生成通过先找到提到问题中实体的段落来回答问题。检索器把问题和每个段落编码成向量，"
    "按余弦相似度排序，再取前几个交给生成器。质量主要来自检索器：没有进入提示的段落无法被使用，"
    "顺序错误的段落则是干扰。重排序器之所以存在，是因为双塔编码器把段落压缩成一个向量，"
    "丢掉了交叉编码器能看到的细微差别。重排序器同时读取问题和段落，并为它们的交互打分。"
    "在 CPU 上，第二遍计算的成本决定了重排序是否负担得起。",
    # An empty input, and one that is only control characters.
    "",
    "\\x00\\x01\\x02",
]


# (query, document) pairs for the rerank path. The pair is built by llama.cpp's
# `format_prompt_rerank`, which is what `scripts/tokenizer-pair-oracle.cpp` calls.
PAIRS = [
    ("query one", "document one"),
    ("the capital of Indonesia?", "Jakarta is the capital of Indonesia."),
    ("[CLS] literal specials [SEP]", "doc with [MASK] and <s> and </s>"),
    ("café naïve", "Ångström über Straße"),
    ("中文查询", "日本語の文書と한국어"),
    ("it's 2026!", "Don't we've I'll you'd"),
    ("", "only a document"),
    ("only a query", ""),
    ("what changed in v1.2.3?", "line one\\nline two\\t\\n  indented\\n"),
    ("emoji 🙂", "👨\u200d👩\u200d👧 and 🇮🇩"),
]


def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, check=False, **kwargs)


def llama_tokenize_ids(binary, model, text, no_bos):
    cmd = [binary, "-m", model, "-p", text, "--ids"]
    if no_bos:
        cmd.append("--no-bos")
    r = run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"llama-tokenize failed on {text!r}: {r.stderr.strip()}")
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-tokenize", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--provenance", required=True,
                    help="repo and file name, for the fixture header")
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--skip-cross-check", action="store_true")
    ap.add_argument("--pair-oracle", help="writes <out>-pair.tsv with the rerank pairs")
    args = ap.parse_args()

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for case in CASES:
            f.write(case.replace("\n", "") + "\n")
        case_file = f.name
    try:
        r = run([args.oracle, args.model, case_file])
        if r.returncode != 0:
            raise SystemExit(f"oracle failed: {r.stderr.strip()}")
        rows = [line.split("\t") for line in r.stdout.splitlines()]
    finally:
        os.unlink(case_file)

    if len(rows) != len(CASES):
        raise SystemExit(f"oracle returned {len(rows)} rows for {len(CASES)} cases")

    if not args.skip_cross_check:
        for case, row in zip(CASES, rows):
            if not case:
                continue  # llama-tokenize refuses an empty -p; the oracle covers it
            bare = llama_tokenize_ids(args.llama_tokenize, args.model, case, True)
            special = llama_tokenize_ids(args.llama_tokenize, args.model, case, False)
            if bare != row[0] or special != row[1]:
                raise SystemExit(
                    f"llama-tokenize and the oracle disagree on {case!r}:\n"
                    f"  cli      {bare}\n  oracle   {row[0]}\n"
                    f"  cli+s    {special}\n  oracle+s {row[1]}"
                )

    header = [
        f"# {args.provenance}",
        f"# sha256 {args.sha256}",
        "# goldens from llama.cpp v0.9.0 (commit cd959bae99ff4bed22bef2604f45241c09bbb8c8) -- llama-tokenize --ids and",
        "# --ids --no-bos for the ids, llama_detokenize for the last column; see",
        "# scripts/gen-tokenizer-golden.py",
        "# columns: bare ids <TAB> ids with the file's special tokens <TAB> escaped text"
        " <TAB> detokenized text",
    ]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        for row in rows:
            f.write("\t".join(row) + "\n")
    print(f"{args.name}: {len(rows)} cases -> {args.out}", file=sys.stderr)

    if args.pair_oracle:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            for query, doc in PAIRS:
                f.write(f"{query}\t{doc}\n")
            pair_file = f.name
        try:
            r = run([args.pair_oracle, args.model, pair_file])
            if r.returncode != 0:
                raise SystemExit(f"pair oracle failed: {r.stderr.strip()}")
            pair_rows = r.stdout.splitlines()
        finally:
            os.unlink(pair_file)
        if len(pair_rows) != len(PAIRS):
            raise SystemExit(f"pair oracle returned {len(pair_rows)} rows")
        pair_out = args.out.replace(".tsv", "-pair.tsv")
        with open(pair_out, "w", encoding="utf-8") as f:
            f.write("\n".join(header[:4]) + "\n")
            f.write("# columns: query <TAB> document <TAB> ids, as");
            f.write(" llama.cpp's format_prompt_rerank builds them\n")
            for (query, doc), ids in zip(PAIRS, pair_rows):
                f.write(f"{query}\t{doc}\t{ids}\n")
        print(f"{args.name}: {len(PAIRS)} pairs -> {pair_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
