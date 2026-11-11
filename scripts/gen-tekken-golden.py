#!/usr/bin/env python3
"""Generate the Tekken golden fixture from llama.cpp.

The Tekken pre-tokenizer (`tokenizer.ggml.pre = "pixtral"`, which llama.cpp maps to
`LLAMA_VOCAB_PRE_TYPE_TEKKEN`) is checked against llama.cpp v0.9.0 (commit cd959ba)
on NVIDIA's Nemotron 3.5 Embed 1B vocabulary. The ids in the fixture must be what
llama.cpp produced, including on the inputs where its `std::regex` backtracks.

    # a llama.cpp checkout at cd959ba with build/bin/llama-tokenize built
    g++ -std=c++17 -O2 -I$LLAMA/include -I$LLAMA/common -I$LLAMA/ggml/include \\
        -L$LLAMA/build/bin -Wl,-rpath,$LLAMA/build/bin scripts/tokenizer-oracle.cpp \\
        -o /tmp/tekken-oracle -lllama -lggml
    python3 scripts/gen-tekken-golden.py \\
        --oracle /tmp/tekken-oracle \\
        --llama-tokenize $LLAMA/build/bin/llama-tokenize \\
        --model Nemotron-3.5-Embed-1B-Q4_K_M.gguf \\
        --out crates/peregrine-core/tests/fixtures/tokenize-golden-tekken.tsv \\
        --provenance "nvidia/Nemotron-3.5-Embed-1B GGUF / Nemotron-3.5-Embed-1B-Q4_K_M.gguf" \\
        --sha256 HASH

Columns, TAB separated: llama.cpp's ids without special tokens (`add_special=false`),
with them (`add_special=true`), the escaped input, llama.cpp's detokenization of the
bare ids, and the ids with special tokens but `parse_special=false`. The last column is
what `scripts/tokenizer-oracle.cpp` now prints for every vocabulary; the five encoder
fixtures keep their four-column files and ignore it.

The model file is 750 MB and not committed, so `tests/tokenize_tekken.rs` builds the
tokenizer from `~/models/` (or `models/`) and skips when it is absent. The splitter's
own tests run in CI without any file.
"""

import argparse
import os
import subprocess
import sys
import tempfile

# The fixture strings. `\t`, `\n`, `\r`, `\\` and `\xNN` are the escapes llama.cpp's
# `-p` understands; everything else is UTF-8 as written.
BASE = [
    # The strings the other tokenizer fixtures already use, so the vocabularies can be
    # compared on the same input.
    "Hello world",
    "  leading and trailing  ",
    "a  b   c    d",
    "line one\nline two\n\nline four",
    "Tab\there\tand\ttab",
    "SKU-12345 / batch#7  qty:  48",
    "3.14159265358979 and 1,000,000 and 007",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "a,b;c... (d) [e] {f} <g> \"h\" 'i' !?@#$%^&*",
    "it's 2026! Don't we've I'll you'd",
    "emoji 🙂 and 🚀 👩\u200d💻",
    "你好，世界！这是一个测试。",
    "吾輩は猫である。名前はまだ無い。",
    "café naïve — “quoted” über Ångström",
    "Привет, мир! Γειά σου Κόσμε",
    "<s>",
    "</s>",
    # Case: every shape the two word alternatives distinguish.
    "HTMLParser",
    "HTTPServer",
    "JSONData",
    "PDFReader",
    "XMLHttpRequest",
    "ABc",
    "ABCDef",
    "iOS",
    "macOS",
    "iPhone",
    "iPad",
    "eBay",
    "Hello WORLD camelCase PascalCase SCREAMING_SNAKE",
    "The Quick Brown Fox Jumps Over The Lazy Dog",
    # A leading non-letter attaches to the word: punctuation, space, digit, mark.
    " hello",
    ' "hello',
    "(hello",
    "-hello",
    "_hello",
    ".hello",
    "'hello",
    "/hello",
    "\thello",
    "\u200bhello",
    "日hello",
    "🙂hello",
    "Ⅻhello",
    # Single digits, digit runs, and the separator shapes around them.
    "0",
    "1",
    "42",
    "1234567890",
    "12345",
    "1 22 333 4444 55555",
    "3.14159",
    "1,000,000",
    "2026",
    "007",
    "0xdeadBEEF",
    "v1.2.3",
    "abc123def",
    "A1B2C3",
    "١٢٣٤٥",
    "１２３４５",
    "ⅫⅧ",
    "²³",
    # Punctuation runs, and what they take with them.
    "a.../b",
    "x!?\n\ny",
    "a//b",
    "///",
    "!!\n//!?",
    "a/\n/b",
    "**bold** _italic_ `code` #heading - list 1. item",
    "!!!???...---===___+++///\\\\\\",
    "©®™ ± § ¶ • … € £ ¥ ° µ ×÷≈≠",
    "→← ↑↓ ⇒ ⇔",
    "┌─┬─┐\n│ │ │\n└─┴─┘",
    # Newline runs, CRLF, and whitespace around them.
    "a\n\nb",
    "a\n\n\nb",
    "\n",
    "\n\n\n",
    "a\r\nb",
    "a\r\rb",
    " \n ",
    "a \n \n b",
    "a\t\n\tb",
    # Trailing whitespace, and whitespace before non-whitespace.
    "abc ",
    "abc  ",
    "abc\t",
    "abc \t\n",
    "a b",
    "a \tb",
    "a\t b",
    "a\u00a0b",
    "a\u2009b",
    "a\u3000b",
    "a\u2028b",
    "a\u2029b",
    "a\u0085b",
    "a\u000bb",
    "a\u000cb",
    # The zero-width-non-joiner rule (nemotron-3.5-embed only).
    "[\u200cab",
    "<\u200ctag>",
    "x[\u200cy",
    "[\u200c",
    "<\u200c",
    "a<\u200c>",
    "[[\u200cab",
    "[\u200c\u200cab",
    "<\u200c<\u200cb",
    "[\u200dzwj]",
    "[zwj]\u200c",
    "a[\u200c]b",
    "[\u200c]",
    "x<\u200c/INST>",
    "[\u200c[\u200cINST]",
    "][\u200c]",
    "[a\u200c",
    "<a\u200c",
    "a\u200c",
    "\u200c[",
    # Accented Latin: precomposed and decomposed.
    "café naïve résumé coöperate Ångström",
    "e\u0301 vs é and a\u0308 and n\u0303 and o\u0308",
    "Straße ﬁle ẞ İstanbul ılık I I",
    "Tiếng Việt có dấu, Nguyễn Trãi",
    "Œuf œuvre Æsir Øresund Þingvellir Ð",
    "Zürich Genève Malmö Åbo",
    # Cyrillic and Greek: non-ASCII letters never split on case.
    "Привет, мир!",
    "ПРИВЕТ мир",
    "Съешь же ещё этих мягких французских булок",
    "Γειά σου Κόσμε",
    "ΑΘΗΝΑ Αθήνα",
    "αβγδε ζηθικ λμνξο",
    # CJK: kana, kanji, hangul, outside the BMP, and mixed with ASCII.
    "ひらがなとカタカナと漢字が混ざった文です。",
    "안녕하세요 세계! 한국어 테스트입니다.",
    "中文简体、繁體與日本語の漢字",
    "𠮷野家 𠀋 と 𩸽",
    "東京2026オリンピック",
    "Ｆｕｌｌ　ｗｉｄｔｈ ｔｅｘｔ",
    # Arabic, Hebrew, and Indic scripts with combining marks.
    "مرحبا بالعالم",
    "العربية لغة سامية",
    "שלום עולם",
    "नमस्ते दुनिया",
    "हिन्दी में मात्राएँ",
    "क्ष त्र ज्ञ श्र",
    "ि",
    "x\u093fय",
    "বাংলা ভাষা",
    "தமிழ் மொழி",
    "తెలుగు భాష",
    "ಕನ್ನಡ ಭಾಷೆ",
    "മലയാളം ഭാഷ",
    "සිංහල භාෂාව",
    "ไทย ภาษาไทย",
    "ພາສາລາວ",
    "မြန်မာဘာသာ",
    "ភាសាខ្មែរ",
    "ქართული ენა",
    "Հայերեն լեզու",
    "አማርኛ ቋንቋ",
    # Emoji: single, ZWJ sequence, flags, skin tones, keycaps.
    "🙂",
    "🚀",
    "👩\u200d💻",
    "👨\u200d👩\u200d👧\u200d👦",
    "🇮🇩 🇯🇵",
    "👋🏽",
    "1️⃣ 2️⃣",
    "🙂🙂🙂",
    "a🙂b",
    "🙂 a",
    # Contractions, and apostrophes that are not.
    "it's",
    "IT'S",
    "I'LL",
    "don't",
    "we've",
    "you'd",
    "o'clock",
    "'",
    "''",
    "'s",
    "rock 'n' roll",
    "l'heure d'été",
    # Long words, and runs of one character.
    "supercalifragilisticexpialidocious pneumonoultramicroscopicsilicovolcanoconiosis "
    "antidisestablishmentarianism floccinaucinihilipilification",
    "a" * 120,
    "A" * 120,
    "1" * 120,
    " " * 40 + "x" + " " * 40,
    "thequickbrownfoxjumpsoverthelazydogandkeptrunninguntiltheendofthelinewithoutaspace",
    # The prefixes this embedding model is used with.
    "query: the capital of Indonesia?",
    "passage: Jakarta is the capital of Indonesia.",
    "query: ",
    "passage: ",
    "query:passage:",
    "Query: HTMLParser",
    "QUERY: 12345",
    # Literal control-token spellings: tokens with `parse_special`, text without it.
    "[INST] hello [/INST]",
    "x[INST]y",
    "<unk>",
    "<pad>",
    "[TOOL_CALLS] x",
    "[/AVAILABLE_TOOLS]",
    "[INST][INST][INST]",
    "a<s>b</s>c<unk>d",
    "[IMG]",
    "[/INST]",
    # URLs, code, identifiers.
    "https://example.com/a/b?c=d&e=f mailto:user@example.com",
    "snake_case.kebab-case/v1.2.3 struct Foo { x: i32 }",
    "fn main() { println!(\"{}\", 1 + 2); }",
    "SELECT * FROM t WHERE a = 'x' -- comment\n",
    # Whitespace-only and empty inputs.
    "",
    " ",
    "   ",
    "\t\n\r ",
    "\u00a0",
    "\u3000",
    "\n\n\n\n\n",
    "\t\t\t",
    # A paragraph of prose, and a CJK one, at the length an embedding input has.
    "Retrieval-augmented generation answers a question by first finding passages that "
    "mention the entities in it. The retriever embeds the question and every passage, "
    "ranks passages by cosine similarity, and hands the top few to the generator. Most "
    "of the quality comes from the retriever: a passage that never reaches the prompt "
    "cannot be used, and a passage that reaches it out of order is a distraction.",
    "检索增强生成通过先找到提到问题中实体的段落来回答问题。检索器把问题和每个段落编码成向量，"
    "按余弦相似度排序，再取前几个交给生成器。质量主要来自检索器：没有进入提示的段落无法被使用，"
    "顺序错误的段落则是干扰。",
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
    # The same alternatives at different offsets: what precedes a run decides whether
    # it attaches, and the first alternative's `U*` backtracks from its greedy end.
    " hello",
    "  hello",
    "\thello",
    "\nhello",
    "xhello",
    "Xhello",
    "1hello",
    ".hello",
    "🙂hello",
    "日hello",
    "hello ",
    "hello  ",
    "hello\n",
    "hello/",
    "hello\r\n",
    "hello.",
    "hello1",
    "hellox",
    "hello🙂",
    "HTMLParser ",
    " HTMLParser",
    "xHTMLParser",
    "HTMLParser1",
    "HTMLParser.",
    "iPhone1",
    " iPhone",
    "xiPhone",
    "iPhone/",
    "1234567 ",
    " 1234567",
    "x1234567",
    "1234567.",
    "1234567/",
    "abc123 ",
    " abc123",
    "abc123x",
    "a/b ",
    "a/b\n",
    " a/b",
    "!? ",
    "!?\n",
    " !?",
    "!?/",
    # Multi-line and mixed-script sentences.
    "İstanbul'da 2026 yılının ilk günü.",
    "Γειά σου, Κόσμε! 123",
    "Привет 2026 мир",
    "مرحبا 2026 بالعالم",
    "नमस्ते 2026 दुनिया",
    "ไทย 2026 ภาษาไทย",
    "日本語2026テスト",
    "한국어 2026 테스트",
    "🙂2026🚀",
    "query: café\npassage: naïve",
    "query: 123\npassage: 456",
    "query: [INST]\npassage: </s>",
    "a\u0301\u0302\u0303b",
    "\u0301",
    "\u0301abc",
    "abc\u0301",
    "a\u200cb[\u200c]c",
    "<\u200c<\u200c<",
    "[INST]\u200c [INST]",
    "   [INST]   ",
    "\n[INST]\n",
    "x\n\ty\n\rz",
    # ASCII case boundaries, one shape at a time.
    "a",
    "A",
    "ab",
    "AB",
    "Ab",
    "aB",
    "aBc",
    "AbC",
    "aBcD",
    "ABcd",
    "abcDEF",
    "fooBAR",
    "FooBarBaz",
    "getHTTPResponseCode",
    "parseJSONToXML",
    "HTML5Parser",
    "utf8String",
    "SHA256Hash",
    "IPv6Address",
    "OAuth2Token",
    "iOS17",
    "macOS14",
    "aaaaAAAAaaaa",
    "1111aaaa1111",
    # Digits in runs of every length, and the formats around them.
    "1234",
    "12345678901234567890",
    "1,2,3,4",
    "3.14/2",
    "0.0001",
    "1e10",
    "-42",
    "+7",
    "(555) 123-4567",
    "2026-11-11T10:00:00Z",
    "550e8400-e29b-41d4-a716-446655440000",
    "deadbeefDEADBEEF0123456789abcdef",
    "SGVsbG8gV29ybGQh",
    # Punctuation that is a path, and punctuation before a newline.
    "and/or",
    "either//or",
    "a/b/c/d",
    "path/to/file.txt",
    "../..",
    "./src/main.rs",
    "a\n/b\n/c",
    "a   b\n  c\t\td",
    "\n \n \n",
    " \t \t ",
    "x\u00a0\u00a0y",
    # More scripts, with the marks the Unicode tables distinguish.
    "हिंदी",
    "कर्म",
    "ज्ञान",
    "مُحَمَّد",
    "שָׁלוֹם",
    "น้ำ",
    "Đường",
    "ㅎㅏㄴ",
    "ラーメン",
    "「」『』、。",
    "∑∫√∞≤≥",
    "a≤b≥c",
    "！？。，",
    "ａｂｃ１２３",
    "ⅠⅡⅢⅣ",
    "☺️",
    "❤️",
    "🏳️\u200d🌈",
    "🧑🏽\u200d🚀",
    "⚡",
    "☀",
    "🤦🏼\u200d♀️",
    # Control-token spellings this vocabulary carries as control tokens.
    "[SYSTEM_PROMPT]",
    "[/SYSTEM_PROMPT]",
    "[ARGS]",
    "[CALL_ID]",
    "[BEGIN_AUDIO]",
    "[PREFIX]",
    "[MIDDLE]",
    "[SUFFIX]",
    "[IMG_BREAK]",
    "[TOOL_CONTENT]",
    "a[PREFIX]b[SUFFIX]c",
    # Structured text at embedding-input length.
    '{"query": "café", "ids": [1, 22, 333], "ok": true}\n',
    "- name: Nemotron\n  dims: 2048\n  langs: [en, de, fr]\n",
    "| a | b |\n|---|---|\n| 1 | 2 |\n",
    "query: Which passage mentions HTMLParser and iPhone?\n"
    "passage: The passage mentions HTMLParser, iPhone, and 1234567.",
]


def escape(s: str) -> str:
    """The escapes llama.cpp's `-p` and the fixture's Rust reader understand."""
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append("\\x%02X" % ord(ch))
        else:
            out.append(ch)
    return "".join(out)


def run(cmd):
    """Run a command, decoding its output leniently: llama.cpp's load log prints
    token bytes that are not valid UTF-8."""
    r = subprocess.run(cmd, capture_output=True, check=False)
    return (
        r.returncode,
        r.stdout.decode("utf-8", "replace"),
        r.stderr.decode("utf-8", "replace"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True, help="scripts/tokenizer-oracle.cpp, built")
    ap.add_argument("--llama-tokenize", help="cross-check this many cases against the CLI")
    ap.add_argument("--cross-check", type=int, default=0)
    ap.add_argument(
        "--pre",
        help="metadata override for tokenizer.ggml.pre (e.g. nemotron-3.5-embed); "
        "keeps only the strings containing U+200C, where that pre-type's normalization "
        "can change the ids",
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--provenance", required=True)
    ap.add_argument("--sha256", required=True)
    args = ap.parse_args()

    cases = []
    seen = set()
    for case in BASE:
        if case not in seen:
            seen.add(case)
            cases.append(case)
    if not cases[-1]:
        # The oracle reads one escaped case per line and cannot see a trailing empty
        # one; the empty input belongs in the middle.
        raise SystemExit("the last fixture string must not be empty")
    if args.pre:
        # Only a joiner directly after `[` or `<` can be rewritten, so only those
        # strings can tokenize differently under the override; the rest of the corpus
        # is the first fixture.
        cases = [c for c in cases if "\u200c" in c]
    if not cases:
        raise SystemExit("no cases left")
    if args.pre and args.cross_check:
        raise SystemExit("--cross-check cannot verify --pre: the CLI drops --override-kv")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(escape(c) for c in cases) + "\n")
        case_file = f.name
    try:
        cmd = [args.oracle, args.model, case_file]
        if args.pre:
            cmd.append(f"tokenizer.ggml.pre={args.pre}")
        code, out, err = run(cmd)
        if code != 0:
            raise SystemExit(f"oracle failed: {err.strip()[-2000:]}")
        rows = [line.split("\t") for line in out.split("\n") if line]
    finally:
        os.unlink(case_file)

    if len(rows) != len(cases):
        raise SystemExit(f"oracle returned {len(rows)} rows for {len(cases)} cases")

    if args.llama_tokenize and args.cross_check:
        stride = max(1, len(cases) // args.cross_check)
        for i in range(0, len(cases), stride):
            case = cases[i]
            if not case:
                continue
            checks = [
                (["--ids", "--no-bos"], rows[i][0]),
                (["--ids"], rows[i][1]),
                (["--ids", "--no-parse-special"], rows[i][4]),
            ]
            for flags, want in checks:
                code, out, err = run(
                    [args.llama_tokenize, "-m", args.model, "-p", escape(case), *flags])
                if code != 0:
                    raise SystemExit(f"llama-tokenize failed on {case!r}: {err.strip()}")
                if out.strip() != want.strip():
                    raise SystemExit(
                        f"llama-tokenize and the oracle disagree on {case!r} {flags}:\n"
                        f"  cli    {out.strip()}\n  oracle {want.strip()}"
                    )

    header = [
        f"# {args.provenance}",
        f"# sha256 {args.sha256}",
        "# goldens from llama.cpp v0.9.0 (commit cd959bae99ff4bed22bef2604f45241c09bbb8c8)",
        "# llama_tokenize(add_special=false), llama_tokenize(add_special=true),",
        "# llama_tokenize(true, parse_special=false), and llama_detokenize; see",
        "# scripts/gen-tekken-golden.py",
        "# columns: bare ids <TAB> ids with special tokens <TAB> escaped text"
        " <TAB> detokenized text <TAB> ids without special parsing",
    ]
    if args.pre:
        header.insert(
            2,
            f"# tokenizer.ggml.pre overridden to \"{args.pre}\" (only the strings"
            " containing U+200C can differ from the file's own pre-tokenizer)",
        )
    lines = header + ["\t".join([row[0], row[1], row[2], row[3], row[4]]) for row in rows]
    body = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    print(f"wrote {args.out}: {len(cases)} cases, {len(body)} bytes")
    if len(body) > 2 * 1024 * 1024:
        print("WARNING: fixture is larger than 2 MB", file=sys.stderr)


if __name__ == "__main__":
    main()
