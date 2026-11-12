#!/usr/bin/env python3
"""The pass width of large float encoders on ARM: PASS_ROWS=32 against 128.

Two `peregrine serve` processes of the same file and thread count, one with
`--opt PASS_ROWS=32` (what a large float encoder carried before #251) and one with
`--opt PASS_ROWS=128` (the engine's default after it); the cells are timed against
both, alternating their order inside every round. The embedding result cache is off
(`--opt EMBED_CACHE_MB=0`) and every timed request sends **fresh** texts — a new
random window of a fixed word pool, all texts of a request distinct — so no cache
and no shared prefix can flatter an arm.

    python3 pass_rows.py --model <file> --alias m --threads 10 --rounds 5 \
        --out pass-rows-<model>-<threads>t.jsonl

Cells: one 16-, 128- and 512-token text; sixteen 32-token texts; thirty-two
128-token texts; eight 512-token texts; and the mixed batch of 32 whose lengths come
from the ARM table's pool (8..512, sum ~14 000 tokens, seed 20261111 per round).

The token count is calibrated per model through `usage.prompt_tokens` and kept
within 5 % of the target.
"""
import argparse, http.client, json, os, random, signal, socket, subprocess, sys, time

WORDS = """
the of and to in a is that it for on as with was at by from this which be or an are not
have has had were their its can will would there all when who been more one about other
into than time only new some could these two may then do first any my now such like our
over man me even most made after also did many before must through back years where much
your way well down should because each just those people mr how too little state good very
make world still own see men work long get here between both life being under never day
same another know while last might us great old year off come since against go came right
used take three states himself few house use during without again place around however home
small found thought went say part once general high upon school every does got united
left number course war until always away something fact water though less public put think
almost hand enough far took head yet government system better set told nothing night end why
called eyes find going look asked later knew point next program business give order
side present case president team within week second four five six seven eight nine ten
""".split()

POOL = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]


def text_of(n_tokens, words_per_token, rng):
    n_words = max(1, int(round(n_tokens * words_per_token)))
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


CONNS = {}


def post(port, alias, texts, timeout=900):
    """One /v1/embeddings request on a kept-alive connection (the 16-token cell is
    ~5 ms of work, so a fresh TCP connection would be part of its time)."""
    body = json.dumps({"model": alias, "input": texts}).encode()
    for attempt in (0, 1):
        c = CONNS.get(port)
        if c is None:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
            CONNS[port] = c
        t0 = time.perf_counter()
        try:
            c.request("POST", "/v1/embeddings", body,
                      {"Content-Type": "application/json"})
            r = c.getresponse()
            payload = r.read()
            secs = time.perf_counter() - t0
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status}: {payload[:200]}")
            out = json.loads(payload)
            return secs, out.get("usage", {}).get("prompt_tokens"), len(out.get("data", []))
        except (http.client.HTTPException, OSError):
            CONNS.pop(port, None)
            if attempt:
                raise


def wait_port(proc, port, seconds=300):
    end = time.time() + seconds
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"server on {port} exited ({proc.returncode})")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"server on {port} did not open")


def lengths_for(name, spec, rnd, rows):
    if name == "mixed":
        mrng = random.Random(20261111 + rnd * 131)
        return [mrng.choice(POOL) for _ in range(32)]
    return [spec[0]] * spec[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--alias", default="m")
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin", default=os.path.expanduser("~/peregrine/target/release/peregrine"))
    ap.add_argument("--base-port", type=int, default=18300)
    ap.add_argument("--cells", default="16,128,512,16x32,32x128,8x512,mixed")
    args = ap.parse_args()

    cpus = "5" if args.threads == 1 else "5-9,15-19"
    servers = {}
    for rows, port in ((32, args.base_port), (128, args.base_port + 1)):
        cmd = ["taskset", "-c", cpus, args.bin, "serve", "-m", args.model,
               "--alias", args.alias, "-t", str(args.threads),
               "--opt", f"PASS_ROWS={rows}", "--opt", "EMBED_CACHE_MB=0",
               "--port", str(port), "--no-profile", "--slots", "1"]
        log = open(f"/tmp/passrows-{rows}-{os.getpid()}.log", "w")
        servers[rows] = (subprocess.Popen(cmd, stdout=log, stderr=log), port)
        wait_port(servers[rows][0], port)

    # Calibrate words per token through the 128-row arm.
    rng = random.Random(7)
    wpt = 1.0
    for _ in range(6):
        _, tok, _ = post(servers[128][1], args.alias, [text_of(128, wpt, rng)])
        if tok and abs(tok - 128) <= 128 * 0.05:
            break
        if tok:
            wpt = wpt * 128.0 / tok
        else:
            wpt *= 1.3
    print(f"calibrated: {wpt:.3f} words/token, 128 -> {tok} tokens", flush=True)

    cells = []
    for c in args.cells.split(","):
        if c == "mixed":
            cells.append(("mixed", None))
        elif "x" in c:
            n, b = c.split("x")
            cells.append((c, (int(n), int(b))))
        else:
            cells.append((c, (int(c), 1)))

    with open(args.out, "w") as out:
        for rnd in range(args.rounds):
            order = [32, 128] if rnd % 2 == 0 else [128, 32]
            for name, spec in cells:
                for rows in order:
                    rng = random.Random(1_000_003 * rnd + 7919 * rows + abs(hash(name)) % 65536)
                    lens = lengths_for(name, spec, rnd, rows)
                    texts = [text_of(L, wpt, rng) for L in lens]
                    secs, tok, n = post(servers[rows][1], args.alias, texts)
                    load = float(open("/proc/loadavg").read().split()[0])
                    rec = dict(cell=name, rows=rows, round=rnd, threads=args.threads,
                               secs=secs, tokens=tok, texts=n, load1=load,
                               tokens_per_s=(tok / secs if tok else 0),
                               texts_per_s=(n / secs if secs else 0))
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    print(f"  r{rnd} {name:8} PASS_ROWS={rows:3} {secs:8.3f}s "
                          f"{tok:6} tok {tok / secs:8.1f} tok/s", flush=True)

    for proc, _ in servers.values():
        proc.send_signal(signal.SIGTERM)
    for proc, _ in servers.values():
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
