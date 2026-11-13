import json, sys, time, urllib.request
port = sys.argv[1]
TOOLS = [{"type":"function","function":{"name":"get_weather","description":"Current weather for a city",
   "parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city","unit"],"additionalProperties":False}}}]
def run(extra, stream, mt=500):
    body = {"model":"x","messages":[{"role":"user","content":"What is the weather in Paris in celsius?"}],"max_tokens":mt,"temperature":0,"tools":TOOLS,"stream":stream}
    body.update(extra)
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    resp = urllib.request.urlopen(r, timeout=280)
    if not stream:
        out = json.load(resp); ch = out["choices"][0]; m = ch["message"]
        ph = out["timings"]["peregrine"]["phases_ms"]
        d = dict(zip(ph["names"], ph["decode"])); n = out["usage"]["completion_tokens"]
        return {"finish": ch["finish_reason"], "think": len(m.get("reasoning_content") or ""), "content": m.get("content"),
                "calls": [(c["function"]["name"], c["function"]["arguments"]) for c in (m.get("tool_calls") or [])],
                "tokens": n, "sampling_ms_per_token": round(d["sampling"]/max(n,1), 3), "decode_ms_per_token": round(sum(d.values())/max(n,1), 2)}
    think = content = ""; calls = {}; finish = None; chunks = 0
    for line in resp:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]": continue
        ev = json.loads(line[6:]); chunks += 1
        for ch in ev.get("choices", []):
            d = ch.get("delta", {})
            think += d.get("reasoning_content") or ""; content += d.get("content") or ""
            for tc in d.get("tool_calls") or []:
                c = calls.setdefault(tc["index"], {"name": "", "arguments": ""})
                c["name"] += (tc.get("function") or {}).get("name") or ""; c["arguments"] += (tc.get("function") or {}).get("arguments") or ""
            finish = ch.get("finish_reason") or finish
    return {"finish": finish, "think": len(think), "content": content or None, "calls": [(c["name"], c["arguments"]) for _, c in sorted(calls.items())], "chunks": chunks}
for name, extra in [("auto (unconstrained)", {}), ("required", {"tool_choice": "required"})]:
    a = run(extra, False); b = run(extra, True)
    same = (a["finish"], a["think"], a["content"], a["calls"]) == (b["finish"], b["think"], b["content"], b["calls"])
    print(f"{name}: {a}\n   streamed the same: {same} ({b['chunks']} chunks)" + ("" if same else f"\n   STREAM: {b}"), flush=True)
