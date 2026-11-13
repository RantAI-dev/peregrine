import json, sys, time, urllib.request, urllib.error
port = sys.argv[1]
TOOLS = [{"type":"function","function":{"name":"get_weather","description":"Current weather for a city",
   "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":False}}}]
VERDICT = {"type":"json_schema","json_schema":{"name":"v","schema":{"type":"object","properties":{"answer":{"type":"string","maxLength":40},"confident":{"type":"boolean"}},"required":["answer","confident"],"additionalProperties":False}}}
Q = "Is water wet? Reply with a JSON object with the keys answer (a short string) and confident (a boolean)."
def req(extra):
    body = {"model":"x","messages":[{"role":"user","content":Q}],"max_tokens":60,"temperature":0,"logprobs":True,"top_logprobs":5}
    body.update(extra)
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    try:
        out = json.load(urllib.request.urlopen(r, timeout=120))
        ch = out["choices"][0]; m = ch["message"]
        calls = [(c["function"]["name"], c["function"]["arguments"]) for c in (m.get("tool_calls") or [])]
        lp = (ch.get("logprobs") or {}).get("content") or []
        first = [(t["token"], round(t["logprob"],2)) for t in lp[:3]]
        return f"{ch['finish_reason']:10s} content={m.get('content')!r:.80} calls={calls}\n      first-token top: {first}"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code} {e.read().decode()[:200]}"
for name, extra in [
  ("no tools, no format", {}),
  ("tools only (auto)", {"tools":TOOLS}),
  ("format only", {"response_format":VERDICT}),
  ("tools + format", {"tools":TOOLS,"response_format":VERDICT}),
  ("tools + format, choice none", {"tools":TOOLS,"response_format":VERDICT,"tool_choice":"none"}),
]:
    print(f"{name:28s} {req(extra)}", flush=True)
