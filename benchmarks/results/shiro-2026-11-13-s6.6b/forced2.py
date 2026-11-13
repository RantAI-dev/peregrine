import json, sys, time, urllib.request, urllib.error
port = sys.argv[1]
TOOLS = [
 {"type":"function","function":{"name":"get_weather","description":"Current weather for a city",
   "parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]},"days":{"type":"integer"}},"required":["city","unit"],"additionalProperties":False}}},
 {"type":"function","function":{"name":"get_time","description":"Current time in a timezone",
   "parameters":{"type":"object","properties":{"timezone":{"type":"string"},"utc":{"type":"boolean"}},"required":["timezone"],"additionalProperties":False}}},
]
OFF = {"chat_template_kwargs":{"enable_thinking":False}}
VERDICT = {"type":"json_schema","json_schema":{"name":"v","schema":{"type":"object","properties":{"answer":{"type":"string","maxLength":40},"confident":{"type":"boolean"}},"required":["answer","confident"],"additionalProperties":False}}}
def req(extra, user, max_tokens):
    body = {"model":"x","messages":[{"role":"user","content":user}],"max_tokens":max_tokens,"temperature":0,"tools":TOOLS}
    body.update(extra)
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    t = time.perf_counter()
    try:
        out = json.load(urllib.request.urlopen(r, timeout=280))
        ch = out["choices"][0]; m = ch["message"]
        calls = [(c["function"]["name"], c["function"]["arguments"]) for c in (m.get("tool_calls") or [])]
        think = m.get("reasoning_content") or ""
        return f"{time.perf_counter()-t:5.1f}s {ch['finish_reason']:10s} think={len(think):4d}ch content={m.get('content')!r:.70} calls={calls} tok={out['usage']['completion_tokens']}"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code} {e.read().decode()[:200]}"
named = lambda n: {"type":"function","function":{"name":n}}
Q = "What is the weather in Paris in celsius, and what time is it in Europe/Paris?"
cases = [
 ("required, thinking off", {"tool_choice":"required", **OFF}, Q, 160),
 ("named get_time, off", {"tool_choice":named("get_time"), **OFF}, "Is it cold in Oslo?", 120),
 ("required, thinking ON", {"tool_choice":"required"}, Q, 700),
 ("auto+response_format, off, chat", {"response_format":VERDICT, **OFF}, "Is water wet? Reply with a JSON object with the keys answer (a short string) and confident (a boolean).", 120),
 ("auto+response_format, off, tool q", {"response_format":VERDICT, **OFF}, Q, 160),
]
for name, extra, user, mt in cases:
    print(f"{name:34s} {req(extra, user, mt)}", flush=True)
