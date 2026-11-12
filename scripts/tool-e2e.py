#!/usr/bin/env python3
"""Tool calling end to end against a running server (S6.3).

Sends 16 chat requests with three tools (weather, calculator, search) at temperature 0:
12 where one known call is the right answer and 4 where no call is, then a two-step
exchange that sends a tool's result back. Works against `peregrine serve` and against
`llama-server --jinja`, so the two engines can be compared on the same model file:

    python3 scripts/tool-e2e.py <port> <out.json> [<extra request fields as JSON>]

The optional third argument is merged into every request: `'{"max_tokens": 512}'` for a
model that thinks first, `'{"chat_template_kwargs": {"enable_thinking": false}}'` to switch
thinking off.

The output file holds, per request, the finish reason, the text, the reasoning (when the
server returns one) and the calls (arguments parsed as JSON), and the second step's answer.
"""
import json, sys, urllib.request, urllib.error
port = int(sys.argv[1]); out = sys.argv[2]
extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
weather = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city name"}}, "required": ["city"]}}}
calc = {"type": "function", "function": {"name": "calculate", "description": "Evaluate an arithmetic expression and return the number.", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "For example 12 * (3 + 4)"}}, "required": ["expression"]}}}
search = {"type": "function", "function": {"name": "web_search", "description": "Search the web and return the top results.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
T = [weather, calc, search]
asks = [
 ("What is the weather in Paris right now?", "get_weather"), ("Weather in Tokyo?", "get_weather"), ("Is it raining in London today?", "get_weather"),
 ("How warm is it in Cairo at the moment?", "get_weather"), ("What is 1234 * 5678?", "calculate"), ("Compute (17 + 5) * 3 - 8.", "calculate"),
 ("What is 2 to the power of 20?", "calculate"), ("Divide 987654 by 321.", "calculate"), ("Search the web for the latest Rust release notes.", "web_search"),
 ("Find recent news about the James Webb telescope.", "web_search"), ("Look up who won the 2022 football World Cup.", "web_search"), ("Search for reviews of the Framework 13 laptop.", "web_search"),
 ("Hello! How are you?", None), ("Write one sentence about autumn.", None), ("Thanks, that is all.", None), ("Tell me a short joke.", None),
]
def post(body):
    body = {**body, **extra}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r: return json.loads(r.read())
    except urllib.error.HTTPError as e: return {"http_error": e.code, "body": e.read().decode()[:400]}
res = []
for q, want in asks:
    r = post({"messages": [{"role": "user", "content": q}], "tools": T, "temperature": 0, "max_tokens": 128})
    if "choices" not in r: res.append({"q": q, "want": want, "error": r}); continue
    m = r["choices"][0]["message"]; calls = m.get("tool_calls") or []
    res.append({"q": q, "want": want, "finish": r["choices"][0]["finish_reason"], "content": m.get("content"),
                "reasoning": m.get("reasoning_content"),
                "calls": [{"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])} for c in calls]})
# two-step exchange
first = post({"messages": [{"role": "user", "content": "What is the weather in Paris right now?"}], "tools": T, "temperature": 0, "max_tokens": 128})
step2 = None
if "choices" in first and first["choices"][0]["message"].get("tool_calls"):
    am = first["choices"][0]["message"]
    msgs = [{"role": "user", "content": "What is the weather in Paris right now?"},
            {"role": "assistant", "content": am.get("content"), "tool_calls": am["tool_calls"],
             **({"reasoning_content": am["reasoning_content"]} if am.get("reasoning_content") else {})},
            {"role": "tool", "tool_call_id": am["tool_calls"][0]["id"], "content": "{\"temperature_c\": 18, \"sky\": \"clear\"}"}]
    r2 = post({"messages": msgs, "tools": T, "temperature": 0, "max_tokens": 96})
    step2 = r2["choices"][0]["message"] if "choices" in r2 else r2
json.dump({"requests": res, "two_step": step2}, open(out, "w"), ensure_ascii=False, indent=1)
ok = sum(1 for x in res if "error" not in x and ((x["want"] is None and not x["calls"]) or (x["want"] and x["calls"] and x["calls"][0]["name"] == x["want"])))
print(f"{ok}/{len(res)} as expected; two-step answer: {json.dumps(step2, ensure_ascii=False)[:300] if step2 else None}")
