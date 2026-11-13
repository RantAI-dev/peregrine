import json, sys, time, urllib.request, urllib.error, statistics
port, label = sys.argv[1], sys.argv[2]
TOOLS = [
 {"type":"function","function":{"name":"get_weather","description":"Current weather for a city",
   "parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city","unit"],"additionalProperties":False}}},
 {"type":"function","function":{"name":"get_time","description":"Current time in a timezone",
   "parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"],"additionalProperties":False}}},
 {"type":"function","function":{"name":"book_table","description":"Book a restaurant table",
   "parameters":{"type":"object","properties":{"restaurant":{"type":"string"},"people":{"type":"integer"},"date":{"type":"string"},"outdoor":{"type":"boolean"}},"required":["restaurant","people","date"],"additionalProperties":False}}},
]
QS = ["What is the weather in Paris in celsius?", "What time is it in Asia/Tokyo?", "Book a table for four at Chez Louis on 2026-12-01, outdoors.",
      "Tell me a joke.", "Weather in Oslo, Lima and Cairo in fahrenheit please.", "Is it late in America/New_York?",
      "Reserve Sushi Go for 2 people tomorrow.", "Hello!", "How hot is Dubai right now?", "Table for 6 at Nonna's, 2026-11-20."]
def check(name, args):
    t = next((t["function"] for t in TOOLS if t["function"]["name"] == name), None)
    if t is None: return False
    try: a = json.loads(args)
    except Exception: return False
    if not isinstance(a, dict): return False
    p = t["parameters"]
    if any(k not in a for k in p["required"]): return False
    for k, v in a.items():
        s = p["properties"].get(k)
        if s is None: return False
        ty = s["type"]
        ok = {"string": isinstance(v, str), "integer": isinstance(v, int) and not isinstance(v, bool), "boolean": isinstance(v, bool)}[ty]
        if not ok or ("enum" in s and v not in s["enum"]): return False
    return True
valid = total = replies_with_call = 0; rates = []
for choice in ["required", {"type":"function","function":{"name":"book_table"}}]:
    for q in QS:
        body = {"model":"x","messages":[{"role":"user","content":q}],"max_tokens":160,"temperature":0,"tools":TOOLS,"tool_choice":choice}
        r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
        try:
            out = json.load(urllib.request.urlopen(r, timeout=180))
        except urllib.error.HTTPError as e:
            print("HTTP", e.code, e.read().decode()[:200]); continue
        calls = out["choices"][0]["message"].get("tool_calls") or []
        replies_with_call += bool(calls)
        if not calls: print('   no call:', 'named' if isinstance(choice, dict) else choice, '|', q, '|', out['choices'][0]['finish_reason'], repr(out['choices'][0]['message'].get('content'))[:70])
        for c in calls:
            total += 1; valid += check(c["function"]["name"], c["function"]["arguments"])
        tps = (out.get("timings") or {}).get("predicted_per_second")
        if tps and out["usage"]["completion_tokens"] >= 8: rates.append(tps)
print(f"{label}: replies with a call {replies_with_call}/20, calls valid for their schema {valid}/{total}, decode tok/s median {statistics.median(rates):.1f} (min {min(rates):.1f}, max {max(rates):.1f})")
