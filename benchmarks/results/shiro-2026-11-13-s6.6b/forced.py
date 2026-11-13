import json, sys, time, urllib.request, urllib.error
port = sys.argv[1]
TOOLS = [
 {"type":"function","function":{"name":"get_weather","description":"Current weather for a city",
   "parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"],"additionalProperties":False}}},
 {"type":"function","function":{"name":"get_time","description":"Current time in a timezone",
   "parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"],"additionalProperties":False}}},
]
def req(extra, user):
    body = {"model":"x","messages":[{"role":"user","content":user}],"max_tokens":96,"temperature":0,"tools":TOOLS}
    body.update(extra)
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    t = time.perf_counter()
    try:
        out = json.load(urllib.request.urlopen(r, timeout=120))
        ch = out["choices"][0]; m = ch["message"]
        calls = [(c["function"]["name"], c["function"]["arguments"]) for c in (m.get("tool_calls") or [])]
        return f"{time.perf_counter()-t:5.2f}s {ch['finish_reason']:10s} content={m.get('content')!r:.50} calls={calls} tok={out['usage']['completion_tokens']}"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code} {e.read().decode()[:160]}"
named = lambda n: {"type":"function","function":{"name":n}}
cases = [
 ("auto, tool question", {}, "What is the weather in Paris?"),
 ("auto, chit-chat", {}, "Say hello in one word."),
 ("required, chit-chat", {"tool_choice":"required"}, "Say hello in one word."),
 ("required, weather", {"tool_choice":"required"}, "What is the weather in Paris?"),
 ("required, two things", {"tool_choice":"required"}, "What is the weather in Tokyo and what time is it in Asia/Tokyo?"),
 ("required, no parallel", {"tool_choice":"required","parallel_tool_calls":False}, "What is the weather in Tokyo and what time is it in Asia/Tokyo?"),
 ("named get_time, weather q", {"tool_choice":named("get_time")}, "What is the weather in Paris?"),
 ("named get_weather", {"tool_choice":named("get_weather")}, "Is it cold in Oslo, in fahrenheit?"),
 ("named unknown", {"tool_choice":named("nope")}, "hi"),
 ("required, stream off, temp 1", {"tool_choice":"required","temperature":1.0,"seed":7}, "Plan my day."),
]
for name, extra, user in cases:
    print(f"{name:28s} {req(extra, user)}", flush=True)
