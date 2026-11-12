import json, sys, urllib.request
QUERY = "Which of these passages answers the question about the capital of France?"
PARA = ("The committee met on Tuesday to discuss the proposal, and after a long debate it "
 "agreed that the new index would be built over the winter months while the servers "
 "were quiet. Several members asked whether the change could be measured before it "
 "was deployed, and the chair promised a report at the next meeting. ")
def doc(tokens, seed):
    t = ""
    while len(t) < tokens*4:
        t += PARA
    return f"Passage {seed}. " + t[:tokens*4].rsplit(" ",1)[0]
def ask(base, docs):
    body = json.dumps({"query": QUERY, "documents": docs, "top_n": len(docs)}).encode()
    req = urllib.request.Request(base + "/v1/rerank", data=body, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)
def tau(a, b):
    n=len(a); c=d=0
    for i in range(n):
        for j in range(i+1,n):
            s=(a[i]-a[j])*(b[i]-b[j]); c+= s>0; d+= s<0
    return (c-d)/(c+d)
mode, tag = sys.argv[1], sys.argv[2]
for tokens in (123, 449):
    docs = [doc(tokens, i) for i in range(8)]
    res = {r["index"]: r["relevance_score"] for r in ask("http://127.0.0.1:18112" if mode=="peregrine" else "http://127.0.0.1:18111", docs)["results"]}
    json.dump(res, open(f"/home/shiro/tmp/s49p/{tag}-order-{tokens}.json","w"))
    print(tag, tokens, sorted(res.items(), key=lambda kv:-kv[1]))
if mode == "peregrine":
    for tokens in (123, 449):
        pg = json.load(open(f"/home/shiro/tmp/s49p/{tag.split('-')[0]}-order-{tokens}.json")) if False else None
