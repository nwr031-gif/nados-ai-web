import json, sys, io, glob, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
files = sys.argv[1:]
if not files:
    files = sorted(glob.glob(r"D:\ka\nados-ai-web-local\u*-out.txt"))
for f in files:
    raw = open(f, encoding="utf-8").read()
    text, done, err = "", None, None
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            try:
                o = json.loads(line[6:])
            except Exception:
                continue
            if o.get("type") == "delta":
                text += o["delta"]
            if o.get("type") == "done":
                done = o["reply"]
            if o.get("type") == "error":
                err = o.get("message")
    head = text[:120]
    dup = bool(head) and text.count(head) > 1
    fence = "```" in text
    digits = any(ch.isdigit() for ch in text)
    print("=== %s" % os.path.basename(f))
    print("teacher=%s chars=%d dup=%s fence=%s digits=%s" % (
        (done or {}).get("teacher"), len(text), dup, fence, digits))
    if err:
        print("ERROR:", err[:150])
    print(text[:500].replace("\n", " | "))
    print()
