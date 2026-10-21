import json, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
names = ['factual', 'math', 'code', 'reasoning', 'dialect', 'artifact']
base = r'D:\ka\nados-ai-web-local'
for i, n in enumerate(names, 1):
    raw = open(base + '\\n%d.txt' % i, encoding='utf-8').read()
    deltas, done = '', None
    for line in raw.split('\n'):
        line = line.strip()
        if line.startswith('data: '):
            try:
                o = json.loads(line[6:])
                if o.get('type') == 'delta':
                    deltas += o['delta']
                if o.get('type') == 'done':
                    done = o['reply']
            except Exception:
                pass
    nd = (done or {}).get('nados') or {}
    teacher = (done or {}).get('teacher')
    score = nd.get('final_score')
    has_html = bool(re.search(r'```html', deltas, re.I))
    print('=== %s | teacher=%s | score=%s | len=%d | html_block=%s' % (n, teacher, score, len(deltas), has_html))
    print(deltas[:230].replace('\n', ' | '))
    print()
