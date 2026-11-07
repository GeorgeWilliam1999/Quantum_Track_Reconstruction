r"""Clean reading copy of main.tex: drop reviewer comments (\js \mm \gs) and struck text (\gsx, gsxblock);
keep teal suggestions (\gss) in teal."""
import re, sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
pre, body = s.split(r"\begin{document}", 1)
def match_brace(t, k):
    assert t[k] == "{"; d = 0
    for i in range(k, len(t)):
        if t[i] == "\\": continue
        if t[i] == "{" and t[i-1] != "\\": d += 1
        elif t[i] == "}" and t[i-1] != "\\":
            d -= 1
            if d == 0: return i
    raise ValueError
def strip_cmd(t, name):
    pat = re.compile(r"\\" + name + r"\s*\{")
    out, k = [], 0
    while True:
        m = pat.search(t, k)
        if not m: out.append(t[k:]); break
        e = match_brace(t, m.end() - 1)
        out.append(t[k:m.start()]); k = e + 1
    return "".join(out)
body = re.sub(r"\\begin\{gsxblock\}.*?\\end\{gsxblock\}", "", body, flags=re.S)
body = re.sub(r"\\paragraph\{\\gsx\{[^{}]*\}\}", "", body)
for c in ("js", "mm", "gs", "gsx"):
    body = strip_cmd(body, c)
body = re.sub(r"\\footnote\{\s*\}", "", body)
body = body.replace(r"\textit{Draft, to be finalised with the results sections.}", "")
body = re.sub(r"\n[ \t]*\\\\[ \t]*\n", "\n", body)  # orphan line breaks left by removed comments
body = re.sub(r"[ \t]+\n", "\n", body); body = re.sub(r"\n{3,}", "\n\n", body)
open(dst, "w").write(pre + r"\begin{document}" + body)
