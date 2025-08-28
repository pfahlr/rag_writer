I've added the functions for improved citations, but I haven't used them in the output yet, these are the instructions
for using the _fmt_cited() _norm() and _extract_cited() functions

```
answer = resp.content
print(answer)

cited, uncited = _extract_cited(docs, answer)

def _fmt_cited(d):
    title = d.metadata.get("title") or Path(d.metadata.get("source"," ")).stem
    page = d.metadata.get("page")
    src  = d.metadata.get("source")
    doi  = d.metadata.get("doi")
    bits = [f"{title}"]
    if page: bits.append(f"p.{page}")
    if doi: bits.append(f"doi:{doi}")
    if src: bits.append(str(src))
    return " (" + " · ".join(bits) + ")"

print("\nCITED SOURCES:")
if cited:
    for d in cited:
        print(f"-{_fmt_cited(d)}")
else:
    print("- (none detected — consider tightening the prompt to require explicit citations by Title)")

print("\nFURTHER CONTEXT (retrieved but not cited):")
for d in uncited:
    print(f"-{_fmt(d)}")
```

and the functions _extract_cited() and _norm() for reference

```

import re

DOI_RE = re.compile(r'\b10\.\d{4,9}/[-._;()/:a-z0-9]*[a-z0-9]\b', re.I)

def _norm(s: str) -> str:
    return re.sub(r'\W+', ' ', (s or '')).strip().lower()

def _extract_cited(docs, answer: str):
    """Return (cited_docs, uncited_docs) based on title/DOI matches in the answer text."""
    ans = answer.lower()
    ans_norm = _norm(answer)
    cited, uncited = [], []

    for d in docs:
        title = d.metadata.get("title") or Path(d.metadata.get("source", " ")).stem
        title_norm = _norm(title)
        doi = (d.metadata.get("doi") or "").lower()

        hit = False
        # 1) Exact/normalized title presence
        if title and (title.lower() in ans or title_norm and title_norm in ans_norm):
            hit = True
        # 2) DOI presence
        if not hit and doi:
            if doi in ans or any(m.group(0).lower() == doi for m in DOI_RE.finditer(ans)):
                hit = True

        if hit:
            cited.append(d)
        else:
            uncited.append(d)

    return cited, uncited
```
