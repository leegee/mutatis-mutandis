import faiss
from pathlib import Path

bad = []
for p in sorted(Path(r"g:\corpus-out\indexes\faiss").glob("tier1_*.faiss")):
    try:
        idx = faiss.read_index(str(p))
        print("ok  ", p.name, p.stat().st_size, idx.ntotal)
        del idx
    except Exception as e:
        print("FAIL", p.name, p.stat().st_size, str(e)[:100])
        bad.append(p.name)
print(bad)