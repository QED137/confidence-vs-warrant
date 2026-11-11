#!/usr/bin/env bash
# Write a committable summary-only copy of each result file.
# The full JSONs carry per-item probability vectors and are large; the summaries
# are a few kB and are enough to regenerate every number in the paper.
# Originals are left untouched.
set -euo pipefail
cd "$(dirname "$0")/.."

DIR="${1:-results_arc}"
python - "$DIR" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
n = 0
for p in sorted(glob.glob(os.path.join(d, "*.json"))):
    if p.endswith(".summary.json"):
        continue
    blob = json.load(open(p))
    out = p.replace(".json", ".summary.json")
    json.dump({"summary": blob["summary"], "records": {}}, open(out, "w"), indent=2)
    n += 1
    print(f"  {os.path.basename(out)}  "
          f"{os.path.getsize(p)/1e6:.1f} MB -> {os.path.getsize(out)/1e3:.1f} kB")
print(f"wrote {n} summary file(s); these are the ones git tracks")
PY
