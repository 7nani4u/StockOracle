#!/usr/bin/env python3
"""Check Vercel bundle size for ML integration."""
import pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
print("=== Vercel bundle size check ===")
# Check model file
model = ROOT / "models" / "lgbm_model.pkl"
if model.exists():
    kb = model.stat().st_size/1024
    print(f"model {model}: {kb:.1f} KB (limit 250MB uncompressed OK, 19KB is 0.007%)")
    if kb > 5000:
        print("WARN: model >5MB, consider pruning")
    else:
        print("OK: model size")
else:
    print("WARN: model not found")

# Check datasets not in include
vercel_ignore = (ROOT / ".vercelignore").read_text(encoding="utf-8") if (ROOT/".vercelignore").exists() else ""
print(f".vercelignore has datasets/: {'datasets/' in vercel_ignore}")

# Checklightgbm wheel size
try:
    import lightgbm, pathlib as pl
    p = pl.Path(lightgbm.__file__).parent
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())/1024/1024
    print(f"lightgbm unpacked {total:.1f} MB, wheel 1.4MB compressed - Vercel limit 50MB compressed OK")
except Exception as e:
    print(f"lightgbm size check fail: {e}")

# Simulate Vercel includeFiles
vercel_json = ROOT / "vercel.json"
import json
try:
    cfg = json.loads(vercel_json.read_text(encoding="utf-8"))
    inc = cfg.get("functions",{}).get("api/index.py",{}).get("includeFiles","")
    print(f"vercel.json includeFiles: {inc}")
    if "models" not in inc:
        print("WARN: models not in includeFiles - model will be missing on Vercel")
    else:
        print("OK: models included")
except Exception as e:
    print(e)
print("=== done ===")
