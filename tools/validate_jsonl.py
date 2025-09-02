import json, sys, pathlib
from jsonschema import Draft202012Validator

schema_path = pathlib.Path(sys.argv[1])
jsonl_path  = pathlib.Path(sys.argv[2])
schema = json.loads(schema_path.read_text(encoding="utf-8"))
validator = Draft202012Validator(schema)

ok = True
for i, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip(): 
        continue
    obj = json.loads(line)
    errs = sorted(validator.iter_errors(obj), key=lambda e: e.path)
    for e in errs:
        ok = False
        print(f"{jsonl_path}:{i}: {list(e.path)} -> {e.message}")
if ok:
    print("OK")

