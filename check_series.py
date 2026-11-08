"""Throwaway check: are any two series in the trained artifact using identical history?"""
import hashlib
from collections import defaultdict

import joblib

artifact = joblib.load("models/beialert_models.joblib")

by_hash = defaultdict(list)
for key, entry in artifact["series"].items():
    values = entry["history"].to_numpy(dtype=float).round(2)
    payload = ",".join(f"{v:.2f}" for v in values).encode()
    digest = hashlib.sha1(payload).hexdigest()
    by_hash[digest].append(key)
    print(f"{key}\n  n_obs={len(values)}  last_observed_price={entry['last_observed_price']:.2f}  sha1={digest}")

print()
dupes = {h: keys for h, keys in by_hash.items() if len(keys) > 1}
if dupes:
    print(f"DUPLICATES FOUND: {len(dupes)} hash(es) shared by multiple series")
    for h, keys in dupes.items():
        print(f"  {h}:")
        for k in keys:
            print(f"    - {k}")
else:
    print(f"No duplicates: all {len(artifact['series'])} series have distinct history hashes.")
