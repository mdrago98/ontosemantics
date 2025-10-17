import pandas as pd
import json
from collections import defaultdict
from sklearn.metrics import classification_report
from hierarchical_verifier import HierarchicalVerifier
from concurrent.futures import ThreadPoolExecutor, as_completed

hv = HierarchicalVerifier()

# ---------- Load CSV ----------
df = pd.read_csv("../../data/results_bioc.csv")

cases = []
for doc_id, group in df.groupby("document_id"):
    entities = set()
    relations = []
    for _, row in group.iterrows():
        entities.add(row["entity1"])
        entities.add(row["entity2"])
        relations.append({
            "subject": row["entity1"],
            "predicate": row["predicted_relation"],
            "object": row["entity2"],
            "gold": row["relation"]
        })
    case = {
        "id": str(doc_id),
        "text": group["abstract_clean"].iloc[0] if "abstract_clean" in group else group["abstract"].iloc[0],
        "entities": [{"text": e} for e in entities],
        "relations": relations
    }
    cases.append(case)

# ---------- Run verification (parallel) ----------
def process_case(case):
    out = hv.run_case_stream(case)
    # Collect results for each relation in the doc
    results = []
    for rel in case["relations"]:
        key = f"{rel['subject']}:{rel['predicate']}:{rel['object']}"
        val = out.get("validation_status", {}).get(key, {})
        results.append({
            "id": case["id"],
            "subject": rel["subject"],
            "predicate": rel["predicate"],
            "object": rel["object"],
            "gold": rel["gold"],
            "pred_base": rel["predicate"],
            "pred_verified": val.get("label", "no_relation") if val else "no_relation",
            "score": val.get("score", 0.0),
            "inferred": out.get("inferred_relations", [],),
            "entity_mappings": out["entity_mappings"],
            "validation_status": out["validation_status"]
        })
    return results

save_path = "biored_with_verification.csv"

pd.DataFrame([], columns=[
    "id","subject","predicate","object","gold",
    "pred_base","pred_verified","score","inferred", 'entity_mappings', 'validation_status'
]).to_csv(save_path, index=False)

results = []
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_case, case) for case in cases]
    for i, f in enumerate(as_completed(futures), 1):
        rows = f.result()
        results.extend(rows)

        # append rows to CSV
        pd.DataFrame(rows).to_csv(save_path, mode="a", header=False, index=False)



res_df = pd.DataFrame(results)
res_df.to_csv("biored_with_verification.csv", index=False)

# ---------- Metrics ----------
y_true = res_df["gold"].tolist()
y_base = res_df["pred_base"].tolist()
y_ver = res_df["pred_verified"].tolist()

print("\n=== Base Extractor ===")
print(classification_report(y_true, y_base, zero_division=0))

print("\n=== After Verification ===")
print(classification_report(y_true, y_ver, zero_division=0))

# ---------- Inference Metrics ----------
inferred_gold, inferred_pred = [], []
for r in results:
    gold = r["gold"]
    for inf in r["inferred"]:
        inferred_pred.append(inf["predicate"])
        inferred_gold.append(gold)

if inferred_pred:
    print("\n=== Inferred Relations ===")
    print(classification_report(inferred_gold, inferred_pred, zero_division=0))
else:
    print("\n=== Inferred Relations ===")
    print("No inferred relations produced.")
