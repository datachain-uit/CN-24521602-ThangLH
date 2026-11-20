
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "configs" / "paths.json", encoding="utf-8") as f: PATHS = json.load(f)
OUTPUT = ROOT / PATHS["t6b_full_dir"]

df = pd.read_csv(OUTPUT / "results_all.csv")

metrics = ["f1_star", "homogeneity", "weighted_precision"]

keys = ["pipeline", "configuration", "data_seed", "community_seed"]

assert len(df) == 2100
assert not df.duplicated(keys).any()
assert df[metrics].notna().all().all()

# Allow negligible floating-point errors.
TOLERANCE = 1e-12

assert df[metrics].apply(lambda s: s.between(-TOLERANCE, 1.0 + TOLERANCE).all()).all(), "Metric values exceed numerical tolerance"

# Normalize negligible numerical deviations.
df[metrics] = df[metrics].clip(lower=0.0, upper=1.0)


assert set(df["data_seed"]) == set(range(20261108, 20261138))
assert set(df["community_seed"]) == set(range(42, 47))

grouped = df.groupby(["pipeline", "configuration"])

assert len(grouped) == 14
assert grouped.size().eq(150).all()

records = []

for (pipeline, config), group in grouped:
    record = {"pipeline": pipeline, "configuration": config, "n_records": len(group)}

    for metric in metrics:
        for name, quantile in [("p10", 0.1), ("median", 0.5), ("p90", 0.9)]:
            record[f"{metric}_{name}"] = group[metric].quantile(quantile)

    records.append(record)

summary = pd.DataFrame(records)

summary.to_csv(OUTPUT / "summary_statistics.csv", index=False)

u4 = []

THRESHOLD = 0.67

for pipeline, group in df[df["configuration"] == "3"].groupby("pipeline"):

    f1 = group["f1_star"]

    p10 = f1.quantile(0.1)
    p90 = f1.quantile(0.9)
    n_success = int((f1 >= THRESHOLD).sum())

    u4.append({
        "pipeline": pipeline,
        "n_records": len(group),
        "n_data_seeds": group["data_seed"].nunique(),
        "min": f1.min(),
        "p10": p10,
        "median": f1.median(),
        "p90": p90,
        "max": f1.max(),
        "published_f1": THRESHOLD,
        "n_f1_ge_threshold": n_success,
        "fraction_f1_ge_threshold": n_success / len(group),
        "u4_pass": p10 <= THRESHOLD <= p90,
        "community_seed_effective": bool(group["community_seed_effective"].all()),
    })

u4_df = pd.DataFrame(u4)

u4_df.to_csv(OUTPUT / "u4_distribution.csv", index=False)

beta_frames = [pd.read_csv(path) for path in sorted(OUTPUT.glob("betas_*.csv"))]

betas = pd.concat(beta_frames, ignore_index=True)

assert len(betas) == 180

assert not betas.duplicated(["pipeline", "simulation", "data_seed"]).any()

betas.to_csv(OUTPUT / "betas_all.csv", index=False)

beta_records = []

for (pipeline, sim), group in betas.groupby(["pipeline", "simulation"]):
    values = group["beta"]

    at_boundary = int((values == 10.0).sum())

    median = float(values.median())

    beta_records.append({
        "pipeline": pipeline,
        "simulation": sim,
        "n": len(group),
        "beta_min": values.min(),
        "beta_median": median,
        "beta_max": values.max(),
        "median_reported": ">=10" if median == 10.0 else str(median),
        "n_upper_boundary": at_boundary,
        "fraction_upper_boundary": at_boundary / len(group),
    })

beta_summary = pd.DataFrame(beta_records)

beta_summary.to_csv(OUTPUT / "beta_summary.csv", index=False)

print("=== T6b VALIDATION ===")
print("Result rows:", len(df))
print("Beta rows:", len(betas))
print("Summary groups:", len(summary))

print("\n=== U4 DISTRIBUTION ===")
print(u4_df.to_string(index=False))

print("\n=== BETA SUMMARY ===")
print(beta_summary.to_string(index=False))

print("\nALL SUMMARY CHECKS PASSED")
