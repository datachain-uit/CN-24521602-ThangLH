import numpy as np
import pandas as pd
import pytest

from src.eval.protocol_v3 import DESCRIPTIVE, INCONCLUSIVE, LOSE, WIN, compare, f1_floor, loco_choice, normalized_gain, seed_average


def episodes(diffs, base=0.5, weights=None):
    """Dựng per_episode dạng dài với 2 phương pháp A, R: f1_A − f1_R = diffs; episode xáo thứ tự để kiểm việc sắp xếp."""
    n = len(diffs)
    order = np.random.default_rng(0).permutation(n)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    a = pd.DataFrame(dict(episode=order + 1, method="A", f1=base + np.asarray(diffs, dtype=float)[order], w=w[order]))
    r = pd.DataFrame(dict(episode=order + 1, method="R", f1=base, w=w[order]))
    return pd.concat([r, a], ignore_index=True)


def test_floor_and_gain_hand_values():
    assert f1_floor(6, 8) == pytest.approx(12 / 20)
    assert f1_floor(9, 10) == pytest.approx(18 / 28)
    floor = f1_floor(5, 5)
    assert floor == pytest.approx(2 / 3)
    assert normalized_gain(0.8, floor) == pytest.approx((0.8 - 2 / 3) / (1 / 3))
    assert normalized_gain(floor, floor) == pytest.approx(0.0)
    assert normalized_gain(1.0, floor) == pytest.approx(1.0)
    series = f1_floor(pd.Series([6, 9]), pd.Series([8, 10]))
    assert np.allclose(series, [0.6, 18 / 28])


def test_compare_descriptive_below_five():
    out = compare(episodes([0.1, 0.2, 0.3, 0.4]), "A", "R")
    assert out["verdict"] == DESCRIPTIVE and out["n"] == 4 and np.isnan(out["lo"]) and np.isnan(out["hi"])
    assert out["mean"] == pytest.approx(0.25)


def test_compare_win_lose_inconclusive():
    assert compare(episodes([0.10, 0.12, 0.08, 0.11, 0.09, 0.10]), "A", "R")["verdict"] == WIN
    assert compare(episodes([-0.10, -0.12, -0.08, -0.11, -0.09, -0.10]), "A", "R")["verdict"] == LOSE
    assert compare(episodes([0.3, -0.3, 0.25, -0.25, 0.01, -0.02]), "A", "R")["verdict"] == INCONCLUSIVE
    small = compare(episodes([0.010, 0.012, 0.011, 0.009, 0.010]), "A", "R")
    assert small["lo"] > 0 and small["verdict"] == INCONCLUSIVE


def test_compare_matches_gvhd_reference_code():
    per_episode = episodes(np.random.default_rng(1).normal(0.02, 0.1, 17))
    wide = per_episode.pivot(index="episode", columns="method", values="f1")
    d = (wide["A"] - wide["R"]).dropna().to_numpy()
    boot = d[np.random.default_rng(42).integers(0, len(d), (2000, len(d)))].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    out = compare(per_episode, "A", "R")
    assert out["mean"] == pytest.approx(d.mean()) and out["lo"] == pytest.approx(lo) and out["hi"] == pytest.approx(hi)


def test_compare_equal_weights_equal_unweighted():
    per_episode = episodes(np.random.default_rng(2).normal(0.0, 0.1, 12), weights=np.full(12, 7.0))
    plain, weighted = compare(per_episode, "A", "R"), compare(per_episode, "A", "R", weights="w")
    assert plain["verdict"] == weighted["verdict"] and plain["n"] == weighted["n"]
    assert np.allclose([plain["mean"], plain["lo"], plain["hi"]], [weighted["mean"], weighted["lo"], weighted["hi"]], rtol=0, atol=1e-12)


def test_compare_weighted_mean_and_determinism():
    diffs, weights = [0.1, -0.2, 0.3, 0.0, 0.05, -0.1], [1, 2, 3, 4, 5, 6]
    per_episode = episodes(diffs, weights=weights)
    out = compare(per_episode, "A", "R", weights="w")
    assert out["mean"] == pytest.approx(np.average(diffs, weights=weights))
    assert out == compare(per_episode, "A", "R", weights="w")
    shuffled = per_episode.sample(frac=1, random_state=3)
    assert out == compare(shuffled, "A", "R", weights="w")


def p5_frame():
    rows = []
    for campaign, values in {"X": {("s1", "high"): 0.9, ("s1", "low"): 0.1, ("s2", "high"): 0.2, ("s2", "low"): 0.5}, "Y": {("s1", "high"): 0.3, ("s1", "low"): 0.2, ("s2", "high"): 0.1, ("s2", "low"): 0.6}, "Z": {("s1", "high"): 0.4, ("s1", "low"): 0.1, ("s2", "high"): 0.2, ("s2", "low"): 0.7}}.items():
        for (score, direction), f1 in values.items():
            rows.append(dict(campaign=campaign, method=score, direction=direction, F1_star=f1, eligible=True))
            rows.append(dict(campaign=campaign, method=score, direction=direction, F1_star=1.0, eligible=False))
    return pd.DataFrame(rows)


def test_loco_choice_uses_other_campaigns_only():
    p5 = p5_frame()
    assert loco_choice(p5, "X") == ("s2", "low")
    changed = p5.copy()
    changed.loc[changed["campaign"].eq("X") & changed["method"].eq("s1") & changed["direction"].eq("high"), "F1_star"] = 0.0
    changed.loc[changed["campaign"].eq("X") & changed["method"].eq("s2") & changed["direction"].eq("low"), "F1_star"] = 0.0
    assert loco_choice(changed, "X") == loco_choice(p5, "X")
    with_extreme = pd.concat([p5, pd.DataFrame([dict(campaign="X", method="s1", direction="high", F1_star=100.0, eligible=True)])], ignore_index=True)
    assert loco_choice(with_extreme, "X") == ("s2", "low")
    assert loco_choice(with_extreme, "Y") == ("s1", "high")


def test_seed_average_mean_and_std():
    df = pd.DataFrame(dict(campaign="C", q=0.25, episode=[1, 1, 1, 2], method="B1", seed=[42, 43, 44, np.nan], f1_star=[0.5, 0.5, 0.8, 0.7], n_io=[5, 5, 5, 6]))
    out = seed_average(df, metrics=["f1_star"], first=["n_io"]).set_index("episode")
    assert out.loc[1, "f1_star"] == pytest.approx(0.6) and out.loc[1, "n_seeds"] == 3
    assert out.loc[1, "f1_star_seed_std"] == pytest.approx(np.std([0.5, 0.5, 0.8], ddof=1))
    assert np.isnan(out.loc[2, "f1_star_seed_std"]) and out.loc[2, "n_io"] == 6
    with pytest.raises(ValueError):
        seed_average(df.assign(n_io=[5, 6, 5, 6]), metrics=["f1_star"], first=["n_io"])
