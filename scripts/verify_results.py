"""Recompute the reported counts and release decisions from saved incident records."""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def metrics(frame):
    joint = frame.action.eq("joint")
    service = frame.action.eq("service")
    released = joint | service
    joint_wrong = int((joint & ~frame.correct).sum())
    service_wrong = int((service & ~frame.correct).sum())
    return {
        "n": len(frame),
        "J": int(joint.sum()),
        "S": int(service.sum()),
        "R": int((~released).sum()),
        "wrong": joint_wrong + service_wrong,
        "joint_wrong": joint_wrong,
        "C_J": float(joint.mean()),
        "C_A": float(released.mean()),
        "R_J": joint_wrong / int(joint.sum()) if joint.any() else None,
        "R_A": (joint_wrong + service_wrong) / int(released.sum()) if released.any() else None,
    }


def close(actual, expected, label):
    if actual is None or pd.isna(actual):
        assert expected is None or pd.isna(expected), label
    else:
        assert expected is not None and abs(actual - float(expected)) < 1e-10, label


def check_summary(actions, summary, scope):
    for row in summary.to_dict("records"):
        frame = actions[actions.method.eq(row["method"])]
        if "model" in row and isinstance(row["model"], str):
            frame = frame[frame.model.eq(row["model"])]
        if row[scope] != "ALL":
            frame = frame[frame.suite.eq(row[scope]) | frame.dataset.eq(row[scope])]
        assert frame.case.nunique() == len(frame) and len(frame) > 0
        actual = metrics(frame)
        for key, value in actual.items():
            if key in row:
                close(value, row[key], f'{row["method"]}/{row[scope]}/{key}')


def apply_rule(frame, rule):
    def decide(row):
        if not row.valid or not row.get("supported", True):
            return "refer"
        if (row.service_score < rule["service_threshold"] or
                row.service_margin < rule["margin"]):
            return "refer"
        return "joint" if row.joint_score >= rule["joint_threshold"] else "service"
    return frame.apply(decide, axis=1)


def main():
    revision = DATA / "revision"
    actions = pd.read_csv(revision / "actions.csv")
    summary = pd.read_csv(revision / "summary.csv")
    check_summary(actions, summary, "scope")

    primary = actions[actions.method.eq("fixed_service")].copy()
    observed = metrics(primary)
    assert (observed["n"], observed["J"], observed["S"], observed["R"]) == (645, 397, 162, 86)
    assert (observed["wrong"], observed["joint_wrong"]) == (40, 34)
    close(observed["C_J"], 397 / 645, "complete coverage")
    close(observed["R_J"], 34 / 397, "complete error")
    close(observed["R_A"], 40 / 559, "overall release error")

    rules = json.loads((revision / "selected_rules.json").read_text())
    rules = {r["fold"]: r["rule"] for r in rules
             if r["method"] == "fixed_service" and r["budget"] == 0.05}
    for fold, frame in primary.groupby("fold"):
        assert apply_rule(frame, rules[fold]).equals(frame.action)

    groups = primary.set_index("case").group
    test_cases = []
    for fold in range(5):
        config = json.loads((DATA / f"retrospective/fold_{fold}.json").read_text())
        parts = [set(config[k]) for k in ["train", "calibration", "test"]]
        assert set.union(*parts) == set(groups.index)
        group_sets = [set(groups.loc[list(part)].unique()) for part in parts]
        assert not group_sets[0] & group_sets[1]
        assert not group_sets[0] & group_sets[2]
        assert not group_sets[1] & group_sets[2]
        test_cases.extend(config["test"])
    assert len(test_cases) == len(set(test_cases)) == 645

    transitions = pd.read_csv(revision / "revision_decomposition.csv")
    transitions = transitions[transitions.method.eq("fixed_service_direct")]
    assert Counter(transitions.transition) == {
        "retained": 313, "damaged": 52, "repaired": 209, "other": 71}

    provider = pd.read_csv(revision / "anchored_provider_actions.csv")
    provider_summary = pd.read_csv(revision / "anchored_provider_summary.csv")
    for row in provider_summary.to_dict("records"):
        frame = provider[provider.model.eq(row["model"])]
        assert len(frame) == frame.case.nunique() == 270
        actual = metrics(frame)
        for key in ["J", "S", "wrong", "joint_wrong", "C_J", "C_A", "R_J", "R_A"]:
            close(actual[key], row[key], f'{row["model"]}/{key}')

    external = pd.read_csv(DATA / "external/actions.csv")
    hybrid = external[external.method.eq("hybrid_joint")]
    external_metrics = metrics(hybrid)
    assert (external_metrics["n"], external_metrics["J"], external_metrics["joint_wrong"]) == (45, 23, 3)
    frozen = json.loads((DATA / "external/frozen.json").read_text())
    rule = frozen["gates"]["hybrid_joint"]["rule"]
    assert apply_rule(hybrid, rule).equals(hybrid.action)

    cohort = json.loads((DATA / "external/cohort.json").read_text())["rows"]
    assert Counter(r["split"] for r in cohort) == {
        "development": 36, "calibration": 36, "test": 45}
    for attribute in ["session", "group"]:
        parts = [{r[attribute] for r in cohort if r["split"] == stage}
                 for stage in ["development", "calibration", "test"]]
        assert not parts[0] & parts[1]
        assert not parts[0] & parts[2]
        assert not parts[1] & parts[2]

    print(json.dumps({
        "status": "passed",
        "retrospective": observed,
        "external": external_metrics,
        "provider_streams": int(provider.model.nunique()),
        "folds": 5,
    }, indent=2))


if __name__ == "__main__":
    main()
