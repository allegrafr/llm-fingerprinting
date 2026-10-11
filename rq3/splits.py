"""Leave an entire domain out while preserving the shared source assignments."""

from collections import Counter, defaultdict

from common.data import check_splits


def make_domain_fold(rows, held_out):
    domains = {row["domain"] for row in rows}
    labels = {row["model"] for row in rows}
    if held_out not in domains or len(domains) < 2:
        raise ValueError("Expected a known held-out domain and at least two domains")
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_id"]].append(row)
    for group in groups.values():
        if (len(group) != len(labels) or {r["model"] for r in group} != labels
                or any(len({r[key] for r in group}) != 1 for key in ("domain", "split", "prompt"))):
            raise ValueError("Every source must be complete, matched, and confined to one domain/split")
    train = [r for r in rows if r["domain"] != held_out and r["split"] == "train"]
    validation = [r for r in rows if r["domain"] != held_out and r["split"] == "validation"]
    test = [r for r in rows if r["domain"] == held_out]
    check_splits(train, validation, test)
    for split in (train, validation, test):
        if len(set(Counter(r["model"] for r in split).values())) != 1:
            raise ValueError("Expected balanced model labels in every fold partition")
    return train, validation, test
