"""Build the shared, source-matched RAID subset using only Python's standard library."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
URL = "https://dataset.raid-bench.xyz/train_none.csv"
MODELS = ("chatgpt", "llama-chat", "mistral-chat", "cohere-chat")
DOMAINS = ("news", "reviews", "recipes", "poetry")
FIELDS = ("id", "source_id", "domain", "model", "prompt", "generation",
          "decoding", "repetition_penalty", "attack", "split")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(path):
    """Cache the labeled, attack-free RAID training pool (about 802 MB)."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {URL}", flush=True)
    with urlopen(URL, timeout=120) as response, partial.open("wb") as handle:
        expected = response.headers.get("Content-Length")
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    if expected and partial.stat().st_size != int(expected):
        raise ValueError("Incomplete RAID download; rerun to download again.")
    partial.replace(path)


def text_key(value):
    # Used only to check duplicates; saved text retains its original formatting.
    return " ".join(value.split()).casefold()


def rank(seed, purpose, value):
    return hashlib.sha256(f"{seed}:{purpose}:{value}".encode()).hexdigest()


def read_groups(path):
    """Retain only one generation condition and require complete matched sources."""
    groups = defaultdict(dict)
    audit = Counter()
    csv.field_size_limit(10_000_000)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(FIELDS[:-1]) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing RAID columns: {sorted(missing)}")
        for row in reader:
            audit["raw_rows"] += 1
            if (row["model"] not in MODELS or row["domain"] not in DOMAINS
                    or (row["attack"], row["decoding"], row["repetition_penalty"])
                    != ("none", "sampling", "no")):
                continue
            audit["filtered_rows"] += 1
            if not all(isinstance(row[k], str) and row[k].strip()
                       for k in ("id", "source_id", "prompt", "generation")):
                audit["empty_rows_removed"] += 1
                continue
            key = (row["domain"], row["source_id"])
            if row["model"] in groups[key]:
                raise ValueError(f"Duplicate generation for {key}, {row['model']}")
            groups[key][row["model"]] = {k: row[k] for k in FIELDS[:-1]}

    complete = {}
    for key, models in groups.items():
        if set(models) != set(MODELS):
            audit["incomplete_sources_removed"] += 1
        elif len({row["prompt"] for row in models.values()}) != 1:
            audit["mismatched_prompt_sources_removed"] += 1
        else:
            complete[key] = [models[model] for model in MODELS]
    audit["complete_sources_before_deduplication"] = len(complete)
    return complete, audit


def select_rows(groups, count, seed):
    """Deduplicate sources before sampling and split whole groups 80/10/10."""
    if count < 10 or count % 10:
        raise ValueError("--prompts-per-domain must be a positive multiple of 10.")
    candidates = defaultdict(list)
    seen_sources, seen_prompts, seen_outputs = set(), set(), set()
    excluded = Counter()
    # Stable hash ordering makes selection independent of CSV row order.
    for key in sorted(groups, key=lambda k: rank(seed, "deduplicate", repr(k))):
        domain, source_id = key
        rows = groups[key]
        prompt = text_key(rows[0]["prompt"])
        outputs = {text_key(row["generation"]) for row in rows}
        if source_id in seen_sources or prompt in seen_prompts or outputs & seen_outputs:
            excluded[domain] += 1
            continue
        seen_sources.add(source_id)
        seen_prompts.add(prompt)
        seen_outputs.update(outputs)
        candidates[domain].append(key)

    selected = []
    for domain in DOMAINS:
        keys = sorted(candidates[domain], key=lambda k: rank(seed, "sample", repr(k)))
        if len(keys) < count:
            raise ValueError(f"{domain}: need {count} matched sources, found {len(keys)}")
        keys = sorted(keys[:count], key=lambda k: rank(seed, "split", repr(k)))
        for index, key in enumerate(keys):
            split = "train" if index < count * 8 // 10 else (
                "validation" if index < count * 9 // 10 else "test")
            selected.extend({**row, "split": split} for row in groups[key])
    return selected, {d: len(candidates[d]) for d in DOMAINS}, dict(excluded)


def validate(rows, count):
    """Fail if balance, matching, or separation would invalidate experiments."""
    if len(rows) != len(DOMAINS) * len(MODELS) * count:
        raise ValueError("Unexpected dataset size")
    groups = defaultdict(list)
    partitions = {field: defaultdict(set) for field in ("source_id", "prompt", "generation")}
    for row in rows:
        groups[row["source_id"]].append(row)
        for field in partitions:
            value = row[field] if field == "source_id" else text_key(row[field])
            partitions[field][value].add(row["split"])
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Repeated generation IDs")
    for group in groups.values():
        if (len(group) != len(MODELS) or {r["model"] for r in group} != set(MODELS)
                or any(len({r[k] for r in group}) != 1 for k in ("domain", "prompt", "split"))):
            raise ValueError("Invalid matched source group")
    for field, values in partitions.items():
        if any(len(splits) > 1 for splits in values.values()):
            raise ValueError(f"Cross-split leakage in {field}")
    counts = Counter((r["domain"], r["model"], r["split"]) for r in rows)
    for domain in DOMAINS:
        for model in MODELS:
            for split, expected in (("train", count * 8 // 10),
                                    ("validation", count // 10), ("test", count // 10)):
                if counts[domain, model, split] != expected:
                    raise ValueError(f"Imbalance in {domain}/{model}/{split}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=ROOT / "data/raw/train_none.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--prompts-per-domain", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.prompts_per_domain < 10 or args.prompts_per_domain % 10:
        parser.error("--prompts-per-domain must be a positive multiple of 10")
    download(args.raw_path)
    groups, audit = read_groups(args.raw_path)
    rows, available, excluded = select_rows(groups, args.prompts_per_domain, args.seed)
    validate(rows, args.prompts_per_domain)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "raid_balanced.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "source_url": URL,
        "source_partition": "RAID train (labeled, without adversarial attacks)",
        "source_sha256": sha256_file(args.raw_path),
        "builder_sha256": sha256_file(Path(__file__)),
        "dataset_sha256": sha256_file(output),
        "seed": args.seed,
        "models": MODELS,
        "domains": DOMAINS,
        "filters": {"attack": "none", "decoding": "sampling", "repetition_penalty": "no"},
        "prompts_per_domain": args.prompts_per_domain,
        "audit": dict(audit),
        "duplicate_sources_removed_by_domain": excluded,
        "eligible_sources_by_domain": available,
        "rows": len(rows),
        "sources": len({r["source_id"] for r in rows}),
        "rows_by_model": dict(Counter(r["model"] for r in rows)),
        "rows_by_domain": dict(Counter(r["domain"] for r in rows)),
        "rows_by_split": dict(Counter(r["split"] for r in rows)),
        "split_sources": {split: sorted({r["source_id"] for r in rows if r["split"] == split})
                          for split in ("train", "validation", "test")},
        "validation": "Balanced complete groups; no cross-split source, prompt, or output duplicates",
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} rows to {output}")
    print(json.dumps({k: manifest[k] for k in ("eligible_sources_by_domain", "rows_by_split")}, indent=2))


if __name__ == "__main__":
    main()
