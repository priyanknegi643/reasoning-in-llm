#!/usr/bin/env python3
"""
Step 2b: artifact inventory, configuration provenance, log-row linkage, diagnostic subset.

Read-only. Does NOT call torch.load and does not import torch. Archive members of each
checkpoint are listed via the zip index only; no claim is made about the top-level
dictionary keys of the serialized object -- that is settled at Step 2c by the native load.

Run from anywhere:
    python Reasoning/diag/inventory.py
Paths are resolved from this file's location, not the working directory.
"""

import csv
import hashlib
import json
import os
import re
import sys
import zipfile
from datetime import datetime, timezone

DIAG_DIR = os.path.dirname(os.path.abspath(__file__))
REASONING = os.path.dirname(DIAG_DIR)
NEXTLAT = os.path.join(REASONING, "nextlat", "NextLat")
OUT_ROOT = os.path.join(NEXTLAT, "output", "stargraph")

# Diagnostic subset size and selection seed. Both are recorded in the output.
SUBSET_N = 200
SUBSET_SEED = 20260917

CKPT_ITER_RE = re.compile(r"ckpt_iter_(\d+)(?:_([0-9.]+))?\.pt$")

# Config fields compared across runs. This is a SELECTION, not the whole config --
# its hash identifies these values only, never the complete configuration.
SELECTED_FIELDS = [
    "seed",
    "use_nextlat",
    "use_bst",
    "use_mtp_gloeckle",
    "use_mtp_jtp",
    "trainer.compile",
    "trainer.experiment_name",
    "trainer.train_batches",
    "trainer.val_batches",
    "trainer.test_batches",
    "trainer.val_interval",
    "trainer.test_interval",
    "trainer.save_last_checkpoint",
    "trainer.save_best_checkpoint",
    "trainer.sampling_mode",
    "trainer.out_dir",
    "data.dataset",
    "data.effective_batch_size",
    "data.stargraph_train_data_path",
    "data.stargraph_test_data_path",
    "data.stargraph_max_nodes",
    "data.test_generalization",
    "model.n_layer",
    "model.n_head",
    "model.n_embd",
    "model.vocab_size",
    "model.block_size",
    "model.context_length",
    "model.mtp_horizon",
    "model.lambda_kl",
    "model.lambda_mse",
    "model.lambda_ce",
    "model.proj_factor",
    "optimizer.learning_rate",
]


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_yaml(path):
    """Prefer OmegaConf (repo's own loader); fall back to PyYAML; else None."""
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(OmegaConf.load(path), resolve=True), "omegaconf"
    except Exception:
        pass
    try:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f), "pyyaml"
    except Exception as e:
        return None, f"unavailable: {type(e).__name__}"


def dig(d, dotted):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return "<absent>"
        cur = cur[part]
    return cur


# --------------------------------------------------------------------------- #
# 1. Artifact inventory
# --------------------------------------------------------------------------- #
def inventory_runs():
    runs = []
    for name in sorted(os.listdir(OUT_ROOT)):
        run_dir = os.path.join(OUT_ROOT, name)
        if not os.path.isdir(run_dir):
            continue

        entry = {"run_directory": name, "checkpoints": [], "metrics_csv": None}

        cfg_path = os.path.join(run_dir, "materialized_config.yaml")
        if os.path.isfile(cfg_path):
            cfg, loader = load_yaml(cfg_path)
            entry["materialized_config"] = {
                "path": os.path.relpath(cfg_path, REASONING),
                "file_sha256": sha256_file(cfg_path),
                "loader": loader,
            }
            if cfg is not None:
                summary = {f: dig(cfg, f) for f in SELECTED_FIELDS}
                entry["materialized_config"]["selected_field_summary"] = summary
                entry["materialized_config"]["selected_field_summary_sha256"] = (
                    sha256_text(json.dumps(summary, sort_keys=True, default=str))
                )
                entry["materialized_config"]["selected_field_summary_note"] = (
                    "Hash covers the SELECTED fields above only, not the complete "
                    "configuration. Use file_sha256 to identify the whole artifact."
                )
        else:
            entry["materialized_config"] = None

        for fn in sorted(os.listdir(run_dir)):
            if not fn.endswith(".pt"):
                continue
            pt = os.path.join(run_dir, fn)
            m = CKPT_ITER_RE.search(fn)
            ck = {
                "filename": fn,
                "size_bytes": os.path.getsize(pt),
                "mtime_utc": datetime.fromtimestamp(
                    os.path.getmtime(pt), timezone.utc
                ).isoformat(),
                "sha256": sha256_file(pt),
                "filename_iteration": int(m.group(1)) if m else None,
                "filename_loss_suffix": m.group(2) if m and m.group(2) else None,
                "top_level_keys": None,
                "top_level_keys_status": "pending_runtime_confirmation",
                "top_level_keys_note": (
                    "Not inspected. Archive members below are zip entries; entry names "
                    "are not evidence of top-level dictionary keys. Settled at Step 2c."
                ),
            }
            try:
                with zipfile.ZipFile(pt) as z:
                    members = z.namelist()
                ck["archive"] = {
                    "member_count": len(members),
                    "first_members": members[:12],
                    "inspection_method": "zip index only; no torch.load, no unpickling",
                }
            except Exception as e:
                ck["archive"] = {"error": f"{type(e).__name__}: {e}"}
            entry["checkpoints"].append(ck)

        mcsv = os.path.join(run_dir, "version_0", "metrics.csv")
        if os.path.isfile(mcsv):
            with open(mcsv) as f:
                n_rows = sum(1 for _ in f) - 1
            entry["metrics_csv"] = {
                "path": os.path.relpath(mcsv, REASONING),
                "file_sha256": sha256_file(mcsv),
                "data_rows": n_rows,
            }

        entry["has_weights"] = bool(entry["checkpoints"])
        runs.append(entry)
    return runs


def inventory_latest_ckpt():
    p = os.path.join(NEXTLAT, "output", "stargraph", "latest_ckpt")
    if not os.path.isfile(p):
        return {"present": False}
    raw = open(p).read().strip()
    # The stored path is relative to the NextLat repo root.
    resolved = os.path.normpath(os.path.join(NEXTLAT, raw))
    return {
        "present": True,
        "raw_contents": raw,
        "relative_to": "nextlat/NextLat/",
        "resolved_path": os.path.relpath(resolved, REASONING),
        "resolves": os.path.isfile(resolved),
    }


# --------------------------------------------------------------------------- #
# 2. Log-row linkage
# --------------------------------------------------------------------------- #
def linkage_rows(runs):
    """
    metrics.csv carries an all-zero `step` column, so the training step is derived,
    never read. Rows are classified by which column families are populated.

    A mapping is only emitted where the row layout supports it. Where it does not,
    mapping_status is 'ambiguous' or 'unresolved' and derived_training_step is null.
    """
    out = []
    for run in runs:
        if not run["metrics_csv"]:
            continue
        name = run["run_directory"]
        mcsv = os.path.join(REASONING, run["metrics_csv"]["path"])
        cfg = (run.get("materialized_config") or {}).get("selected_field_summary") or {}
        val_interval = cfg.get("trainer.val_interval")
        test_interval = cfg.get("trainer.test_interval")
        train_batches = cfg.get("trainer.train_batches")

        rows = list(csv.DictReader(open(mcsv)))
        acc_keys = [k for k in (rows[0] if rows else {}) if k and "test_accuracy" in k]

        # One row per optimizer step is the only layout this mapping is valid for.
        one_row_per_step = (
            isinstance(train_batches, int) and len(rows) == train_batches
        )

        ckpt_iters = {
            c["filename_iteration"]: c["filename"]
            for c in run["checkpoints"]
            if c["filename_iteration"] is not None
        }

        for i, r in enumerate(rows):
            have = {k for k, v in r.items() if v not in (None, "")}
            families = []
            if "loss" in have:
                families.append("training")
            if "val/loss" in have:
                families.append("validation")
            if any(k in have for k in acc_keys):
                families.append("accuracy")
            event_type = "+".join(families) or "unknown"

            step = None
            status = "unresolved"
            evidence = []
            if not one_row_per_step:
                evidence.append(
                    f"row count {len(rows)} != trainer.train_batches {train_batches}; "
                    "one-row-per-step layout not established"
                )
                status = "ambiguous"
            else:
                step = i + 1
                status = "derived"
                evidence.append(
                    f"row_index+1; {len(rows)} rows == train_batches {train_batches}"
                )
                if "validation" in families and isinstance(val_interval, int):
                    if step % val_interval == 0:
                        evidence.append(
                            f"validation row at step {step}, consistent with "
                            f"val_interval {val_interval}"
                        )
                    else:
                        status = "ambiguous"
                        evidence.append(
                            f"validation row at derived step {step} is NOT a multiple "
                            f"of val_interval {val_interval}"
                        )
                if "accuracy" in families and isinstance(test_interval, int):
                    if step % test_interval != 0:
                        status = "ambiguous"
                        evidence.append(
                            f"accuracy row at derived step {step} is NOT a multiple "
                            f"of test_interval {test_interval}"
                        )

            ck = ""
            if status == "derived" and step in ckpt_iters:
                ck = ckpt_iters[step]

            out.append(
                {
                    "run_directory": name,
                    "csv_row_index": i,
                    "event_type": event_type,
                    "derived_training_step": step if step is not None else "",
                    "mapping_status": status,
                    "evidence": "; ".join(evidence),
                    "checkpoint_match": ck,
                }
            )
    return out


def unmatched_checkpoints(runs, links):
    """Checkpoints with no evaluation row. Absence of a row is the expected case for
    a last-checkpoint save; it is recorded, not treated as an error."""
    matched = {
        (l["run_directory"], l["checkpoint_match"])
        for l in links
        if l["checkpoint_match"]
    }
    notes = []
    for run in runs:
        for c in run["checkpoints"]:
            if (run["run_directory"], c["filename"]) not in matched:
                notes.append(
                    {
                        "run_directory": run["run_directory"],
                        "checkpoint": c["filename"],
                        "note": (
                            "No log row maps to this checkpoint's iteration. Checkpoint "
                            "saving and evaluation are gated separately in core_train.py, "
                            "so a last-checkpoint save need not coincide with an "
                            "evaluation event. Do not infer its metrics from a nearby row."
                        ),
                    }
                )
    return notes


# --------------------------------------------------------------------------- #
# 3. Configuration and decoding provenance
# --------------------------------------------------------------------------- #
def config_provenance():
    """
    Two of the three stages in the plan. Stage 3 (the settings actually used for the
    new execution) is written by smoke_load_generate.py, not here.

    Applies the repository's own merge: OmegaConf.merge(defaults.yaml, <config>).
    Probes the decoding keys the evaluator reads at data/stargraph.py:125-126.
    """
    stages = []
    try:
        from omegaconf import OmegaConf
    except Exception as e:
        return {
            "status": f"omegaconf unavailable ({type(e).__name__}); run on the pod env",
            "stages": [],
        }

    defaults_path = os.path.join(NEXTLAT, "defaults.yaml")
    targets = [
        (
            "defaults + source YAML",
            os.path.join(
                NEXTLAT, "config", "stargraph", "2_10", "nextlat_stargraph_2_10_ckpt.yaml"
            ),
        ),
        (
            "defaults + materialized_config.yaml",
            os.path.join(
                OUT_ROOT,
                "NextLat-proj_factor0.5_lambda_kl1.0_mtp_horizon8_seed1234_lambda_mse1.0",
                "materialized_config.yaml",
            ),
        ),
    ]

    for label, path in targets:
        rec = {"stage": label, "config_path": os.path.relpath(path, REASONING)}
        if not os.path.isfile(path):
            rec["error"] = "file not found"
            stages.append(rec)
            continue
        cfg = OmegaConf.merge(OmegaConf.load(defaults_path), OmegaConf.load(path))
        rec["struct_mode"] = OmegaConf.is_struct(cfg)
        # Exactly the expressions the evaluator uses.
        rec["observed"] = {
            "temperature": repr(getattr(cfg.trainer, "temperature", 1.0)),
            "top_k": repr(getattr(cfg.trainer, "top_k", None)),
            "sampling_mode": repr(getattr(cfg.trainer, "sampling_mode", "<absent>")),
            "speculative_gamma": repr(
                getattr(cfg.trainer, "speculative_gamma", "<absent>")
            ),
            "compile": repr(getattr(cfg.trainer, "compile", "<absent>")),
        }
        rec["keys_present_in_trainer"] = {
            k: (k in cfg.trainer) for k in ("temperature", "top_k", "compile")
        }
        stages.append(rec)

    return {
        "status": "ok",
        "stages": stages,
        "cli_overrides": [],
        "notes": [
            "temperature/top_k are absent from both defaults.yaml and the 2_10 configs; "
            "the values above are what the evaluator's getattr() defaults resolve to "
            "under the repository's own merge.",
            "SEED vs RNG STATE: a seed starts a sequence; ~20000 training steps consume "
            "it. The RNG state in force at any historical evaluation is UNKNOWN and is "
            "not recoverable from these files. Only the new smoke-test seed is controlled.",
            "The compile discrepancy (true in source YAML, false materialized) is why "
            "all stages are recorded separately rather than collapsed into one.",
        ],
    }


# --------------------------------------------------------------------------- #
# 4. Diagnostic subset -- NOT a holdout
# --------------------------------------------------------------------------- #
def diagnostic_subset():
    import random

    test_path = os.path.join(NEXTLAT, "data", "stargraph", "graph_2_10_test_20000.txt")
    if not os.path.isfile(test_path):
        return {"error": f"not found: {test_path}"}
    with open(test_path) as f:
        n_lines = sum(1 for _ in f)

    rng = random.Random(SUBSET_SEED)
    idx = sorted(rng.sample(range(n_lines), min(SUBSET_N, n_lines)))

    return {
        "source_file": os.path.relpath(test_path, REASONING),
        "source_file_sha256": sha256_file(test_path),
        "source_file_line_count": n_lines,
        "indexing_convention": (
            "0-based index into the newline-separated lines of source_file as written "
            "on disk, counting the first line as index 0; no shuffling, no filtering"
        ),
        "selection_seed": SUBSET_SEED,
        "selection_method": (
            f"random.Random({SUBSET_SEED}).sample(range(n_lines), {SUBSET_N}), sorted"
        ),
        "count": len(idx),
        "indices": idx,
        "IS_NOT_A_HOLDOUT": (
            "This subset belongs to the checkpoint-selection dataset. The whole file "
            "backs val_dataloader, which drives validation loss, best-checkpoint "
            "selection AND the reported accuracy (core_train.py:674-682). Selecting "
            "indices after those decisions does not make these examples unseen. Use it "
            "as a repeatable diagnostic sample for reproducing reported behaviour only; "
            "it cannot support a claim about independent test performance."
        ),
        "for_independent_evaluation": (
            "Generate or obtain examples not used in training or checkpoint selection, "
            "document their provenance, and check for overlap against both data files. "
            "Note prepare.py seeds per-sample from a counter that continues from the "
            "train count, so a regenerated test file only matches byte-for-byte when "
            "regenerated with the same combined invocation."
        ),
    }


def main():
    if not os.path.isdir(OUT_ROOT):
        sys.exit(f"output root not found: {OUT_ROOT}")
    os.makedirs(DIAG_DIR, exist_ok=True)

    runs = inventory_runs()
    latest = inventory_latest_ckpt()
    links = linkage_rows(runs)
    unmatched = unmatched_checkpoints(runs, links)

    inv = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "reasoning_root": REASONING,
        "nextlat_root": os.path.relpath(NEXTLAT, REASONING),
        "run_count": len(runs),
        "runs_with_weights": [r["run_directory"] for r in runs if r["has_weights"]],
        "latest_ckpt_pointer": latest,
        "runs": runs,
        "checkpoints_without_evaluation_row": unmatched,
        "method_note": (
            "No torch.load was performed. Checkpoint dictionary-key claims are marked "
            "pending_runtime_confirmation and are resolved by Step 2c's native load."
        ),
    }
    with open(os.path.join(DIAG_DIR, "inventory.json"), "w") as f:
        json.dump(inv, f, indent=2, default=str)

    cols = [
        "run_directory",
        "csv_row_index",
        "event_type",
        "derived_training_step",
        "mapping_status",
        "evidence",
        "checkpoint_match",
    ]
    with open(os.path.join(DIAG_DIR, "log_step_linkage.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(links)

    with open(os.path.join(DIAG_DIR, "config_provenance.json"), "w") as f:
        json.dump(config_provenance(), f, indent=2, default=str)

    with open(os.path.join(DIAG_DIR, "diagnostic_indices.json"), "w") as f:
        json.dump(diagnostic_subset(), f, indent=2, default=str)

    # --- console summary ---
    print(f"runs: {len(runs)}   with weights: {inv['runs_with_weights']}")
    for r in runs:
        for c in r["checkpoints"]:
            print(
                f"  {r['run_directory']}/{c['filename']}  "
                f"{c['size_bytes']/1e6:.1f} MB  sha256 {c['sha256'][:16]}…  "
                f"iter={c['filename_iteration']} loss_suffix={c['filename_loss_suffix']}"
            )
    print(
        f"latest_ckpt -> {latest.get('raw_contents')}  "
        f"resolves={latest.get('resolves')}"
    )
    from collections import Counter

    print("linkage mapping_status:", dict(Counter(l["mapping_status"] for l in links)))
    print(
        "linkage event_type:",
        dict(Counter(l["event_type"] for l in links)),
    )
    for u in unmatched:
        print(f"  NO EVAL ROW: {u['run_directory']}/{u['checkpoint']}")
    print(f"wrote 4 files to {DIAG_DIR}")


if __name__ == "__main__":
    main()
