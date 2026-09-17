# Archive — retired checkpoint-analysis script

**Retired 2026-09-17.** The files here are **retired, not validated diagnostic tools.** They are
retained only for the list of proposed experiments in section 3 below. Do not run them, and do not
repair them — the replacement path is described in section 4.

## 1. What was moved

| Original path | Destination |
|---|---|
| `Reasoning/run_checkpoint_analysis.py` | `Reasoning/archive/run_checkpoint_analysis.py.retired` |
| `Reasoning/.ipynb_checkpoints/run_checkpoint_analysis-checkpoint.py` | `Reasoning/archive/run_checkpoint_analysis-checkpoint.py.retired` |

Both copies were verified byte-identical before the move (484 lines,
`sha256` prefix `ea72c82562c8f472`), and both hashes are unchanged after it. Both were tracked in
git, so the move was made with `git mv` rather than a plain `mv` — the retirement therefore
propagates to the GX10 pod on pull instead of appearing there as a local deletion.

Nothing was deleted. The companion write-ups (`nextlat_checkpoint_analysis.md` and its
`.ipynb_checkpoints` copy) were left in place and are out of scope.

## 2. Why it was retired

The script never executed successfully; no checkpoint load, generated answer, or diagnostic result
was ever produced by it. Line numbers below refer to the archived file
`run_checkpoint_analysis.py.retired` and were re-verified against it after the move.

| # | Defect | Location |
|---|---|---|
| 1 | `sys.path.insert(0, '/workspace/Research/nextlat')` — container path that does not exist on this machine or the pod | L16 |
| 2 | `NextLat(NextLatConfig(**config.model))` raises `TypeError`. The `model:` block of `materialized_config.yaml` carries `bst_pair_minimum_gap`, `bst_pair_maximum_gap`, `bst_pair_subsample_rate`, `bst_single_gap_prediction_mode` and `gpt_mode`, none of which are fields of `NextLatConfig` (`models/model_nextlat.py:23-45`) | L46 |
| 3 | `speculative_propose(..., temp=1.0, ...)` — the parameter is `temperature=` (`models/model_nextlat.py:678-685`) | L257, L339 |
| 4 | `generate` and `speculative_propose` invoked on `model._unwrap_module()`, which is a `NextLatTransformer`; both methods live on the `NextLat` wrapper | L239, L257, L339 |
| 5 | Path contradiction: the checkpoint and config paths are relative to `Reasoning/` (L25-26), while `StarGraphDataModule` opens its dataset paths relative to `nextlat/NextLat/` (L36-44). No single working directory satisfies both. | L25-26, L36 |

Defect 2 is the reason the replacement uses `core_train.initialize_model`, which selects config
fields by name (`core_train.py:63-90`) instead of splatting the config node. Defect 5 is the reason
the replacement takes an **absolute** `--checkpoint_path` and runs with cwd = `nextlat/NextLat/`.

## 3. Proposed experiments (the reason this file is kept)

These are **proposals**, not results. Nothing here has been executed or validated. They map onto
Steps 5-9 of the audit in `Current_Plan.md` and are worth reading before those steps are designed.

| Lines | Function | Proposed measurement | Audit step |
|---|---|---|---|
| L23-31 | `load_checkpoint_and_config` | direct `torch.load` + `OmegaConf.load` (superseded — see section 4) | 2 |
| L34-56 | `setup_model_and_data` | model construction and StarGraph data wiring (superseded) | 2 |
| L59-95 | `measure_horizon_error` | dynamics rollout error at depth k = 1..20 under ground-truth tokens | 6 |
| L98-147 | `measure_rollout_mismatch` | teacher-forced vs free-running rollout comparison | 6, 7 |
| L151-231 | `measure_collapse` | state geometry, with an inner `effective_rank` at L192 | 8 |
| L234-287 | `measure_memory_sufficiency` | ordinary generation vs transition-only vs speculative on matched prompts | 7 |
| L289-323 | `probe_backbone_quality` | sklearn logistic-regression probes at k ∈ {1, 2, 3, 5, 8, 10} | 8 |
| L326-378 | `benchmark_speculative` | acceptance and timing at `gamma=4` | 9 |
| L380-481 | `main` | runs all six and writes `checkpoint_analysis_results.json` | — |

Two cautions carried forward from source inspection:

- `max_k=20` in `measure_horizon_error` exceeds the trained `mtp_horizon: 8`
  (`models/model_nextlat.py` horizon loop L374-408), so depths 9-20 are extrapolation beyond the
  training distribution and must be labelled as such.
- The inner `effective_rank` (L192) and the repo's own `ModelBase.compute_hidden_state_rank`
  (`models/model_base.py:466`) are both **uncentered**. Step 8's "centered variance" needs a
  centred variant rather than either of these as-is.

## 4. What replaces it

The repository already contains a working inference-time loading recipe; use it rather than
rebuilding one. Reference: `nextlat/NextLat/data/manhattan/generate_manhattan_trajectories.py:65-91`
— config merge → `L.Fabric()` (asserts `world_size == 1`) → `fabric.seed_everything(...)` →
`core_train.initialize_model(fabric, config, tokenizer, initialize_optimizer=False,
checkpoint_path=...)`, which routes through `ModelBase.load_checkpoint`
(`models/model_base.py:419-441`).

That native loader is the single loading authority. Do not add a second `torch.load` path to produce
diagnostics or key reports.
