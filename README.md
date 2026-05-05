# Annotation Entropy Predicts Per-Example Learning Dynamics in LoRA Fine-Tuning

## Repository layout

```
src/                              # importable library
  data/
    chaosnli.py                   # ChaosNLI loader + entropy annotation
    annotation_entropy.py         # Eq. 1 (Shannon H over annotator counts)
  analysis/                       # correlation, cartography, visualization
  training/                       # tracking utilities
  models/                         # LoRA/FT model construction

scripts/                          # numbered pipeline; see mapping below
configs/experiment.yaml           # canonical hyperparameters (reference)
```

Running the pipeline writes tracker JSONs to `results/tracking/` and plots to `figures/`; those are regenerable outputs and are not checked in.

## Environment

```bash
# Python 3.10 tested
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                  # exposes the `src` package
```

Tested on macOS 14 + Apple M-series (MPS) and on CUDA 12. `HF_TOKEN` is only needed for gated HuggingFace models; none of the six models here are gated.

## Data

SNLI and MNLI pull automatically through `datasets`. **ChaosNLI** must be downloaded manually:

1. Download `chaosNLI_v1.0.zip` from `https://github.com/easonnie/ChaosNLI` (releases page).
2. Unpack into `data/chaosnli/`, leaving the three JSONL files (`chaosNLI_snli.jsonl`, `chaosNLI_mnli_m.jsonl`, `chaosNLI_alphanli.jsonl`) at the top level.
3. Confirm with `python scripts/01_compute_entropy.py --data-dir data/chaosnli/`.

The loader at `src/data/chaosnli.py` also attempts auto-download, but that GitHub layout has changed over time; if auto-download fails, the manual path above is reliable.

## Reproduction (paper table / figure -> script)

| Paper | Command |
|---|---|
| Figure 1 (hero), Figure 2 (trajectories) | `python scripts/06_generate_figures.py` |
| Figure 3 (entropy distribution) | `python scripts/01_prepare_data.py --compute-entropy-only` |
| Table 1 (Spearman rho across 6 models x 2 datasets) | `python scripts/08_expanded_experiments.py` then `python scripts/09_aggregate_expanded.py` |
| Table 2 (un-learning in late training) | `python scripts/09_aggregate_expanded.py --unlearning` |
| Table 3 (rank sweep) | `python scripts/03_rank_sweep.py` + `python scripts/11_bert_rank_sweep.py` |
| Appendix D (noise injection robustness) | `python scripts/10_robustness_experiments.py --noise` then `python scripts/19_noise_aggregate.py` |
| Appendix F (IA3 baseline) | `python scripts/15_adapter_baseline.py` |
| Appendix G (decoder-only Qwen) | `python scripts/16_decoder_experiments.py` then `python scripts/16b_decoder_analysis.py` |
| Appendix H (cartography scatter) | `python scripts/10_robustness_experiments.py --cartography` |
| Appendix J (gradient cosine) | `python scripts/17_gradient_cosine.py` |
| Appendix K (calibration / ECE) | `python scripts/18_calibration_analysis.py` |

## Configuration at a glance

Canonical values (paper Section 3.3):

- **Models:** `roberta-base`, `bert-base-uncased`, `distilbert-base-uncased`, `microsoft/deberta-v3-base`, `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-3B`
- **LoRA:** rank `r in {4, 16}` (main), `{1, 2, 4, 8, 16, 32}` in rank sweep; `alpha = 2r`; dropout 0.05
- **Target modules:** encoders `{query, value}`, decoders `{q_proj, k_proj, v_proj, o_proj}`
- **Training:** AdamW, lr 2e-5, batch 32, 5 epochs, 6% warmup, cosine decay to 10% of peak, grad-clip 1.0
- **Tracking:** per-example CE (against majority-vote gold labels) every 100 steps + end-of-epoch + step 0 => `T ~= 39` checkpoints; AULC is their mean (Eq. 2)
- **Entropy thresholds (nats):** `[0.4, 0.7]` -> clean / ambiguous / contested
- **Seeds:** `{42, 123, 456}`

`configs/experiment.yaml` is kept as a readable reference; actual hyperparameters live at the top of the `scripts/NN_*.py` files so that runs are self-contained.

## License

Code is released under the MIT License (see `LICENSE`). ChaosNLI, SNLI, and MNLI retain their own upstream licenses; please cite those papers if you use the data.
