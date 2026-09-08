# MarsPath

Deep reinforcement learning for partially observable Mars rover navigation, validated against real Jezero Crater terrain and Perseverance rover driving data — with a structural fix for the dominant real-terrain failure mode.

## Overview

This project trains DQN and PPO agents to navigate a custom Gymnasium environment in which the agent observes only a local 13×13 terrain patch, never the full map, mirroring the partial observability a real rover faces. The 13×13 window size is motivated by the sensing range of Perseverance's AutoNav system (~6 meters). Agents are trained on synthetic terrain across five independent random seeds, then evaluated on real terrain derived from NASA/USGS Digital Elevation Model data of Jezero Crater, with terrain-hazard thresholds calibrated against Perseverance's actual recorded driving path.

**Central finding (baseline):** strong synthetic-terrain performance does not reliably predict real-terrain performance. Real-terrain multi-seed evaluation reveals a substantial sim-to-real gap dominated by a deterministic policy-cycling failure mode — 100% of timeout episodes across all seeds and all model configurations were policy cycles. Stochastic action selection mitigates this for PPO but not for DQN, leaving the underlying memory limitation unresolved for half the algorithms tested.

**Central finding (structural fix):** rather than stopping at diagnosis, this project designs, implements, and validates a structural fix — history-aware hard action masking, which forbids the agent from selecting any action that would return it to a recently visited position. This eliminates cycling completely (0.0 ± 0.0% cycle rate, both algorithms, all difficulty tiers) but trades it for a higher collision rate. Layering a terrain-safety constraint on top — using information already present in the agent's own observation — resolves this trade-off, yielding **92–94% success across all six algorithm-difficulty configurations** and closing the substantial DQN–PPO capability gap present in the original evaluation (DQN Hard: 2.0% → 92.8%). Welch's t-tests confirm every improvement is statistically significant (most p<0.0001). SHAP analysis extended to the masked models shows direction-dominant attribution across every outcome, confirming the fix addresses the root cause rather than papering over symptoms. Implementing the masking fix also surfaced and required correcting a latent value-estimation instability in masked off-policy (DQN) learning.

See `MarsPath Report/MarsPath 2.0.pdf` for the full research report.

---

## Key Results

### Baseline (5-seed mean ± std, 50 real-terrain maps, deterministic action selection)

| Model | Success | Cycle Rate | Mean Reward |
|---|---|---|---|
| DQN Easy | 32.0 ± 8.0% | 8.8 ± 8.1% | 29.84 ± 11.33 |
| DQN Medium | 30.8 ± 9.5% | 32.4 ± 19.4% | 16.03 ± 20.02 |
| DQN Hard | 2.0 ± 1.4% | 94.4 ± 1.7% | −57.96 ± 2.75 |
| PPO Easy | 37.2 ± 8.8% | 37.6 ± 10.7% | 12.57 ± 18.08 |
| PPO Medium | 28.8 ± 4.6% | 56.4 ± 6.2% | −8.18 ± 10.29 |
| PPO Hard | 25.6 ± 7.4% | 68.0 ± 9.2% | −18.01 ± 14.81 |

### Stochastic Action Selection (same trained models, no retraining)

| Model | Success | Cycle Rate | Change |
|---|---|---|---|
| DQN Easy | 30.4 ± 7.0% | 0.4 ± 0.9% | −1.6 pts |
| DQN Medium | 37.6 ± 9.3% | 7.6 ± 11.5% | +6.8 pts |
| DQN Hard | 5.2 ± 1.1% | 50.0 ± 2.8% | +3.2 pts |
| PPO Easy | 49.6 ± 3.8% | 0.0 ± 0.0% | +12.4 pts |
| PPO Medium | 52.8 ± 7.9% | 0.0 ± 0.0% | +24.0 pts |
| PPO Hard | 65.6 ± 5.7% | 0.0 ± 0.0% | +40.0 pts |

### Layered Terrain-and-History Masking (5-seed mean ± std)

| Model | Success | Collision | Cycle | Timeout |
|---|---|---|---|---|
| DQN Hard | 92.8 ± 2.7% | 0.0 ± 0.0% | 0.0 ± 0.0% | 7.2% |
| PPO Hard | 93.6 ± 2.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 6.4% |
| *(all six configurations)* | 92–94% | 0.0 ± 0.0% | 0.0 ± 0.0% | 6.0–8.0% |

All twelve baseline-vs-masking success-rate comparisons are statistically significant (Welch's t-test, p<0.05, majority p<0.001, d>2 in every case). The DQN-PPO Hard-difficulty gap, significant under baseline (p=0.0017), is no longer significant under layered masking (p=0.61) — the algorithm capability gap is closed, not merely narrowed.

---

## Project Structure

```
mars_rl_project/
├── environments/
│   ├── terrain_generator.py            # Synthetic terrain generation (3 difficulty tiers)
│   ├── mars_terrain_env.py             # Custom Gymnasium POMDP environment
│   ├── history_mask_wrapper.py         # History-aware hard action masking wrapper
│   └── layered_mask_wrapper.py         # Layered terrain-safety + history masking wrapper
├── train_dqn.py                        # DQN training (Stable-Baselines3)
├── train_ppo.py                        # PPO training (Stable-Baselines3)
├── train_maskable_dqn.py               # Custom MaskableDQN training (masking-aware)
├── train_maskable_ppo.py               # MaskablePPO training (sb3-contrib)
├── run_multiseed.py                    # Multi-seed training and evaluation runner
├── test_env.py                         # Environment verification (check_env, smoke tests)
└── data_processing/
    ├── process_nasa_terrain.py         # DEM to terrain grid, stratified region sampling
    ├── validate_against_rover_path.py  # Calibrates terrain thresholds against real rover data
    ├── evaluate_on_real_terrain.py     # Evaluates trained models on real terrain, cycle detection
    ├── compare_obstacle_geometry.py    # Synthetic vs real obstacle cluster geometry analysis
    ├── build_results_dataset.py        # Unified results CSV (geometry + outcomes, all models)
    ├── render_failure_case.py          # Visualizes a single episode path over terrain
    ├── shap_analysis.py                # SHAP feature-group attribution (patch/direction/distance)
    ├── significance_tests.py           # Welch's t-test + Cohen's d for masking comparisons
    └── residual_failure_analysis.py    # Falsification-based analysis of remaining timeouts
```

---

## Setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Usage

**Train a single baseline model:**
```bash
python -m mars_rl_project.train_dqn --difficulty easy --timesteps 500000
python -m mars_rl_project.train_ppo --difficulty hard --timesteps 500000
```

**Run multi-seed baseline training and evaluation:**
```bash
python -m mars_rl_project.run_multiseed --seeds 5 --maps-dir data/processed_full
python -m mars_rl_project.run_multiseed --seeds 5 --maps-dir data/processed_full --eval-only
python -m mars_rl_project.run_multiseed --seeds 5 --maps-dir data/processed_full --eval-only --stochastic
```

**Train with history-aware or layered action masking:**
```bash
python -m mars_rl_project.train_maskable_dqn --difficulty hard --mask-mode layered --seeds 5
python -m mars_rl_project.train_maskable_ppo --difficulty hard --mask-mode layered --seeds 5
```

**Process real Jezero Crater terrain** (requires CTX DEM mosaic and Perseverance PLACES data — see docstrings in `process_nasa_terrain.py` and `validate_against_rover_path.py` for download links):
```bash
python -m mars_rl_project.data_processing.process_nasa_terrain --all-regions --stratified-bins 5 --n-maps 10
```

**Evaluate a trained model on real terrain:**
```bash
python -m mars_rl_project.data_processing.evaluate_on_real_terrain \
    --model models/dqn_easy_seed0/final_model.zip --algo dqn --maps-dir data/processed_full
```

**Run SHAP feature attribution (baseline or masked models):**
```bash
python -m mars_rl_project.data_processing.shap_analysis \
    --model models/ppo_medium_seed0/final_model --algo ppo --maps-dir data/processed_full

python -m mars_rl_project.data_processing.shap_analysis \
    --model models/ppo_hard_seed0/final_model --algo ppo --maps-dir data/processed_full --stochastic
```

**Run statistical significance tests on masking results:**
```bash
python -m mars_rl_project.data_processing.significance_tests --results-dir outputs
```

---

## Data Sources

- **Terrain elevation:** USGS Astrogeology Science Center, Jezero Crater CTX DEM mosaic (20 m/pixel, public domain)
- **Rover ground truth:** NASA PDS Geosciences Node, Mars 2020 Perseverance PLACES localization bundle

---

## Requirements

See `requirements.txt`. Core dependencies: `gymnasium`, `stable-baselines3`, `sb3-contrib`, `torch`, `rasterio`, `shap`, `scipy`, `wandb`.

---

## Citation

If you use this work, please cite the accompanying research report:

```
Bansod, S. P. (2026). Closing the Sim-to-Real Gap in Partially Observable Mars 
Rover Navigation via Structural Action Masking. MarsPath Project.
```
