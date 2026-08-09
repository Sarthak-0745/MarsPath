# MarsPath

Deep reinforcement learning for partially observable Mars rover navigation, validated against real Jezero Crater terrain and Perseverance rover driving data.

## Overview

This project trains DQN and PPO agents to navigate a custom Gymnasium environment in which the agent observes only a local 13x13 terrain patch, never the full map, mirroring the partial observability a real rover faces. Agents are trained on synthetic terrain, then evaluated on real terrain derived from NASA/USGS Digital Elevation Model data of Jezero Crater, with terrain-hazard thresholds calibrated against Perseverance's actual recorded driving path.

The central finding: strong synthetic-terrain performance does **not** reliably predict real-terrain performance. Real-terrain evaluation reveals a substantial sim-to-real gap dominated by a deterministic policy-cycling failure mode, which is largely mitigated for PPO (but not DQN) by switching to stochastic action selection at evaluation time, no retraining required. See `MarsPath_Research_Report.docx` for the full writeup.

## Project structure

```
mars_rl_project/
├── environments/
│   ├── terrain_generator.py       # Synthetic terrain generation (3 difficulty tiers)
│   └── mars_terrain_env.py        # Custom Gymnasium POMDP environment
├── train_dqn.py                   # DQN training (Stable-Baselines3)
├── train_ppo.py                   # PPO training (Stable-Baselines3)
├── test_env.py                    # Environment verification (check_env, smoke tests)
└── data_processing/
    ├── process_nasa_terrain.py         # DEM -> terrain grid, stratified region sampling
    ├── validate_against_rover_path.py  # Calibrates terrain thresholds against real rover driving data
    ├── evaluate_on_real_terrain.py     # Evaluates trained models on real terrain, cycle detection
    ├── compare_obstacle_geometry.py    # Synthetic vs. real obstacle cluster geometry analysis
    ├── build_results_dataset.py        # Unified results CSV (geometry + outcomes, all models)
    ├── render_failure_case.py          # Visualizes a single episode's path over terrain
    └── shap_analysis.py                # SHAP feature-group attribution (patch/direction/distance)
```

## Setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Usage

Train an agent:
```powershell
python -m mars_rl_project.train_dqn --difficulty easy --timesteps 500000
python -m mars_rl_project.train_ppo --difficulty hard --timesteps 500000
```

Process real Jezero Crater terrain (requires the CTX DEM mosaic and Perseverance PLACES data — see docstrings in `process_nasa_terrain.py` and `validate_against_rover_path.py` for download links):
```powershell
python -m mars_rl_project.data_processing.process_nasa_terrain --all-regions --stratified-bins 5 --n-maps 10
```

Evaluate a trained model on real terrain:
```powershell
python -m mars_rl_project.data_processing.evaluate_on_real_terrain --model models/dqn_easy/final_model.zip --algo dqn --maps-dir data/processed_full
```

## Data sources

- Terrain elevation: USGS Astrogeology Science Center, Jezero Crater CTX DEM mosaic (20 m/pixel, public domain)
- Rover ground truth: NASA PDS Geosciences Node, Mars 2020 Perseverance PLACES localization bundle

## Requirements

See `requirements.txt`. Core dependencies: `gymnasium`, `stable-baselines3`, `torch`, `rasterio`, `shap`, `wandb`.
