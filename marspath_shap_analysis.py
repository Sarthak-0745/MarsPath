"""
SHAP Explainability Analysis for MarsPath
Completes Step 7 of the original proposal: Post-hoc explainability using SHAP.

Usage:
    python marspath_shap_analysis.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
from pathlib import Path
from stable_baselines3 import DQN, PPO
from mars_rl_project.environments.mars_terrain_env import MarsTerrainEnv

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
REAL_MAPS_PATH = PROJECT_ROOT / "data" / "processed_full"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "shap_analysis"

MODEL_PATHS = {
    'DQN_Medium': MODEL_DIR / "dqn_medium_seed0" / "final_model.zip",
    'DQN_Hard':   MODEL_DIR / "dqn_hard_seed0"   / "final_model.zip",
    'PPO_Medium': MODEL_DIR / "ppo_medium_seed0" / "final_model.zip",
    'PPO_Hard':   MODEL_DIR / "ppo_hard_seed0"   / "final_model.zip",
}

ANALYZE_MODELS = [
    'DQN_Medium',
    'DQN_Hard',
    'PPO_Medium',
    'PPO_Hard',
]

N_STATES = 150
BACKGROUND_SIZE = 50
N_SAMPLES = 100

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# STEP 1: LOAD REAL TERRAIN MAPS
# ============================================================

def load_real_maps(path):
    maps = []
    map_files = sorted([f for f in os.listdir(path) if f.endswith('.npz')])
    
    if len(map_files) == 0:
        raise FileNotFoundError(f"No .npz files found in {path}")
    
    for f in map_files[:50]:
        data = np.load(os.path.join(path, f))
        if 'grid' in data:
            grid = data['grid']
        elif 'terrain' in data:
            grid = data['terrain']
        elif 'map' in data:
            grid = data['map']
        elif 'arr_0' in data:
            grid = data['arr_0']
        else:
            keys = list(data.keys())
            if len(keys) == 1:
                grid = data[keys[0]]
            else:
                print(f"  Warning: Unknown keys in {f}: {keys}")
                continue
        maps.append(grid)
    
    print(f"Loaded {len(maps)} real-terrain maps from {path}")
    return maps

# ============================================================
# STEP 2: COLLECT STATES
# ============================================================

def collect_states(model, model_name, env, maps, n_states=150):
    print(f"  Collecting {n_states} states for {model_name}...")
    states = []
    actions = []
    
    maps_to_use = min(len(maps), n_states)
    total_collected = 0
    
    for map_idx, grid in enumerate(maps[:maps_to_use]):
        safe_positions = np.argwhere(grid == 0)
        if len(safe_positions) < 2:
            continue
        
        start_idx = np.random.choice(len(safe_positions))
        goal_idx = np.random.choice(len(safe_positions))
        while goal_idx == start_idx:
            goal_idx = np.random.choice(len(safe_positions))
        
        start = tuple(safe_positions[start_idx])
        goal = tuple(safe_positions[goal_idx])
        
        obs, info = env.reset(options={
            "fixed_grid": grid,
            "fixed_start": start,
            "fixed_goal": goal
        })
        
        done = False
        steps = 0
        
        while not done and steps < 200 and total_collected < n_states:
            states.append(obs.copy())
            total_collected += 1
            
            action, _ = model.predict(obs, deterministic=False)
            action = int(action)
            actions.append(action)
            
            obs, reward, done, truncated, info = env.step(action)
            if truncated:
                done = True
            steps += 1
        
        if total_collected >= n_states:
            break
        
        if (map_idx + 1) % 10 == 0:
            print(f"  Processed {map_idx + 1}/{min(len(maps), maps_to_use)} maps")
    
    print(f"  {model_name}: collected {len(states)} states")
    return states, actions

# ============================================================
# STEP 3: FLATTEN OBSERVATION
# ============================================================

def flatten_obs(obs):
    terrain_patch = obs['terrain_patch'].squeeze()
    terrain_flat = terrain_patch.flatten()
    direction = obs['direction_to_goal']
    distance = obs['distance_to_goal']
    return np.concatenate([terrain_flat, direction, distance])

def unflatten_obs(x):
    return {
        'terrain_patch': x[:169].reshape(13, 13, 1).astype(np.float32),
        'direction_to_goal': x[169:171].astype(np.float32),
        'distance_to_goal': np.array([x[171]], dtype=np.float32)
    }

# ============================================================
# STEP 4: PREDICTION FUNCTION (RETURNS PROBABILITIES FOR SHAP)
# ============================================================

def make_prediction_function(model, model_name):
    def predict_fn(x):
        batch_size = x.shape[0]
        predictions = []
        for i in range(batch_size):
            obs_dict = unflatten_obs(x[i])
            try:
                if 'DQN' in model_name:
                    # For DQN, return Q-values
                    q_values = model.q_net(obs_dict)
                    predictions.append(q_values.detach().cpu().numpy().flatten())
                else:
                    # For PPO, return action probabilities
                    dist = model.policy.get_distribution(obs_dict)
                    probs = dist.distribution.probs.detach().cpu().numpy().flatten()
                    predictions.append(probs)
            except Exception as e:
                # Fallback: one-hot encode the action
                action, _ = model.predict(obs_dict, deterministic=False)
                probs = np.zeros(8)
                probs[action] = 1.0
                predictions.append(probs)
        return np.array(predictions)
    return predict_fn

# ============================================================
# STEP 5: RUN SHAP ANALYSIS (CORRECTLY HANDLING SHAP OUTPUT)
# ============================================================

def run_shap_analysis(model, model_name, states, output_dir):
    print(f"\n--- Running SHAP for {model_name} ---")
    
    flat_states = np.array([flatten_obs(s) for s in states])
    background = flat_states[:min(BACKGROUND_SIZE, len(flat_states))]
    predict_fn = make_prediction_function(model, model_name)
    
    print("  Initializing KernelExplainer...")
    explainer = shap.KernelExplainer(predict_fn, background)
    
    print(f"  Computing SHAP values for {min(len(flat_states), 100)} states...")
    n_to_explain = min(len(flat_states), 100)
    shap_values = explainer.shap_values(flat_states[:n_to_explain], nsamples=N_SAMPLES)
    
    # --- PROPERLY HANDLE SHAP OUTPUT ---
    # shap_values is a list of arrays: one per output (action)
    # Each array has shape (n_samples, n_features)
    # We want to average across outputs to get (n_samples, n_features)
    if isinstance(shap_values, list):
        print(f"  SHAP is a list of {len(shap_values)} arrays")
        # Take mean absolute across all outputs
        shap_avg = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        print(f"  After averaging: shap_avg shape = {shap_avg.shape}")
    else:
        shap_avg = np.abs(shap_values)
        print(f"  shap_avg shape = {shap_avg.shape}")
    
    # Average across samples to get feature importance
    mean_shap = np.mean(shap_avg, axis=0)
    print(f"  mean_shap shape = {mean_shap.shape}")
    
    # The number of features should be 172 (169 terrain + 3 goal features)
    n_features = mean_shap.shape[0]
    print(f"  Number of features: {n_features}")
    
    # If we have more features than expected, take only the first 172
    if n_features > 172:
        print(f"  Warning: Truncating from {n_features} to 172 features")
        mean_shap = mean_shap[:172]
        n_features = 172
    
    feature_names = [f'Cell_{i}' for i in range(169)] + ['Dir_X', 'Dir_Y', 'Dist']
    
    # Get top 20 indices
    top_indices = np.argsort(mean_shap)[-20:][::-1].flatten().tolist()
    top_values = mean_shap[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    # --- Plot 1: Top 20 Features ---
    print("  Generating Top 20 Features plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_names, top_values, color='steelblue')
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title(f'{model_name}: Top 20 Most Influential Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_top_features.png'), dpi=300)
    plt.close()
    
    # --- Plot 2: Heatmap ---
    print("  Generating Spatial Heatmap...")
    shap_grid = mean_shap[:169].reshape(13, 13)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(shap_grid, cmap='RdBu_r', interpolation='bilinear')
    ax.set_title(f'{model_name}: Spatial Attention Map')
    ax.set_xlabel('Horizontal Position')
    ax.set_ylabel('Vertical Position')
    ax.scatter(6, 6, color='black', marker='o', s=50, label='Rover')
    plt.colorbar(im, label='Mean |SHAP Value|')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_heatmap.png'), dpi=300)
    plt.close()
    
    # --- Plot 3: SHAP Summary ---
    print("  Generating SHAP Summary plot...")
    try:
        top_20_indices = np.argsort(mean_shap)[-20:][::-1].flatten().tolist()
        shap_top20 = shap_avg[:, top_20_indices]
        feature_names_top20 = [feature_names[i] for i in top_20_indices]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        shap.summary_plot(shap_top20, flat_states[:n_to_explain, :][:, top_20_indices],
                          feature_names=feature_names_top20, show=False, ax=ax)
        ax.set_title(f'{model_name}: SHAP Summary Plot')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_summary_plot.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"  Warning: Could not generate summary plot: {e}")
    
    print(f"  Saved SHAP plots for {model_name}")
    return shap_avg, shap_values

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("MarsPath SHAP Explainability Analysis (Step 7)")
    print("=" * 60)
    
    print(f"\nLoading maps from: {REAL_MAPS_PATH}")
    maps = load_real_maps(REAL_MAPS_PATH)
    
    if len(maps) == 0:
        print("ERROR: No maps found.")
        return
    
    print("\nInitializing environment...")
    env = MarsTerrainEnv(difficulty="medium", max_steps=200)
    
    for model_name in ANALYZE_MODELS:
        model_path = MODEL_PATHS.get(model_name)
        if model_path is None:
            continue
            
        print(f"\n{'='*50}")
        print(f"Processing: {model_name}")
        print(f"Model path: {model_path}")
        print(f"{'='*50}")
        
        if not os.path.exists(model_path):
            print(f"ERROR: Model not found at {model_path}")
            continue
        
        try:
            if 'DQN' in model_name:
                print("  Loading DQN model...")
                model = DQN.load(str(model_path))
            else:
                print("  Loading PPO model...")
                model = PPO.load(str(model_path))
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            continue
        
        states, actions = collect_states(model, model_name, env, maps, n_states=N_STATES)
        
        if len(states) < 10:
            print(f"ERROR: Only {len(states)} states collected. Skipping.")
            continue
        
        shap_avg, shap_values = run_shap_analysis(model, model_name, states, OUTPUT_DIR)
        
        shap_data = {
            'model_name': model_name,
            'n_states': len(states),
            'states': states,
            'actions': actions,
            'shap_avg': shap_avg,
            'shap_values': shap_values
        }
        
        with open(os.path.join(OUTPUT_DIR, f'{model_name}_shap_data.pkl'), 'wb') as f:
            pickle.dump(shap_data, f)
        print(f"  Saved SHAP data to: {OUTPUT_DIR}/{model_name}_shap_data.pkl")
    
    print("\n" + "=" * 60)
    print("SHAP Analysis Complete!")
    print(f"Figures saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()