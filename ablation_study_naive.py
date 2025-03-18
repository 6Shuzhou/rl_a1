# ablation_study.py
import gym
import numpy as np
import matplotlib.pyplot as plt
from agent import QLearningAgent
import hashlib
import os
import json
import pandas as pd
from tqdm import tqdm
import torch

# Fixed random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Experiment configuration
BASE_CONFIG = {
    "total_steps": 1_000_000,
    "max_steps_per_episode": 500,
    "eval_interval": 10_000,
    "default_params": {
        "learning_rate": 1e-3,
        "update_interval": 1,
        "hidden_dim": 128,
        "epsilon_decay": 0.999,
    },
    "ablation_params": {
        "learning_rate": [1e-4, 1e-3, 1e-2],
        "update_interval": [1, 4, 16],
        "hidden_dim": [64, 128, 256],
        "epsilon_decay": [0.99, 0.999, 0.9999]
    }
}

def train(config, params):
    """Training function"""
    env = gym.make("CartPole-v1")
    
    agent = QLearningAgent(
        env,
        use_target_net=False,
        use_replay_buffer=False,
        hidden_dim=params["hidden_dim"],
        lr=params["learning_rate"],
        gamma=0.99,
        batch_size=128,
        buffer_size=10_000,
        tau=0.01,
        epsilon_decay=params["epsilon_decay"],
        epsilon_min=0.01
    )
    
    # Initialize records container
    eval_records = []
    step_count = 0
    episode = 0
    
    # Training loop
    with tqdm(total=config["total_steps"], desc=generate_exp_id(params)) as pbar:
        while step_count < config["total_steps"]:
            state, _ = env.reset()
            terminated, truncated = False, False
            episode_steps = 0

            while True:
                if terminated or truncated or episode_steps >= config["max_steps_per_episode"]:
                    break
                
                # Perform environment interaction
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Update network at intervals
                if step_count % params["update_interval"] == 0:
                    agent.train_step(state, action, reward, next_state, done)
                
                state = next_state
                step_count += 1
                episode_steps += 1
                
                # Regular evaluation
                if step_count % config["eval_interval"] == 0:
                    eval_reward = evaluate_policy(agent, env)
                    eval_records.append({
                        "step": step_count,
                        "eval_reward": eval_reward,
                        "epsilon": agent.epsilon
                    })
                
                pbar.update(1)
                if step_count >= config["total_steps"]:
                    break
            
            agent.update_epsilon()
            episode += 1
    
    env.close()
    return {
        "eval_records": eval_records,
        "params": params,
        "final_epsilon": agent.epsilon
    }


def evaluate_policy(agent, env, n_episodes=5):
    """Policy evaluation function"""
    total_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            total_reward += reward
            
        total_rewards.append(total_reward)
    return np.mean(total_rewards)

def generate_exp_id(params):
    """Generate unique experiment ID"""
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.sha256(param_str.encode()).hexdigest()[:12]

def save_results(results):
    """Save results"""
    os.makedirs("ablation_results", exist_ok=True)
    exp_id = generate_exp_id(results["params"])
    
    # Save evaluation records
    df = pd.DataFrame(results["eval_records"])
    df.to_parquet(f"ablation_results/{exp_id}.parquet")
    
    # Save metadata
    with open(f"ablation_results/{exp_id}_meta.json", "w") as f:
        json.dump({
            "params": results["params"],
            "final_epsilon": results["final_epsilon"]
        }, f)

def load_results(exp_id):
    """Load results"""
    try:
        df = pd.read_parquet(f"ablation_results/{exp_id}.parquet")
        with open(f"ablation_results/{exp_id}_meta.json", "r") as f:
            meta = json.load(f)
        return {
            "eval_records": df.to_dict("records"),
            **meta
        }
    except FileNotFoundError:
        return None

def ablation_study():
    """Main ablation study function"""
    os.makedirs("ablation_plots", exist_ok=True)
    
    param_order = ["learning_rate", "update_interval", "hidden_dim", "epsilon_decay"]
    best_params = BASE_CONFIG["default_params"].copy()
    
    for param_name in param_order:
        print(f"\n=== Analyzing parameter: {param_name} ===")
        experiments = []
        
        # Generate parameter combinations
        for value in BASE_CONFIG["ablation_params"][param_name]:
            params = best_params.copy()
            params[param_name] = value
            exp_id = generate_exp_id(params)
            
            # Check cache
            cached = load_results(exp_id)
            if cached:
                print(f"Using cached result: {exp_id}")
                experiments.append((params, cached))
                continue
                
            # Run new experiment
            print(f"Starting experiment: {param_name}={value}")
            results = train(BASE_CONFIG, params)
            save_results(results)
            experiments.append((params, results))
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        for params, result in experiments:
            value = params[param_name]
            steps = [r["step"] for r in result["eval_records"]]
            rewards = [r["eval_reward"] for r in result["eval_records"]]
            plt.plot(steps, rewards, label=f"{param_name}={value}")
        
        plt.xlabel("Environment Steps")
        plt.ylabel("Evaluation Reward (5-episode avg)")
        plt.title(f"Parameter Ablation Study: {param_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"ablation_plots/{param_name}.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Select best parameter
        best_value = analyze_results(experiments, param_name)
        best_params[param_name] = best_value
        print(f"Best parameter selected: {param_name} = {best_value}")

def analyze_results(experiments, param_name):
    """Analyze results to select best parameter"""
    best_score = -np.inf
    best_value = None
    
    for params, result in experiments:
        # Take average of last 5 evaluations
        last_rewards = [r["eval_reward"] for r in result["eval_records"][-5:]]
        score = np.mean(last_rewards)
        
        if score > best_score:
            best_score = score
            best_value = params[param_name]
    
    return best_value

if __name__ == "__main__":
    ablation_study()