#!/usr/bin/env python3
# Copyright (c) 2025, MimicKit Developers.
# All rights reserved.
#
# Compare script for MimicKit with Isaac Lab - Policy + Reference motion mode.

"""Compare trained policy with reference motion side by side."""

import argparse
import os
import sys

# Import Isaac Lab app launcher first
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Compare policy with reference motion (both visible).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--use_last_checkpoint", action="store_true", help="Use the last checkpoint.")
parser.add_argument("--load_run", type=str, default=None, help="Name of the run folder to load from.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

# Add mimickit_isaaclab to Python path
_mimickit_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "source")
if _mimickit_path not in sys.path:
    sys.path.insert(0, _mimickit_path)

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from isaaclab.envs import DirectRLEnvCfg

# Import MimicKit tasks (must be after Isaac Sim is launched)
import mimickit_isaaclab.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Compare policy with reference motion (both visible side by side)."""
    # Override number of environments
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Enable reference character visualization for comparison
    env_cfg.visualize_ref_char = True
    
    # Disable early termination for continuous playback
    env_cfg.enable_early_termination = False
    env_cfg.pose_termination = False
    
    print("[INFO] Mode: COMPARE (Policy + Reference motion side by side)")
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="human")
    
    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    # Get checkpoint path
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        load_run = args_cli.load_run if args_cli.load_run else agent_cfg.load_run
        resume_path = get_checkpoint_path(log_root_path, load_run, agent_cfg.load_checkpoint)
    
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # Create a dummy runner to load the policy
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    
    # Get policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Reset environment
    obs = env.get_observations()
    
    print("[INFO] Comparing policy-controlled character (left) with reference motion (right)...")
    
    # Simulate
    while simulation_app.is_running():
        # Run everything in inference mode
        with torch.inference_mode():
            # Compute actions
            actions = policy(obs)
            # Step environment
            step_result = env.step(actions)
            obs = step_result[0]  # Compatible with both 4 and 5 return values

    # Close the simulator
    env.close()


if __name__ == "__main__":
    # Run empty system initialization
    simulation_app.update()
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()

