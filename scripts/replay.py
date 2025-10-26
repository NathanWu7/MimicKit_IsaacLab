#!/usr/bin/env python3
# Copyright (c) 2025, MimicKit Developers.
# All rights reserved.
#
# Replay script for MimicKit with Isaac Lab - Motion replay only mode.

"""Replay motion data without policy."""

import argparse
import os
import sys

# Import Isaac Lab app launcher first
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion data (reference motion only, no policy).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to motion file to replay (e.g., data/motions/g1/g1_walk.pkl).")

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

from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import DirectRLEnvCfg

# Import MimicKit tasks (must be after Isaac Sim is launched)
import mimickit_isaaclab.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg):
    """Replay motion data (reference motion only, no policy)."""
    # Override number of environments
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Override motion file if specified
    if args_cli.motion_file:
        env_cfg.motion_file = args_cli.motion_file
        print(f"[INFO] Using motion file: {args_cli.motion_file}")
    else:
        print(f"[INFO] Using default motion file: {env_cfg.motion_file}")
    
    # Enable replay mode (only reference character, no controllable character)
    env_cfg.replay_mode = True
    
    # Enable reference character visualization (will be at center in replay mode)
    env_cfg.visualize_ref_char = True
    
    # Disable early termination for continuous playback
    env_cfg.enable_early_termination = False
    env_cfg.pose_termination = False
    
    print("[INFO] Mode: REPLAY (Reference motion only, no controllable character)")
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="human")
    
    # Reset environment
    env.reset()
    
    print("[INFO] Playing back reference motion at center...")
    print("[INFO] Press Ctrl+C or close window to exit")
    
    # Simulate - just step with zero actions to see reference motion
    zero_actions = torch.zeros((env_cfg.scene.num_envs, env.action_space.shape[0]), 
                                device=env.unwrapped.device)
    
    while simulation_app.is_running():
        # Step with zero actions - only reference motion will be visible
        env.step(zero_actions)
        
    # Close the simulator
    env.close()


if __name__ == "__main__":
    # Run empty system initialization
    simulation_app.update()
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()