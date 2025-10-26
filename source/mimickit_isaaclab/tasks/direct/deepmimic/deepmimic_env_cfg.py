# Copyright (c) 2025, MimicKit Developers.
# All rights reserved.
#
# DeepMimic environment configuration for Isaac Lab.

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# Import robot configurations
from mimickit_isaaclab.assets import HUMANOID_CFG, SMPL_HUMANOID, G1_CFG


@configclass
class DeepMimicEnvCfg(DirectRLEnvCfg):
    """Configuration for the DeepMimic environment."""

    # Environment settings
    episode_length_s = 10.0
    decimation = 2
    action_scale = 1.0
    action_space = MISSING  # Will be set dynamically
    observation_space = MISSING  # Will be set dynamically
    state_space = 0  # Optional state space

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True)

    # Character settings (USD asset configuration)
    char_asset: ArticulationCfg = MISSING  # USD robot configuration
    
    # Motion settings
    motion_file: str = MISSING  # Path to motion file or dataset
    
    # Control mode (can be "pos", "vel", "torque", "pd")
    control_mode: str = "pd"
    
    # Observation settings
    global_obs: bool = False  # Use global or local observations
    root_height_obs: bool = True  # Include root height in observations
    enable_phase_obs: bool = True  # Include motion phase in observations
    num_phase_encoding: int = 2  # Number of positional encodings for phase
    enable_tar_obs: bool = False  # Include target observations
    tar_obs_steps: list[int] = [1]  # Target observation time steps
    
    # Termination settings
    enable_early_termination: bool = True
    termination_height: float = 0.3
    pose_termination: bool = False
    pose_termination_dist: float = 1.0
    
    # Key bodies for tracking (empty list = no key bodies)
    key_bodies: list[str] = []
    contact_bodies: list[str] = []  # Bodies that can contact ground
    
    # Reward weights
    reward_pose_w: float = 0.5
    reward_vel_w: float = 0.05
    reward_root_pose_w: float = 0.15
    reward_root_vel_w: float = 0.1
    reward_key_pos_w: float = 0.2
    
    # Reward scales
    reward_pose_scale: float = 2.0
    reward_vel_scale: float = 0.1
    reward_root_pose_scale: float = 20.0
    reward_root_vel_scale: float = 2.0
    reward_key_pos_scale: float = 2.0
    
    # Joint error weights (optional, will default to 1.0 for all joints)
    joint_err_w: list[float] | None = None
    
    # Reference character visualization
    visualize_ref_char: bool = False  # False for training (better performance); True for visualization/debugging
    ref_char_offset: list[float] = [2.0, 0.0, 0.0]  # X, Y, Z offset (meters)
    ref_char_height_offset: float = 0.0
    
    # Replay mode (only show reference character, hide controllable character)
    replay_mode: bool = False  # True: only reference character visible; False: normal mode  
    
    # Random reset
    rand_reset: bool = True
    
    # Camera settings
    camera_mode: str = "track"  # "track" or "still"
    
    # Initial pose (optional, will default to zero pose)
    init_pose: list[float] | None = None
    
    def __post_init__(self):
        """Post initialization to set viewer camera."""
        super().__post_init__()
        # Camera position: slightly left and farther back
        self.viewer.eye = (6.0, 0.5, 4.0)
        self.viewer.lookat = (-0.4, -0.4, 0.8)


# SMPL humanoid configuration (primary)
@configclass
class DeepMimicSMPLEnvCfg(DeepMimicEnvCfg):
    """Configuration for SMPL humanoid DeepMimic using USD asset."""

    def __post_init__(self):
        super().__post_init__()
        
        # Use USD robot configuration with correct prim_path
        self.char_asset = SMPL_HUMANOID.replace(prim_path="/World/envs/env_.*/Robot")
        
        # Motion file
        self.motion_file = "data/motions/smpl/smpl_walk.pkl"
        
        # Contact bodies
        self.contact_bodies = ["L_Knee", "L_Ankle", "L_Toe", "R_Knee", "R_Ankle", "R_Toe"]
        
        # Key bodies
        self.key_bodies = ["L_Toe", "R_Toe", "Head", "L_Hand", "R_Hand"]
        
        # Observation and action spaces (will be computed dynamically)
        self.observation_space = 300
        self.action_space = 69


# Humanoid configuration (deprecated, use SMPL instead)
@configclass
class DeepMimicHumanoidEnvCfg(DeepMimicSMPLEnvCfg):
    """Configuration for humanoid DeepMimic (uses SMPL, kept as compatibility alias)."""
    pass


# G1 humanoid configuration
@configclass
class DeepMimicG1EnvCfg(DeepMimicEnvCfg):
    """Configuration for Unitree G1 humanoid DeepMimic using USD asset."""

    def __post_init__(self):
        super().__post_init__()
        
        # Use USD robot configuration with correct prim_path
        self.char_asset = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        
        # Motion file
        self.motion_file = "data/motions/g1/g1_walk.pkl"
        
        # Contact bodies
        self.contact_bodies = [
            "left_knee_link",
            "left_ankle_pitch_link",
            "left_ankle_roll_link",
            "right_knee_link",
            "right_ankle_pitch_link",
            "right_ankle_roll_link",
        ]
        
        # Key bodies
        self.key_bodies = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "head_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ]
        
        # Observation and action spaces
        self.observation_space = 400
        self.action_space = 37


