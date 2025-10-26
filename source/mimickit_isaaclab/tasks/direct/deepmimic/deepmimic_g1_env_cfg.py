
# Copyright (c) 2025, MimicKit Developers.
# All rights reserved.
#
# Configuration for DeepMimic G1 environment.

from isaaclab.utils import configclass

from .deepmimic_env_cfg import DeepMimicEnvCfg
from ....assets.Config.robots.g1 import UNITREE_G1_29DOF_CFG


@configclass
class DeepMimicG1EnvCfg(DeepMimicEnvCfg):
    """Configuration for DeepMimic environment with G1 robot."""
    
    # Override motion file with G1 default (must be declared here, not in __post_init__)
    motion_file: str = "data/motions/g1/g1_spinkick.pkl"
    
    # CRITICAL: Override decimation as class variable to take effect
    decimation: int = 2  # 30 Hz control
    
    def __post_init__(self):
        # Call parent post_init to initialize all attributes
        super().__post_init__()
        
        # Ensure decimation is correctly set
        self.decimation = 2
        
        # Update render_interval to match
        self.sim.render_interval = self.decimation
        
        # Override character asset with G1
        self.char_asset = UNITREE_G1_29DOF_CFG.replace(
            prim_path="/World/envs/env_.*/Character"
        )
        
        # Set action and observation spaces as integers (will be converted to Box spaces)
        # G1 has 29 DOF
        self.action_space = 29
        # Observation space will be computed dynamically, use a reasonable estimate
        self.observation_space = 100
        
        # Update control mode for G1 (use PD control)
        self.control_mode = "pd"
        
        # Key bodies for G1 (important body parts for reward computation)
        # Use semantic names that match KinCharModelG1 structure
        self.key_bodies = [
            "pelvis",
            "left_hip_pitch",
            "left_knee",
            "left_ankle_pitch",
            "right_hip_pitch",
            "right_knee",
            "right_ankle_pitch",
            "waist_pitch",
            "left_shoulder_pitch",
            "left_elbow",
            "left_wrist_pitch",
            "right_shoulder_pitch",
            "right_elbow",
            "right_wrist_pitch",
        ]
        
        # Contact bodies for G1 (bodies that can touch the ground)
        self.contact_bodies = [
            "left_ankle_roll",
            "right_ankle_roll",
        ]
        
        # G1-specific reward weights (tuned for G1's proportions)
        self.reward_pose_w = 0.5
        self.reward_vel_w = 0.05
        self.reward_root_pose_w = 0.2
        self.reward_root_vel_w = 0.1
        self.reward_key_pos_w = 0.15
        
        # G1-specific reward scales
        self.reward_pose_scale = 2.0
        self.reward_vel_scale = 0.1
        self.reward_root_pose_scale = 20.0
        self.reward_root_vel_scale = 2.0
        self.reward_key_pos_scale = 5.0
        
        # Termination settings for G1
        # DeepMimic uses relaxed termination to allow learning from mistakes
        self.enable_early_termination = True
        self.termination_height = 0.1  # Lower height threshold (G1 is shorter)
        
        # Pose termination: enabled but very relaxed for DeepMimic
        # DeepMimic focuses on reward-based learning rather than hard termination
        self.pose_termination = True
        self.pose_termination_dist = 5.0  # Very generous for exploration
        
        # Reference character offset (for visualization)
        self.ref_char_offset = [2.0, 0.0, 0.0]  # 2m to the right
        self.ref_char_height_offset = 0.0
        
        # Observation settings
        self.global_obs = True
        self.root_height_obs = True
        self.enable_phase_obs = True
        self.num_phase_encoding = 4
        self.enable_tar_obs = True
        self.tar_obs_steps = [1, 2, 3, 4, 5, 10, 15, 20]  # Future timesteps to observe
        
        # Motion settings
        self.rand_reset = True  # Random initial time in motion
        
        # Action scale (G1 joints have different ranges than SMPL)
        self.action_scale = 0.5
        
        # Joint error weights (for each SMPL joint, used in reward computation)
        # None means all joints have equal weight
        self.joint_err_w = None
        
        # Camera settings - fixed view looking at robot
        self.viewer.eye = (2.8, 1.5, 2.2)
        self.viewer.lookat = (-0.3, -0.3, 0.8)


# Example configurations for different scenarios

@configclass
class DeepMimicG1WalkCfg(DeepMimicG1EnvCfg):
    """Configuration for G1 walking task."""
    
    # Override with specific motion file
    motion_file: str = "data/motions/g1/g1_walk.pkl"
    
    def __post_init__(self):
        super().__post_init__()
        
        # Adjust rewards for walking
        self.reward_root_pose_w = 0.15
        self.reward_root_vel_w = 0.15
        
        # Adjust termination for walking
        self.pose_termination_dist = 0.8


@configclass
class DeepMimicG1RunCfg(DeepMimicG1EnvCfg):
    """Configuration for G1 running task."""
    
    # Override with specific motion file
    motion_file: str = "data/motions/g1/g1_run.pkl"
    
    def __post_init__(self):
        super().__post_init__()
        
        # Adjust rewards for running (more emphasis on velocity)
        self.reward_vel_w = 0.1
        self.reward_root_vel_w = 0.2
        
        # Adjust termination for running (allow more deviation)
        self.pose_termination_dist = 1.0


@configclass
class DeepMimicG1JumpCfg(DeepMimicG1EnvCfg):
    """Configuration for G1 jumping task."""
    
    # Override with specific motion file
    motion_file: str = "data/motions/g1/g1_spinkick.pkl"  # Using spinkick as a placeholder
    
    def __post_init__(self):
        super().__post_init__()
        
        # Adjust rewards for jumping (more emphasis on root pose)
        self.reward_root_pose_w = 0.3
        self.reward_key_pos_w = 0.2
        
        # Adjust termination for jumping
        self.termination_height = 0.2
        self.pose_termination_dist = 1.0


