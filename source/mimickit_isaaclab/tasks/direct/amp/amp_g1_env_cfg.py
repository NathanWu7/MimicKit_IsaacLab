"""Configuration for AMP environment with G1 robot."""

from isaaclab.utils import configclass

from ..deepmimic.deepmimic_g1_env_cfg import DeepMimicG1EnvCfg


@configclass
class AMPG1EnvCfg(DeepMimicG1EnvCfg):
    """Configuration for AMP environment with G1 robot.
    
    AMP (Adversarial Motion Priors) extends DeepMimic with a discriminator
    that learns to distinguish between real and generated motions.
    """
    
    # Override motion file with G1 default for AMP
    motion_file: str = "data/motions/g1/g1_walk.pkl"
    
    # AMP-specific parameters
    num_disc_obs_steps: int = 10
    """Number of discriminator observation steps."""
    
    def __post_init__(self):
        # Call parent post_init to initialize G1-specific settings
        super().__post_init__()
        
        # AMP training with very relaxed termination to allow learning
        # Enable termination but with generous thresholds
        self.enable_early_termination = True
        self.termination_height = 0.1  # G1 is shorter than SMPL
        
        # Pose termination with very relaxed distance
        # Increased from 3.0 to 5.0 to match successful ASE training behavior
        self.pose_termination = True
        self.pose_termination_dist = 5.0  # Very generous to reduce reset frequency
        
        # AMP reward weights (balanced for discriminator training)
        self.reward_pose_w = 0.4
        self.reward_vel_w = 0.05
        self.reward_root_pose_w = 0.15
        self.reward_root_vel_w = 0.1
        self.reward_key_pos_w = 0.1

