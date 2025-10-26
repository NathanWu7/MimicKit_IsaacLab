"""Configuration for ASE environment with G1 robot."""

from isaaclab.utils import configclass

from ..amp.amp_g1_env_cfg import AMPG1EnvCfg


@configclass
class ASEG1EnvCfg(AMPG1EnvCfg):
    """Configuration for ASE environment with G1 robot.
    
    ASE (Adversarial Skill Embeddings) extends AMP with latent skill embeddings
    that allow for more diverse and controllable motion generation.
    """
    
    # ASE-specific parameters
    default_reset_prob: float = 0.0
    """Probability of resetting to default pose instead of motion pose."""
    
    def __post_init__(self):
        # Call parent post_init to initialize AMP and G1-specific settings
        super().__post_init__()
        
        # ASE is even more exploratory, so use very relaxed termination
        self.pose_termination_dist = 4.0  # Very generous for skill discovery

