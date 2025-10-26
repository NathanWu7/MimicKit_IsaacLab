"""Configuration for ASE (Adversarial Skill Embeddings) environment."""

from __future__ import annotations

from isaaclab.utils import configclass

from ..amp.amp_env_cfg import AMPEnvCfg, AMPSMPLEnvCfg


@configclass
class ASEEnvCfg(AMPEnvCfg):
    """Configuration for ASE environment.
    
    ASE extends AMP with latent skill embeddings that allow for more diverse
    and controllable motion generation.
    """
    
    # ASE-specific parameters
    default_reset_prob: float = 0.0
    """Probability of resetting to default pose instead of motion pose."""


@configclass
class ASESMPLEnvCfg(AMPSMPLEnvCfg):
    """Configuration for ASE environment with SMPL character."""
    
    # ASE-specific parameters
    default_reset_prob: float = 0.0
    """Probability of resetting to default pose instead of motion pose."""


# Alias for backward compatibility
ASEHumanoidEnvCfg = ASESMPLEnvCfg

