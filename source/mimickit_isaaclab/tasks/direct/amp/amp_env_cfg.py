"""Configuration for AMP (Adversarial Motion Priors) environment."""

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab.utils import configclass

from ..deepmimic.deepmimic_env_cfg import DeepMimicEnvCfg, DeepMimicSMPLEnvCfg


@configclass
class AMPEnvCfg(DeepMimicEnvCfg):
    """Configuration for AMP environment.
    
    AMP extends DeepMimic with a discriminator that learns to distinguish
    between real and generated motions.
    """
    
    # AMP-specific parameters
    num_disc_obs_steps: int = 10
    """Number of discriminator observation steps."""


@configclass
class AMPSMPLEnvCfg(DeepMimicSMPLEnvCfg):
    """Configuration for AMP environment with SMPL character."""
    
    # AMP-specific parameters  
    num_disc_obs_steps: int = 10
    """Number of discriminator observation steps."""


# Alias for backward compatibility
AMPHumanoidEnvCfg = AMPSMPLEnvCfg

