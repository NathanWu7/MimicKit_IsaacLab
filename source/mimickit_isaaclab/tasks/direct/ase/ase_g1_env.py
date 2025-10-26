"""ASE environment for G1 robot."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .ase_env import ASEEnv
from ..deepmimic.deepmimic_g1_env import DeepMimicG1Env

if TYPE_CHECKING:
    from .ase_g1_env_cfg import ASEG1EnvCfg


class ASEG1Env(ASEEnv, DeepMimicG1Env):
    """ASE environment specifically for G1 robot.
    
    This class combines ASE functionality with G1-specific kinematics.
    Uses multiple inheritance to get both ASE features and G1 DOF mapping.
    """

    cfg: ASEG1EnvCfg

    def __init__(self, cfg: ASEG1EnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize ASE G1 environment.
        
        Args:
            cfg: Configuration for ASE G1 environment
            render_mode: Render mode for the environment
            **kwargs: Additional keyword arguments
        """
        # Call DeepMimicG1Env's __init__ which handles G1-specific setup
        # This will set up the G1 kinematic model correctly
        DeepMimicG1Env.__init__(self, cfg, render_mode, **kwargs)
        
        # Then add ASE-specific initialization (which includes AMP)
        self._num_disc_obs_steps = cfg.num_disc_obs_steps
        self._build_disc_obs_buffers()
        
        # ASE-specific initialization would go here
        # (latent embeddings, skill discovery, etc.)

