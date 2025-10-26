"""AMP environment for G1 robot."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .amp_env import AMPEnv
from ..deepmimic.deepmimic_g1_env import DeepMimicG1Env

if TYPE_CHECKING:
    from .amp_g1_env_cfg import AMPG1EnvCfg


class AMPG1Env(AMPEnv, DeepMimicG1Env):
    """AMP environment specifically for G1 robot.
    
    This class combines AMP functionality with G1-specific kinematics.
    Uses multiple inheritance to get both AMP features and G1 DOF mapping.
    """

    cfg: AMPG1EnvCfg

    def __init__(self, cfg: AMPG1EnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize AMP G1 environment.
        
        Args:
            cfg: Configuration for AMP G1 environment
            render_mode: Render mode for the environment
            **kwargs: Additional keyword arguments
        """
        # Call DeepMimicG1Env's __init__ which handles G1-specific setup
        # This will set up the G1 kinematic model correctly
        DeepMimicG1Env.__init__(self, cfg, render_mode, **kwargs)
        
        # Then add AMP-specific initialization
        self._num_disc_obs_steps = cfg.num_disc_obs_steps
        self._build_disc_obs_buffers()

