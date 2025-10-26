"""ASE (Adversarial Skill Embeddings) environment implementation.

ASE extends AMP with latent skill embeddings for more diverse motion generation.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from ..amp.amp_env import AMPEnv

if TYPE_CHECKING:
    from .ase_env_cfg import ASEEnvCfg


class ASEEnv(AMPEnv):
    """ASE environment for learning with adversarial skill embeddings.
    
    This environment extends AMP by occasionally resetting to default poses
    instead of motion library poses, allowing for more exploration.
    """

    cfg: ASEEnvCfg

    def __init__(self, cfg: ASEEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize ASE environment.
        
        Args:
            cfg: Configuration for ASE environment
            render_mode: Render mode for the environment
            **kwargs: Additional keyword arguments
        """
        self._default_reset_prob = cfg.default_reset_prob
        super().__init__(cfg, render_mode, **kwargs)

    def _reset_char(self, env_ids: torch.Tensor):
        """Reset character state for specific environments.
        
        With probability default_reset_prob, reset to default pose instead of motion pose.
        
        Args:
            env_ids: Environment indices to reset
        """
        super()._reset_char(env_ids)
        
        n = len(env_ids)
        if n > 0 and self._default_reset_prob > 0:
            # Randomly select environments to reset to default pose
            rand_val = torch.rand(n, device=self.device)
            mask = rand_val < self._default_reset_prob
            default_reset_ids = env_ids[mask]
            
            if len(default_reset_ids) > 0:
                # Reset to default pose (from base CharEnv)
                self._reset_char_to_default(default_reset_ids)

    def _reset_char_to_default(self, env_ids: torch.Tensor):
        """Reset character to default pose.
        
        Args:
            env_ids: Environment indices to reset
        """
        # Set to default root state
        default_root_state = self.character.data.default_root_state[env_ids]
        self.character.write_root_state_to_sim(default_root_state, env_ids)
        
        # Set to default joint positions
        default_joint_pos = self.character.data.default_joint_pos[env_ids]
        default_joint_vel = self.character.data.default_joint_vel[env_ids]
        self.character.write_joint_state_to_sim(
            default_joint_pos, default_joint_vel, None, env_ids
        )

