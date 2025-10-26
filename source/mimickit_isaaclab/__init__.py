# Copyright (c) 2025, MimicKit Developers.
# All rights reserved.
#
# Register MimicKit environments with Isaac Lab.

"""Register MimicKit environments."""

import gymnasium as gym

from . import tasks
from .tasks.direct.deepmimic import (
    DeepMimicEnv,
    DeepMimicEnvCfg,
    DeepMimicHumanoidEnvCfg,
    DeepMimicSMPLEnvCfg,
    DeepMimicG1EnvCfg,
)
from .tasks.direct.amp import (
    AMPEnv,
    AMPEnvCfg,
    AMPSMPLEnvCfg,
    AMPHumanoidEnvCfg,
    AMPG1Env,
    AMPG1EnvCfg,
)
from .tasks.direct.ase import (
    ASEEnv,
    ASEEnvCfg,
    ASESMPLEnvCfg,
    ASEHumanoidEnvCfg,
    ASEG1Env,
    ASEG1EnvCfg,
)

##
# Register Gym environments.
##

# DeepMimic - Base environment
gym.register(
    id="MimicKit-DeepMimic-Direct-v0",
    entry_point="mimickit_isaaclab.tasks.direct.deepmimic:DeepMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DeepMimicEnvCfg,
        "rsl_rl_cfg_entry_point": f"{tasks.__name__}.direct.deepmimic.agents:rsl_rl_ppo_cfg",
    },
)

# DeepMimic - Humanoid
gym.register(
    id="MimicKit-DeepMimic-Humanoid-Direct-v0",
    entry_point="mimickit_isaaclab.tasks.direct.deepmimic:DeepMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DeepMimicHumanoidEnvCfg,
        "rsl_rl_cfg_entry_point": f"{tasks.__name__}.direct.deepmimic.agents:rsl_rl_ppo_cfg",
    },
)

# DeepMimic - SMPL
gym.register(
    id="MimicKit-DeepMimic-SMPL-Direct-v0",
    entry_point="mimickit_isaaclab.tasks.direct.deepmimic:DeepMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DeepMimicSMPLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{tasks.__name__}.direct.deepmimic.agents:rsl_rl_ppo_cfg",
    },
)

# DeepMimic - G1
gym.register(
    id="MimicKit-DeepMimic-G1-Direct-v0",
    entry_point="mimickit_isaaclab.tasks.direct.deepmimic:DeepMimicG1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DeepMimicG1EnvCfg,
        "rsl_rl_cfg_entry_point": f"{tasks.__name__}.direct.deepmimic.agents:rsl_rl_ppo_cfg",
    },
)

# AMP - SMPL
gym.register(
    id="MimicKit-AMP-SMPL-Direct-v0",
    entry_point="mimickit_isaaclab.tasks.direct.amp:AMPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AMPSMPLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{tasks.__name__}.direct.amp.agents:rsl_rl_amp_ppo_cfg",
    },
)

# AMP - Humanoid (alias)
gym.register(
    id="MimicKit-AMP-Humanoid-Direct-v0",
    entry_point="mimickit_isaaclab.tasks.direct.amp:AMPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AMPHumanoidEnvCfg,
        "rsl_rl_cfg_entry_point": f"{tasks.__name__}.direct.amp.agents:rsl_rl_amp_ppo_cfg",
    },
)

# ASE - SMPL
gym.register(
    id="MimicKit-ASE-SMPL-Direct-v0",
    entry_point="mimickit_isaaclab.tasks.direct.ase:ASEEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ASESMPLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{tasks.__name__}.direct.ase.agents:rsl_rl_ase_ppo_cfg",
    },
)

# ASE - Humanoid (alias)
gym.register(
    id="MimicKit-ASE-Humanoid-Direct-v0",
    entry_point="mimickit_isaaclab.tasks.direct.ase:ASEEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ASEHumanoidEnvCfg,
        "rsl_rl_cfg_entry_point": f"{tasks.__name__}.direct.ase.agents:rsl_rl_ase_ppo_cfg",
    },
)

# AMP - G1
gym.register(
    id="MimicKit-AMP-G1-Direct-v0",
    entry_point="mimickit_isaaclab.tasks.direct.amp:AMPG1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AMPG1EnvCfg,
        "rsl_rl_cfg_entry_point": f"{tasks.__name__}.direct.amp.agents:rsl_rl_amp_g1_ppo_cfg",
    },
)

# ASE - G1
gym.register(
    id="MimicKit-ASE-G1-Direct-v0",
    entry_point="mimickit_isaaclab.tasks.direct.ase:ASEG1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ASEG1EnvCfg,
        "rsl_rl_cfg_entry_point": f"{tasks.__name__}.direct.ase.agents:rsl_rl_ase_g1_ppo_cfg",
    },
)

