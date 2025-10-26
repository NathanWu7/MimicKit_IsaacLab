# Copyright (c) 2025, MimicKit Developers.
# All rights reserved.

"""RSL-RL PPO agent configuration for DeepMimic."""

from __future__ import annotations

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# Create RSL-RL PPO configuration
rsl_rl_ppo_cfg = RslRlOnPolicyRunnerCfg(
    seed=42,
    device="cuda:0",
    num_steps_per_env=24,
    max_iterations=10000,
    save_interval=500,
    experiment_name="deepmimic_humanoid",
    run_name="",
    clip_actions=1.0,  # Clip range for actions (not boolean!)
    # Resume training
    resume=False,
    load_run=".*",
    load_checkpoint="model_.*.pt",
    # Logging
    logger="tensorboard",
    # Observation groups
    obs_groups={
        "policy": ["policy"],
    },
    # Policy
    policy=RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,  # Initial exploration noise (higher = more exploration)
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    ),
    # Algorithm
    algorithm=RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # Increase entropy coefficient to maintain exploration
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
)

