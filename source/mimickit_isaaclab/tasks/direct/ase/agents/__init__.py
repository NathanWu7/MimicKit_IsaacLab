"""Agent configuration for ASE."""

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# Import G1-specific configuration
from .rsl_rl_ase_g1_ppo_cfg import rsl_rl_ase_g1_ppo_cfg

# ASE PPO configuration (SMPL)
rsl_rl_ase_ppo_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=24,
    max_iterations=50000,
    save_interval=500,
    experiment_name="ase_smpl",
    run_name="",
    logger="tensorboard",
    resume=False,
    load_run="",
    load_checkpoint="",
    seed=42,
    device="cuda:0",
    clip_actions=1.0,  # Clip range for actions (not boolean!)
    policy=RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,  # Must be > 0
        actor_hidden_dims=[1024, 512],
        critic_hidden_dims=[1024, 512],
        activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
)

