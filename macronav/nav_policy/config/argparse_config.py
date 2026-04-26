import argparse


def parse_train_args():
    parser = argparse.ArgumentParser(description="Parse configuration for RL navigation model")

    parser.add_argument("--use_slurm", action="store_true", help="Run on SLURM")
    # Logging parameters
    parser.add_argument("--exp_name", type=str, help="Experiment name", required=True)
    parser.add_argument("--log_dir", type=str, default=".", help="Directory for logs")
    parser.add_argument("--summary_window", type=int, default=50, help="Summary window size")
    parser.add_argument("--img_save_freq", type=int, default=100, help="Save image every N episodes")
    parser.add_argument("--model_save_freq", type=int, default=100, help="Save model every N episodes")

    # Training parameters
    parser.add_argument("--ray_local_mode", action="store_true", help="Run in local mode for Ray")
    parser.add_argument("--load_model", action="store_true", help="Load model checkpoint")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--max_episode", type=int, default=20000, help="Maximum number of episodes")
    parser.add_argument("--replay_buffer_size", type=int, default=30000, help="Size of the replay buffer")
    parser.add_argument(
        "--min_replay_buffer_size", type=int, default=6000, help="Minimum replay buffer size before training"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size for training")
    parser.add_argument(
        "--replay_sequence_length",
        type=int,
        default=1,
        help="Contiguous sequence length sampled from replay when recurrent models are enabled",
    )
    parser.add_argument("--lr_policy_net", type=float, default=1e-5, help="Learning rate for policy network")
    parser.add_argument("--lr_q_net", type=float, default=2e-5, help="Learning rate for Q network")
    parser.add_argument("--lr_alpha", type=float, default=1e-4, help="Learning rate for alpha")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--decay_step", type=int, default=256, help="Decay step (unused)")
    parser.add_argument("--episode_max_step", type=int, default=128, help="Maximum steps per episode")
    parser.add_argument("--entropy_weight", type=float, default=0.01, help="Entropy weight")
    parser.add_argument("--target_q_net_update_freq", type=int, default=64, help="Target Q network update frequency")
    parser.add_argument("--train_times_per_episode", type=int, default=8, help="Train times per episode")
    parser.add_argument("--num_gpu", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--num_cpu", type=int, default=10, help="Number of CPUs")
    parser.add_argument("--normalize_utility", action="store_true", help="Normalize utility")
    parser.add_argument("--grad_accumu_step", type=int, default=1, help="Gradient accumulation step")

    # Env parameters
    # Node encoding parameters
    parser.add_argument("--k_size", type=int, default=20, help="Number of neighboring nodes")

    # Environment encoding parameters
    parser.add_argument(
        "--env_encoding_model_ckpt", type=str, default=None, help="Environment encoding model checkpoint"
    )

    # Environment parameters
    parser.add_argument("--env_level", type=str, default="origin", help="Environment level")
    parser.add_argument("--env_random_map", action="store_true", help="Use random map")
    parser.add_argument("--env_random_level", action="store_true", help="Use random level")
    parser.add_argument(
        "--num_agent",
        "--num_meta_agent",
        dest="num_agent",
        type=int,
        default=10,
        help="Number of parallel agents",
    )
    parser.add_argument("--sensor_range", type=int, default=80, help="Sensor range in pixels")
    parser.add_argument("--reward_w_astar", type=int, default=1, help="A* reward coefficient")
    parser.add_argument("--reward_w_step", type=int, default=1, help="Step reward coefficient")

    args, _ = parser.parse_known_args()
    return args


def parse_test_args():
    parser = argparse.ArgumentParser(description="RL Navigation Configuration")

    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--log_dir", type=str, required=True, help="Log directory")
    parser.add_argument("--ray_local_mode", action="store_true", help="Run in local mode for Ray")
    parser.add_argument(
        "--num_agent",
        "--num_meta_agent",
        dest="num_agent",
        type=int,
        default=10,
        help="Number of parallel agents",
    )
    parser.add_argument("--num_episode", type=int, default=200, help="Number of episodes")
    parser.add_argument("--num_run", type=int, default=1, help="Number of runs")
    parser.add_argument("--save_gifs", action="store_true", help="Save GIFs of training")
    parser.add_argument("--save_traj", action="store_true", help="Save trajectories")
    parser.add_argument("--episode_max_step", type=int, default=128, help="Maximum steps per episode")
    parser.add_argument("--load_param_from_json", action="store_true", help="Load parameters from JSON")
    parser.add_argument("--env_level", type=str, default="easy", help="Environment level")
    parser.add_argument("--sensor_range", type=int, default=None, help="Sensor range in pixels")
    parser.add_argument("--num_gpu", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--num_cpu", type=int, default=10, help="Number of CPUs")

    args, _ = parser.parse_known_args()
    return args
