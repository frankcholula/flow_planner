import minari
import torch
from torch.utils.data import DataLoader
from src.conf.environment import BipedalWalkerConfig, CarRacingConfig, LunarLanderConfig
from src.models.backbone import MLP, CNN, ConditionalCNN, ConditionalUNet1D
from src.pipelines.preprocessing import collate_fn, get_dataset_stats

env_config_map = {
    "LunarLander-v3": LunarLanderConfig,
    "CarRacing-v3": CarRacingConfig,
    "BipedalWalker-v3": BipedalWalkerConfig,
}


def build_model(args, obs_dim, action_dim):
    horizon = args.horizon
    # setting input dimension based on what we're modeling
    if args.model_target == "obs_act":
        input_dim = (obs_dim + action_dim) * horizon
    elif args.model_target == "obs_only":
        input_dim = obs_dim * horizon
    elif args.model_target == "act_only":
        input_dim = action_dim * horizon
    else:
        raise ValueError(f"Invalid model_target: {args.model_target}")
    print(f"Modeling {args.model_target} with an input_dim of {input_dim}")

    # setting the right model based on args
    # unconditional models
    if args.model_type == "mlp":
        model = MLP(input_dim=input_dim, time_dim=1, hidden_dim=args.hidden_dim).to(
            args.device
        )
    elif args.model_type == "cnn":
        model = CNN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            horizon=horizon,
            kernel_size=args.kernel_size,
        ).to(args.device)

    # conditional models
    elif args.model_type in ["ccnn", "unet"]:
        print(f"Using {args.model_type.upper()} for training...")
        cond_dim = 0
        if args.condition_on:
            if args.condition_on == "reward":
                cond_dim = 1
            elif args.condition_on == "start_obs":
                cond_dim = obs_dim
            elif args.condition_on in ["start_obs_goal", "start_obs_waypoint"]:
                cond_dim = obs_dim * 2
            else:
                raise ValueError(f"Invalid condition_on type: {args.condition_on}")
        else:
            print(f"Running unconditional {args.model_type.upper()} model...")
            cond_dim = 1  # Placeholder for unconditional model init

        if args.model_type == "ccnn":
            model = ConditionalCNN(
                input_dim=input_dim,
                horizon=horizon,
                hidden_dim=args.hidden_dim,
                kernel_size=args.kernel_size,
                cond_dim=cond_dim,
            ).to(args.device)
        elif args.model_type == "unet":
            model = ConditionalUNet1D(
                input_dim=input_dim,
                horizon=horizon,
                hidden_dim=args.hidden_dim,
                cond_dim=cond_dim,
            ).to(args.device)
    else:
        raise ValueError(f"Invalid model_type: {args.model_type}")
    return model, input_dim


def build_dataloader(dataset, args):
    generator = torch.Generator(device=args.device) if args.device == "cuda" else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=generator,
    )
    stats = get_dataset_stats(dataset)
    return dataloader, stats


def load_dataset(args, env_render_mode="rgb_array", eval_env=False):
    if args.environment not in env_config_map:
        raise ValueError(f"Unknown environment: {args.environment}")

    dataset_config = env_config_map[args.environment]()
    dataset = minari.load_dataset(dataset_id=dataset_config.dataset_name)
    recovered_env = dataset.recover_environment(
        eval_env=eval_env, render_mode=env_render_mode
    )
    return dataset_config, dataset, recovered_env
