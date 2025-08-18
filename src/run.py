import os
import time
import random
import pprint
import minari
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader

from src.models.backbone import MLP, CNN, ConditionalCNN, ConditionalUNet1D
from src.utils.args import parse_fm_args
from src.utils.loggers import WandBLogger

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath

from src.conf.environment import BipedalWalkerConfig, CarRacingConfig, LunarLanderConfig
from src.pipelines.preprocessing import (
    collate_fn,
    get_dataset_stats,
    create_normalized_chunks,
)
from src.pipelines.eval import evaluate_open_loop

env_config_map = {
    "LunarLander-v3": LunarLanderConfig,
    "CarRacing-v3": CarRacingConfig,
    "BipedalWalker-v3": BipedalWalkerConfig,
}


def load_dataset(args):
    """Load dataset and environment based on the selected environment."""
    if args.environment not in env_config_map:
        raise ValueError(f"Unknown environment: {args.environment}")

    config = env_config_map[args.environment]()
    dataset = minari.load_dataset(dataset_id=config.dataset_name)
    env = dataset.recover_environment()
    return config, dataset, env


def build_model(args, obs_dim, action_dim):
    horizon = args.horizon
    if args.model_target == "obs_act":
        input_dim = (obs_dim + action_dim) * horizon
    elif args.model_target == "obs_only":
        input_dim = obs_dim * horizon
    elif args.model_target == "act_only":
        input_dim = action_dim * horizon
    else:
        raise ValueError(f"Invalid model_target: {args.model_target}")
    print(f"Input dimension for the model: {input_dim}")

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
    elif args.model_type == "ccnn":
        if args.condition_on == "reward":
            cond_dim = 1
        elif args.condition_on == "start_obs":
            cond_dim = obs_dim
        elif args.condition_on == "start_obs_goal":
            cond_dim = obs_dim * 2
        else:
            raise ValueError(
                f"ConditionalCNN requires a valid --condition-on argument ('reward', 'start_obs', 'start_obs_goal'). Got: {args.condition_on!r}"
            )
        model = ConditionalCNN(
            input_dim=input_dim,
            horizon=horizon,
            hidden_dim=args.hidden_dim,
            kernel_size=args.kernel_size,
            cond_dim=cond_dim,
        ).to(args.device)
    elif args.model_type == "unet":
        print("Using UNet1D model for training.")
        if args.condition_on == "reward":
            cond_dim = 1
        elif args.condition_on == "start_obs":
            cond_dim = obs_dim
        elif args.condition_on == "start_obs_goal":
            cond_dim = obs_dim * 2
        else:
            raise ValueError(
                f"UNet1D requires a valid --condition-on argument ('reward', 'start_obs', 'start_obs_goal'), but got: {args.condition_on!r}"
            )
        model = ConditionalUNet1D(
            input_dim=input_dim,
            horizon=horizon,
            hidden_dim=args.hidden_dim,
            cond_dim=cond_dim,
            fusion_strategy="concat",
            use_mlp_embedding=False,
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


def run_epoch(model, dataloader, path, optim, args, stats):
    total_loss = 0.0
    total_chunks = 0
    for batch in dataloader:
        optim.zero_grad()

        if args.condition_on:
            x1, c = create_normalized_chunks(
                batch,
                args.horizon,
                stats,
                cond_type=args.condition_on,
                model_target=args.model_target,
            )
            if x1 is None:
                continue
            x1, c = x1.to(args.device), c.to(args.device)
        else:
            x1 = create_normalized_chunks(
                batch, args.horizon, stats, model_target=args.model_target
            )
            if x1 is None:
                continue
            x1 = x1.to(args.device)

        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=args.device)
        sample = path.sample(t=t, x_0=x0, x_1=x1)

        if args.condition_on:
            pred = model(sample.x_t, sample.t, c=c)
        else:
            pred = model(sample.x_t, sample.t)

        loss = torch.pow(pred - sample.dx_t, 2).mean()
        loss.backward()
        optim.step()
        total_loss += loss.item()
        total_chunks += 1

    return total_loss / total_chunks if total_chunks > 0 else 0.0


def evaluate(env, model, stats, input_dim, args, logger=None):
    fig, ax = evaluate_open_loop(env, model, stats, input_dim, args, logger=logger)
    plt.close(fig)
    return fig, ax


def train(config, args, dataset, env, run_name=None, logger=None):
    obs_dim = config.obs_dim
    action_dim = config.action_dim

    model, input_dim = build_model(args, obs_dim, action_dim)
    dataloader, stats = build_dataloader(dataset, args)
    path = AffineProbPath(scheduler=CondOTScheduler())
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = "src/checkpoints"
    model_base = run_name if run_name is not None else "model"
    model_save_path = os.path.join(save_dir, model_base + ".pth")

    pp = pprint.PrettyPrinter(indent=2)
    print("Training configuration:")
    pp.pprint(config)
    print("Starting training...")

    for epoch in range(args.num_epochs):
        start_time = time.time()
        epoch_loss = run_epoch(model, dataloader, path, optim, args, stats)
        if logger:
            logger.log({"epoch loss": epoch_loss})
        if (epoch + 1) % args.print_every == 0:
            elapsed = time.time() - start_time
            print(
                f"| Epoch {epoch+1:6d} | {elapsed:.2f} s/epoch | Loss {epoch_loss:8.5f} |"
            )

        if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            evaluate(env, model, stats, input_dim, args, logger=logger)

    print("Training complete. Saving model...")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    if logger:
        logger.save_model(model_save_path)
    print(f"Model saved to {model_save_path}, training complete.")
    return model, stats, input_dim


def main():
    args = parse_fm_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device:
        torch.set_default_device(args.device)
        print(f"Using device: {args.device}")
    config, dataset, env = load_dataset(args)
    cond = args.condition_on if args.condition_on else "none"
    run_name = (
        f"{args.environment}_{args.model_type}_h{args.horizon}_e{args.num_epochs}_"
        f"k{args.kernel_size}_chunk-{args.model_target}_cond-{cond}"
    )
    logger = None
    if not args.no_wandb:
        logger = WandBLogger(
            config={
                "environment": args.environment,
                "horizon": args.horizon,
                "batch_size": args.batch_size,
                "model_type": args.model_type,
                "hidden_dim": args.hidden_dim,
                "kernel_size": (
                    args.kernel_size
                    if args.model_type == "cnn" or args.model_type == "ccnn"
                    else 0
                ),
                "num_epochs": args.num_epochs,
                "lr": args.lr,
            },
            run_name=run_name,
        )
    model, stats, input_dim = train(
        config=config,
        args=args,
        dataset=dataset,
        env=env,
        run_name=run_name,
        logger=logger,
    )
    evaluate(env, model, stats, input_dim, args, logger=logger)
    if logger:
        logger.finish()


if __name__ == "__main__":
    main()
