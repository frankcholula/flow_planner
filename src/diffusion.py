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
from src.utils.args import parse_diffusion_args
from src.utils.loggers import WandBLogger
from src.pipelines.sampling import generate_diffusion_trajectory
from src.conf.environment import BipedalWalkerConfig, CarRacingConfig, LunarLanderConfig
from src.pipelines.preprocessing import (
    collate_fn,
    get_dataset_stats,
    create_normalized_chunks,
)
from src.pipelines.eval import (
    evaluate_open_loop,
    evaluate_open_loop_diffusion,
    evaluate_policy_mpc,
)

# diffusion
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


env_config_map = {
    "LunarLander-v3": LunarLanderConfig,
    "CarRacing-v3": CarRacingConfig,
    "BipedalWalker-v3": BipedalWalkerConfig,
}

scheduler_map = {"ddpm": DDPMScheduler, "ddim": DDIMScheduler}


def load_dataset(args, env_render_mode="rgb_array", eval_env=False):
    if args.environment not in env_config_map:
        raise ValueError(f"Unknown environment: {args.environment}")

    dataset_config = env_config_map[args.environment]()
    dataset = minari.load_dataset(dataset_id=dataset_config.dataset_name)
    recovered_env = dataset.recover_environment(
        eval_env=eval_env, render_mode=env_render_mode
    )
    return dataset_config, dataset, recovered_env


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
    elif args.model_type == "ccnn":
        print("Using CCNN for training...")
        # setting conditioning dimension
        if args.condition_on == "reward":
            cond_dim = 1
        elif args.condition_on == "start_obs":
            cond_dim = obs_dim
        elif args.condition_on == "start_obs_goal":
            cond_dim = obs_dim * 2
        else:
            raise ValueError(
                f"ConditionalCNN requires a valid --condition-on argument ('reward', 'start_obs', 'start_obs_goal'). but got: {args.condition_on!r}"
            )
        model = ConditionalCNN(
            input_dim=input_dim,
            horizon=horizon,
            hidden_dim=args.hidden_dim,
            kernel_size=args.kernel_size,
            cond_dim=cond_dim,
        ).to(args.device)

    elif args.model_type == "unet":
        print("Using UNet1D model for training...")
        if args.condition_on:
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
        else:
            print("Running unconditional UNet1D model...")
            cond_dim = 1
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


def run_epoch(model, dataloader, noise_scheduler, optim, args, stats):
    total_loss = 0.0
    total_chunks = 0
    for batch in dataloader:
        optim.zero_grad()

        # same conditioning logic, but we use x_0 as clean data
        if args.condition_on:
            x0, c = create_normalized_chunks(
                batch,
                args.horizon,
                stats,
                cond_type=args.condition_on,
                model_target=args.model_target,
            )
            if x0 is None:
                continue
            x0, c = x0.to(args.device), c.to(args.device)
        else:
            x0 = create_normalized_chunks(
                batch, args.horizon, stats, model_target=args.model_target
            )
            if x0 is None:
                continue
            x0 = x0.to(args.device)

        noise = torch.randn_like(x0)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (x0.shape[0],),
            device=x0.device,
        ).long()

        x_t = noise_scheduler.add_noise(x0, noise, timesteps)
        if args.condition_on:
            predicted_noise = model(x_t, timesteps, c=c)
        else:
            predicted_noise = model(x_t, timesteps)
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        loss.backward()
        optim.step()
        total_loss += loss.item()
        total_chunks += 1
    return total_loss / total_chunks if total_chunks > 0 else 0.0


def evaluate(
    env,
    model,
    noise_scheduler,
    stats,
    input_dim,
    args,
    logger=None,
    eval_mode="diffusion_open_loop",
):
    print("Evaluating model...")
    if eval_mode == "open_loop":
        fig, ax = evaluate_open_loop(env, model, stats, input_dim, args, logger=logger)
        plt.close(fig)
        return fig, ax
    if eval_mode == "diffusion_open_loop":
        fig, ax = evaluate_open_loop_diffusion(
            env=env,
            model=model,
            noise_scheduler=noise_scheduler,  # Pass the scheduler
            stats=stats,
            input_dim=input_dim,
            args=args,
            logger=logger,
        )
        plt.close(fig)
        return fig, ax
    if eval_mode == "mpc":
        if args.condition_on is None:
            condition_type = "unconditional"
        else:
            condition_type = args.condition_on

        if condition_type == "start_obs_goal":
            goal_obs = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float32)
        else:
            goal_obs = None

        diffusion_planner = lambda cond_dict: generate_diffusion_trajectory(
            model=model,
            noise_scheduler=noise_scheduler,
            stats=stats,
            args=args,
            input_dim=input_dim,
            condition=cond_dict,
        )
        model_rewards, _ = evaluate_policy_mpc(
            env=env,
            planner_fn=diffusion_planner,
            num_episodes=10,
            replan_freq=1,
            render=False,
            max_episode_length=300,
            condition_type=condition_type,
            goal_obs=goal_obs,
        )
        if logger:
            logger.log({"reward mean": np.mean(model_rewards)})
            logger.log({"reward std": np.std(model_rewards)})
        return model_rewards


def train(config, args, dataset, env, noise_scheduler, run_name=None, logger=None):
    # creating the model
    obs_dim = config.obs_dim
    action_dim = config.action_dim
    model, input_dim = build_model(args, obs_dim, action_dim)

    # getting the dataloadergit p and dataset statistics
    dataloader, stats = build_dataloader(dataset, args)
    # TODO: replace path with a noise scheduler
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = "src/checkpoints"
    model_base = run_name if run_name is not None else "model"
    model_save_path = os.path.join(save_dir, model_base + ".pth")

    # print out configuration for sanity checks
    pp = pprint.PrettyPrinter(indent=2)
    print("Dataset configuration:")
    pp.pprint(config)
    print("Training parameters:")
    pp.pprint(vars(args))
    print("Starting training...")

    for epoch in range(args.num_epochs):
        start_time = time.time()
        epoch_loss = run_epoch(model, dataloader, noise_scheduler, optim, args, stats)
        if logger:
            logger.log({"epoch loss": epoch_loss})
        if (epoch + 1) % args.print_every == 0:
            elapsed = time.time() - start_time
            print(
                f"| Epoch {epoch+1:6d} | {elapsed:.2f} s/epoch | Loss {epoch_loss:8.5f} |"
            )

        if args.eval_every and (epoch + 1) % args.eval_every == 0:
            evaluate(
                env=env,
                model=model,
                noise_scheduler=noise_scheduler,
                stats=stats,
                input_dim=input_dim,
                args=args,
                logger=logger,
            )
    print("Training complete. Saving model...")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    if logger:
        logger.save_model(model_save_path)
    print(f"Model saved to {model_save_path}, training complete.")
    return model, stats, input_dim


def runname_builder(args) -> str:
    parts = ["DIFFUSION"]
    if (env := getattr(args, "environment", None)) is not None:
        parts.append(env)
    if (model_type := getattr(args, "model_type", None)) is not None:
        parts.append(model_type)
    if (horizon := getattr(args, "horizon", None)) is not None:
        parts.append(f"h{horizon}")
    if (num_epochs := getattr(args, "num_epochs", None)) is not None:
        parts.append(f"e{num_epochs}")
    if (kernel_size := getattr(args, "kernel_size", None)) is not None:
        parts.append(f"k{kernel_size}")
    if (model_target := getattr(args, "model_target", None)) is not None:
        parts.append(f"target-{model_target}")
    if (condition_on := getattr(args, "condition_on", None)) is not None:
        parts.append(f"cond-{condition_on}")
    return "_".join(parts)


def main():
    args = parse_diffusion_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        torch.set_default_device(args.device)
        print(f"Using device: {args.device}")
    config, dataset, env = load_dataset(
        args, eval_env=True, env_render_mode="rgb_array"
    )
    run_name = runname_builder(args)
    print(run_name)
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
    noise_scheduler = scheduler_map[args.scheduler](args.num_train_timesteps)
    model, stats, input_dim = train(
        config=config,
        args=args,
        dataset=dataset,
        env=env,
        noise_scheduler=noise_scheduler,
        run_name=run_name,
        logger=logger,
    )

    evaluate(
        env=env,
        model=model,
        noise_scheduler=noise_scheduler,
        stats=stats,
        input_dim=input_dim,
        args=args,
        logger=logger,
        eval_mode="mpc",
    )
    if logger:
        logger.finish()


if __name__ == "__main__":
    main()
