import os
import time
import random
import pprint
import numpy as np
from matplotlib import pyplot as plt

import torch

from src.pipelines.sampling import generate_trajectory
from src.utils.args import parse_fm_args
from src.utils.loggers import WandBLogger
from src.utils.training import build_model, build_dataloader, load_dataset
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver

from src.pipelines.preprocessing import (
    create_normalized_chunks,
)
from src.pipelines.eval import (
    WrappedConditionalModel,
    WrappedModel,
    evaluate_open_loop,
    evaluate_policy_mpc,
)


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


def evaluate(
    env, model, stats, input_dim, args, logger=None, eval_mode="open_loop", dataset=None
):
    print("Evaluating the model...")
    if eval_mode == "open_loop":
        fig, ax = evaluate_open_loop(
            env=env,
            model=model,
            stats=stats,
            input_dim=input_dim,
            args=args,
            logger=logger,
            dataset=dataset,
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
        if condition_type == "unconditional":
            wrapped_vf = WrappedModel(model)
        else:
            wrapped_vf = WrappedConditionalModel(model)

        T = torch.linspace(0, 1, 10)
        T = T.to(device=args.device)
        solver = ODESolver(velocity_model=wrapped_vf)
        planner_fn = lambda cond_dict: generate_trajectory(
            stats=stats,
            solver=solver,
            T=T,
            input_dim=input_dim,
            horizon=args.horizon,
            condition=cond_dict,
            batch_size=args.inference_batch_size,
            model_target=args.model_target,
        )
        model_rewards, _ = evaluate_policy_mpc(
            env=env,
            planner_fn=planner_fn,
            num_episodes=10,
            replan_freq=1,
            render=False,
            max_episode_length=300,
            condition_type=condition_type,
            goal_obs=goal_obs,
            args=args,
            dataset=dataset,
        )
        if logger:
            logger.log({"reward mean": np.mean(model_rewards)})
            logger.log({"reward std": np.std(model_rewards)})
    return model_rewards


def train(config, args, dataset, env, run_name=None, logger=None):
    # creating the model
    obs_dim = config.obs_dim
    action_dim = config.action_dim
    model, input_dim = build_model(args, obs_dim, action_dim)

    # getting the dataloader and dataset statistics
    dataloader, stats = build_dataloader(dataset, args)
    path = AffineProbPath(scheduler=CondOTScheduler())
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
        epoch_loss = run_epoch(model, dataloader, path, optim, args, stats)
        if logger:
            logger.log({"epoch loss": epoch_loss})
        if (epoch + 1) % args.print_every == 0:
            elapsed = time.time() - start_time
            print(
                f"| Epoch {epoch+1:6d} | {elapsed:.2f} s/epoch | Loss {epoch_loss:8.5f} |"
            )

        if args.eval_every and (epoch + 1) % args.eval_every == 0:
            evaluate(env, model, stats, input_dim, args, logger=logger)
    print("Training complete. Saving model...")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    if logger:
        logger.save_model(model_save_path)
    print(f"Model saved to {model_save_path}, training complete.")
    return model, stats, input_dim


def runname_builder(args) -> str:
    parts = ["FM"]
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
    args = parse_fm_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device:
        torch.set_default_device(args.device)
        print(f"Using device: {args.device}")
    config, dataset, env = load_dataset(args)
    run_name = runname_builder(args)
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
    evaluate(
        env=env,
        model=model,
        stats=stats,
        input_dim=input_dim,
        args=args,
        logger=logger,
        eval_mode="mpc",
        dataset=dataset,
    )
    if logger:
        logger.finish()


if __name__ == "__main__":
    main()
