import os
import minari
import time
import random
import numpy as np
import pprint

import torch
from torch.utils.data import DataLoader

from models.backbone import MLP, CNN, ConditionalCNN
from utils.args import parse_args
from utils.loggers import WandBLogger
from utils.visualizers import visualize_chunk

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from src.conf.environment import LunarLanderConfig


from pipelines.lunarlander.preprocessing import (
    collate_fn,
    get_dataset_stats,
    create_normalized_chunks,
    unnormalize_trajectory,
)


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)


def train(args):
    if args.environment == "LunarLander-v3":
        config = LunarLanderConfig()
    dataset = config.dataset_name
    obs_dim = config.obs_dim
    action_dim = config.action_dim
    horizon = args.horizon
    input_dim = horizon * (obs_dim + action_dim)

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    if args.model_type == "mlp":
        model = MLP(input_dim=input_dim, time_dim=1, hidden_dim=args.hidden_dim).to(
            args.device
        )
    elif args.model_type == "cnn":
        model = CNN(
            input_dim=input_dim, horizon=horizon, kernel_size=args.kernel_size
        ).to(args.device)
    elif args.model_type == "ccnn":
        model = ConditionalCNN(
            input_dim=input_dim, horizon=horizon, kernel_size=args.kernel_size
        ).to(args.device)

    stats = get_dataset_stats(dataset)
    path = AffineProbPath(scheduler=CondOTScheduler())
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

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
        }
    )
    run_name = f"{args.model_type}_h{args.horizon}_e{args.num_epochs}_k{args.kernel_size}_start_obs"
    model_name = run_name + ".pth"
    save_dir = "src/checkpoints"
    model_save_path = os.path.join(save_dir, model_name)

    pp = pprint.PrettyPrinter(indent=2)
    print("Training configuration:")
    pp.pprint(config)
    print("Run name:", run_name)
    print("Starting training...")

    for epoch in range(args.num_epochs):
        total_loss = 0.0
        total_chunks = 0
        start_time = time.time()
        for batch in dataloader:
            optim.zero_grad()
            x1 = create_normalized_chunks(batch, args.horizon, stats)
            if x1 is None:
                continue
            x1 = x1.to(args.device)
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], device=args.device)
            sample = path.sample(t=t, x_0=x0, x_1=x1)
            pred = model(sample.x_t, sample.t)
            loss = ((pred - sample.dx_t) ** 2).mean()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            total_chunks += 1
        avg_loss = total_loss / total_chunks if total_chunks > 0 else 0.0
        logger.log({"avg_epoch_loss": avg_loss})
        if (epoch + 1) % args.print_every == 0:
            elapsed = time.time() - start_time
            print(
                f"| Epoch {epoch+1:6d} | {elapsed:.2f} s/epoch | Loss {avg_loss:8.5f} |"
            )
            start_time = time.time()
    logger.finish()
    return model, stats, input_dim, obs_dim, action_dim


def generate(model, stats, input_dim, obs_dim, action_dim, args):
    wrapped = WrappedModel(model)
    solver = ODESolver(velocity_model=wrapped)
    T = torch.linspace(0, 1, 10, device=args.device)
    x_init = torch.randn((1, input_dim), device=args.device)
    sol = solver.sample(time_grid=T, x_init=x_init, method="midpoint", step_size=0.05)
    final_chunk = sol[-1].squeeze(0).detach()
    return unnormalize_trajectory(final_chunk, stats, args.horizon, obs_dim, action_dim)


def main():
    args = parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        torch.set_default_device(args.device)
    model, stats, input_dim, obs_dim, action_dim = train(args)
    obs, act = generate(model, stats, input_dim, obs_dim, action_dim, args)
    print("Generated observation shape:", obs.shape)
    print("Generated action shape:", act.shape)


if __name__ == "__main__":
    main()
