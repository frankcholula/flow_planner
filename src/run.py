import minari
import time
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from models.backbone import MLP, CNN, ConditionalCNN
from utils.args import parse_args
from utils.loggers import WandBLogger

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper


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
    dataset = minari.load_dataset(dataset_id=args.dataset_name)
    env = dataset.recover_environment()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    transition_dim = obs_dim + action_dim
    input_dim = args.horizon * transition_dim

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    stats = get_dataset_stats(dataset)

    model = ConditionalCNN(
        horizon=args.horizon, transition_dim=transition_dim, hidden_dim=args.hidden_dim
    ).to(args.device)
    path = AffineProbPath(scheduler=CondOTScheduler())
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    logger = WandBLogger(
        config={
            "horizon": args.horizon,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
        }
    )

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
