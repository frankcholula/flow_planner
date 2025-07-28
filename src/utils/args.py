import argparse
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flow matching trajectory generation.")

    training_args = parser.add_argument_group("Training arguments")
    training_args.add_argument(
        "--environment", type=str, default="LunarLander-v3", help="Environment name."
    )
    training_args.add_argument("--horizon", type=int, default=100)
    training_args.add_argument("--batch-size", type=int, default=32)
    training_args.add_argument("--num-epochs", type=int, default=1000)
    training_args.add_argument("--lr", type=float, default=1e-3)
    training_args.add_argument(
        "--model-type", type=str, default="ccnn", choices=["mlp", "cnn", "ccnn"]
    )
    training_args.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    training_args.add_argument("--seed", type=int, default=42)
    training_args.add_argument(
        "--no-wandb", action="store_true", help="Disable W&B logging"
    )

    inference_args = parser.add_argument_group("Inference arguments")
    inference_args.add_argument(
        "--step-size", type=float, default=0.05, help="Time step size for inference"
    )
    inference_args.add_argument(
        "--solver-method",
        type=str,
        default="midpoint",
        choices=["midpoint", "euler"],
    )

    cnn_args = parser.add_argument_group("CNN-specific arguments")
    cnn_args.add_argument(
        "--kernel-size", type=int, default=5, help="Kernel size for CNN models."
    )

    conditional_args = parser.add_argument_group("Conditional arguments")
    conditional_args.add_argument(
        "--condition-on",
        type=str,
        default="reward",
        choices=["reward", "start_obs"],
        help="Condition type for trajectory generation.",
    )
    return parser.parse_args()
