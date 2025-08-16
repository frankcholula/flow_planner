from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from torchvision import transforms
from src.utils.args import parse_vae_args
from src.utils.loggers import WandBLogger
from src.models.encoder import VAE, vae_loss
import matplotlib.pyplot as plt
import torch
import tqdm
import minari
import random
import os


class MinariDataset(Dataset):
    def __init__(self, dataset_name):
        print(f"Loading Minari dataset '{dataset_name}'...")
        dataset = minari.load_dataset(dataset_name)
        self.observations = np.vstack(
            [e.observations for e in dataset.iterate_episodes()]
        )

        # The ToTensor transform handles scaling to [0,1] and dimension permutation (H,W,C -> C,H,W)
        self.transform = transforms.ToTensor()
        print("Dataset loaded and observations extracted.")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = self.observations[idx]
        return self.transform(obs)


def load_dataset(dataset_name: str):
    minari_dataset = MinariDataset(dataset_name)
    print(f"Total observations: {len(minari_dataset)}")
    print(f"Observation shape: {minari_dataset[0].shape}")

    # train test split
    train_size = int(0.9 * len(minari_dataset))
    val_size = len(minari_dataset) - train_size
    train_dataset, val_dataset = random_split(minari_dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    batch_size = 64

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print("\nData preparation complete.")
    print(
        f"Train DataLoader ready with {len(train_loader)} batches of size {batch_size}."
    )
    print(
        f"Validation DataLoader ready with {len(val_loader)} batches of size {batch_size}."
    )
    return train_loader, val_loader, minari_dataset


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    dataset: Dataset,
    config: dict,
    eval_freq: int = 5,
):
    # hyperparams
    epochs = config.epochs
    learning_rate = config.learning_rate
    latent_dim = config.latent_dim
    beta = config.beta

    # Setup
    run_name = f"CarRacing-v3_vae_e{epochs}_l{latent_dim}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logger = WandBLogger(config=config, run_name=run_name)
    MODEL_SAVE_PATH = os.path.join("src/checkpoints", f"{run_name}.pth")

    # MODIFICATION 1: Add a variable to track the best model
    best_val_loss = float("inf")
    for epoch in range(epochs):
        # training
        model.train()
        train_loss = 0
        pbar_train = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False
        )
        for batch in pbar_train:
            data = batch.to(device)
            recon_batch, mu, log_var = model(data)
            loss = vae_loss(recon_batch, data, mu, log_var, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar_train.set_postfix({"loss": f"{loss.item() / len(data):.4f}"})
        avg_train_loss = train_loss / len(train_loader.dataset)
        logger.log({"training_loss": avg_train_loss})

        # validation
        model.eval()
        val_loss = 0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():  # Disable gradient calculations for validation
            for batch in pbar_val:
                data = batch.to(device)
                recon_batch, mu, log_var = model(data)
                loss = vae_loss(recon_batch, data, mu, log_var, beta)
                val_loss += loss.item()
                pbar_val.set_postfix({"loss": f"{loss.item() / len(data):.4f}"})

        avg_val_loss = val_loss / len(val_loader.dataset)
        logger.log({"validation_loss": avg_val_loss})
        print(
            f"====> Epoch: {epoch+1} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

            print(f"✨ New best model saved with validation loss: {best_val_loss:.4f}")

        if (epoch + 1) % eval_freq == 0 or (epochs + 1) == config.epochs:
            print("Evaluating model...")
            eval(config=config, dataset=dataset, logger=logger, model=model)
    print("\nModel training complete.")
    logger.save_model(MODEL_SAVE_PATH)
    print(
        f"Best model saved to {MODEL_SAVE_PATH} with validation loss: {best_val_loss:.4f}"
    )


def eval(config, dataset, logger, model: VAE = None, model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        if model_path is None:
            raise ValueError("Either model or model_path must be provided.")
        print(f"Loading model from {model_path} for evaluation...")
        model = VAE(latent_dim=config.latent_dim)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

    model.eval()

    sample_idx = random.choice(range(len(dataset)))
    sample_obs_np = dataset.observations[sample_idx]
    image_tensor = dataset[sample_idx]
    image = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        mu, _ = model.encode(image)
        reconstructed_image_tensor = model.decode(mu)

    reconstructed_image_np = (
        reconstructed_image_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(sample_obs_np)
    axes[0].set_title("Before")
    axes[0].axis("off")

    axes[1].imshow(reconstructed_image_np)
    axes[1].set_title("After")
    axes[1].axis("off")

    logger.log({"Reconstruction Progress": fig})
    plt.close(fig)


def main():
    train_loader, val_loader, dataset = load_dataset("Box2D/CarRacing-v3/expert-v0")
    vae_args = parse_vae_args()
    print(vae_args)
    train(train_loader, val_loader, dataset, vae_args)


if __name__ == "__main__":
    main()
