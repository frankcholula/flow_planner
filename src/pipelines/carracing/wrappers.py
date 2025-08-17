import gymnasium as gym
from gymnasium.spaces import Box
import torch
import numpy as np
from torchvision import transforms
from src.models.encoder import VAE
from tqdm import tqdm
import minari


class VAEObservationWrapper(gym.ObservationWrapper):
    """
    Gymnasium wrapper to encode image observations into latent vectors using a VAE.
    """

    def __init__(self, env, vae_model_path: str, latent_dim: int, device):
        super().__init__(env)
        self.device = device

        # Load the pre-trained VAE
        self.vae = VAE(latent_dim=latent_dim)
        self.vae.load_state_dict(torch.load(vae_model_path, map_location=device))
        self.vae.to(self.device)
        self.vae.eval()

        self.transform = transforms.ToTensor()
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(latent_dim,), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # Preprocess the image and add a batch dimension
        with torch.no_grad():
            image_tensor = self.transform(obs).unsqueeze(0).to(self.device)

            mu, _ = self.vae.encode(image_tensor)
            return mu.cpu().numpy().flatten()


def create_latent_dataset(vae_model_path, latent_dim=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load vae model
    print(f"Loading VAE from {vae_model_path}")
    vae = VAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load(vae_model_path, map_location=device))
    vae.eval()

    # load expert dataset
    source_dataset = minari.load_dataset("Box2D/CarRacing-v3/expert-v0")

    # encode
    episode_buffers = []
    print("Encoding all trajectories to latent space...")
    for episode in tqdm(
        source_dataset.iterate_episodes(), total=len(source_dataset.episode_indices)
    ):
        transform = transforms.ToTensor()
        obs_tensors = torch.stack([transform(obs) for obs in episode.observations]).to(
            device
        )
        with torch.no_grad():
            latent_vectors, _ = vae.encode(obs_tensors)

        # Create the episode buffer dictionary
        episode_buffer = {
            "observations": latent_vectors.cpu().numpy(),
            "actions": episode.actions,
            "rewards": episode.rewards,
            "terminations": episode.terminations,
            "truncations": episode.truncations,
        }
        episode_buffers.append(episode_buffer)

    base_env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
    minari_env = VAEObservationWrapper(
        env=base_env,
        vae_model_path=vae_model_path,
        latent_dim=latent_dim,
        device=device,
    )

    latent_dataset_id = "Box2D/CarRacing-v3/latent-v0"
    print(f"Saving new latent dataset: {latent_dataset_id}")
    latent_dataset = minari.create_dataset_from_buffers(
        dataset_id=latent_dataset_id,
        buffer=episode_buffers,
        env=minari_env,
        algorithm_name="PPO",
        description="VAE-Encoded PPO Expert",
        author="Frank Lu",
    )


def main():
    VAE_MODEL_PATH = "src/checkpoints/CarRacing-v3_vae_e50_l32_mixed.pth"
    create_latent_dataset(VAE_MODEL_PATH)


if __name__ == "__main__":
    main()
