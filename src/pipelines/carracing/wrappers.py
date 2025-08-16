import gymnasium as gym
from gymnasium.spaces import Box
import torch
import numpy as np
from torchvision import transforms
from src.models.encoder import VAE


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VAE_MODEL_PATH = "src/checkpoints/CarRacing-v3_vae_e50_l32_mixed.pth"
    LATENT_DIM = 32

    base_env = gym.make("CarRacing-v3", render_mode="human")
    env = VAEObservationWrapper(base_env, VAE_MODEL_PATH, LATENT_DIM, device)
    z_obs, _ = env.reset()
    while True:
        action = env.action_space.sample()
        z_obs, reward, terminated, truncated, info = env.step(action)
        assert(z_obs.shape == (LATENT_DIM,))
        if terminated or truncated:
            break

if __name__ == "__main__":
    main()