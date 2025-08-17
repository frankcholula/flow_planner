import gymnasium as gym
import minari
from tqdm import tqdm
from minari.data_collector import EpisodeBuffer
from minari import EpisodeData, StepData
from rl_zoo3.wrappers import YAMLCompatResizeObservation
from src.pipelines.carracing.wrappers import VAEObservationWrapper
from src.models.encoder import VAE
import torch
from torchvision import transforms


def create_env(vae_model_path, latent_dim, device, input_dim=64):
    base_env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
    minari_env = YAMLCompatResizeObservation(env=base_env, shape=[input_dim, input_dim])
    minari_env = VAEObservationWrapper(
        env=minari_env,
        vae_model_path=vae_model_path,
        latent_dim=latent_dim,
        device=device,
    )
    return minari_env


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

    preprocess = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(64)])
    print("Encoding all trajectories to latent space...")

    for i, episode in enumerate(
        tqdm(
            source_dataset.iterate_episodes(), total=len(source_dataset.episode_indices)
        )
    ):
        obs_tensors = torch.stack([preprocess(obs) for obs in episode.observations]).to(
            device
        )
        with torch.no_grad():
            latent_vectors, _ = vae.encode(obs_tensors)

        episode_data = EpisodeBuffer(
            id=episode.id,
            observations=latent_vectors.cpu().numpy(),
            actions=episode.actions,
            rewards=episode.rewards,
            terminations=episode.terminations,
            truncations=episode.truncations,
            infos=episode.infos,
        )
        episode_buffers.append(episode_data)

    minari_env = create_env(vae_model_path, latent_dim, device)
    latent_dataset_id = "Box2D/CarRacing-v3/latent-v0"
    print(f"Saving new latent dataset: {latent_dataset_id}")
    latent_dataset = minari.create_dataset_from_buffers(
        dataset_id=latent_dataset_id,
        buffer=episode_buffers,
        env=minari_env,
        eval_env=minari_env,
        algorithm_name="PPO",
        description="VAE-Encoded PPO Expert",
        code_permalink="https://github.com/frankcholula/flow_planner",
        author="Frank Lu",
        author_email="lu.phrank@gmail.com",
    )


def main():
    VAE_MODEL_PATH = "src/checkpoints/CarRacing-v3_vae_e50_l32_mixed.pth"
    create_latent_dataset(VAE_MODEL_PATH)


if __name__ == "__main__":
    main()
