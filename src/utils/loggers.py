from stable_baselines3.common.callbacks import BaseCallback
import wandb
import os
import time
import numpy as np
from functools import update_wrapper


class VideoLoggingCallback(BaseCallback):
    def __init__(self, video_dir, check_freq=2000, verbose=0):
        super().__init__(verbose)
        self.video_dir = video_dir
        self.check_freq = check_freq
        self.logged_files = set()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            if os.path.exists(self.video_dir):
                for fname in os.listdir(self.video_dir):
                    if fname.endswith(".mp4") and fname not in self.logged_files:
                        fpath = os.path.join(self.video_dir, fname)
                        wandb.log({f"video": wandb.Video(fpath, format="mp4")})
                        self.logged_files.add(fname)
                        if self.verbose > 0:
                            print(
                                f"[W&B] Logged video: {fname} at step {self.num_timesteps}"
                            )
        return True


class WandBLogger:
    def __init__(
        self, config, project_name="Flow Planner", entity="frankcholula", run_name=None
    ):
        print("Initializing WandB logger...")
        self.run = wandb.init(
            project=project_name, entity=entity, config=config, name=run_name
        )

    def log(self, data):
        self.run.log(data)

    def save_model(self, model_path):
        print(f"Saving {model_path} to WandB artifacts...")
        artifact = wandb.Artifact(name=f"model-{self.run.name}", type="model")
        artifact.add_file(local_path=model_path)
        self.run.log_artifact(artifact)
        print("Model artifact saved.")

    def finish(self):
        self.run.finish()
        print("WandB run finished.")


class EpisodeTimer:
    def __init__(self, func):
        self.func = func
        self.timings = []
        update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        start_time = time.perf_counter()
        result = self.func(*args, **kwargs)
        end_time = time.perf_counter()
        self.timings.append(end_time - start_time)
        return result

    def reset(self):
        self.timings = []

    def report_average_time(self):
        if not self.timings:
            avg_time = 0.0  # Handle cases where no calls were made
        else:
            avg_time = np.mean(self.timings)

        print(f"  Average '{self.func.__name__}' time this episode: {avg_time:.4f}s")
