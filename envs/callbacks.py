import os
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

class SaveBestAndManageCallback(BaseCallback):
    """
    Custom callback that:
      - Saves a checkpoint every `save_freq` timesteps.
      - Keeps only the most recent 5 checkpoint files.
      - Uses an EvalCallback to save the best model based on evaluation performance.
    """
    def __init__(self, eval_env, save_freq, save_path, n_eval_episodes=5, verbose=1):
        super(SaveBestAndManageCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.n_eval_episodes = n_eval_episodes
        self.saved_checkpoints = []  # list to store checkpoint filenames
        
        # Create a separate EvalCallback for saving the best model.
        self.eval_callback = EvalCallback(eval_env,
                                          best_model_save_path=save_path,
                                          n_eval_episodes=n_eval_episodes,
                                          verbose=verbose)

    def _on_step(self) -> bool:
        # Periodically save a checkpoint.
        if self.num_timesteps % self.save_freq == 0:
            ckpt_path = os.path.join(self.save_path, f"model_{self.num_timesteps}.zip")
            self.model.save(ckpt_path)
            self.saved_checkpoints.append(ckpt_path)
            if self.verbose > 0:
                print(f"Saved checkpoint: {ckpt_path}")
            # Delete oldest checkpoints if more than 5 are saved.
            if len(self.saved_checkpoints) > 5:
                old_ckpt = self.saved_checkpoints.pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)
                    if self.verbose > 0:
                        print(f"Deleted old checkpoint: {old_ckpt}")
        return True

    def _on_rollout_end(self):
        # At the end of each rollout, run the evaluation callback.
        self.eval_callback.on_rollout_end()

    def _on_training_end(self):
        # Make sure to call the evaluation callback's training end routine.
        self.eval_callback.on_training_end()