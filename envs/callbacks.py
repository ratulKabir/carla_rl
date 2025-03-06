import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from torch.utils.tensorboard import SummaryWriter

class SaveBestAndManageCallback(BaseCallback):
    """
    Custom callback that:
      - Saves a checkpoint every `save_freq` timesteps.
      - Keeps only the most recent 5 checkpoint files.
      - Uses an EvalCallback to save the best model based on evaluation performance.
      - Logs action frequency distributions to TensorBoard every `save_freq` steps.
    """

    def __init__(self, eval_env, save_freq, save_path, n_eval_episodes=5, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.n_eval_episodes = n_eval_episodes
        self.saved_checkpoints = []  # List to store checkpoint filenames
        self.action_buffer_0 = []  # Store actions between logs
        self.action_buffer_1 = []
        self.writer = None  # TensorBoard writer (initialized in `_on_training_start`)

        # Create an evaluation callback for saving the best model
        self.eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            n_eval_episodes=n_eval_episodes,
            verbose=verbose
        )

    def _on_training_start(self):
        """Called at the beginning of training, initializes the TensorBoard writer."""
        self.writer = SummaryWriter(self.logger.dir)

    def _on_step(self) -> bool:
        """Called at every step during training."""
        actions_0 = self.locals["actions"][0, 0]
        actions_1 = self.locals["actions"][0, 1]  # Get the actions taken at this step
        # Convert to NumPy and store in buffer
        self.action_buffer_0.append(actions_0)
        self.action_buffer_1.append(actions_1)

        # Log action distribution every `save_freq` steps
        if self.num_timesteps % self.save_freq == 0:
            actions_np_0 = np.array(self.action_buffer_0)
            actions_np_1 = np.array(self.action_buffer_1)
            self.writer.add_histogram("action_distribution/maneuver", actions_np_0, self.num_timesteps)
            self.writer.add_histogram("action_distribution/long distance", actions_np_1, self.num_timesteps)
            self.action_buffer_0 = []
            self.action_buffer_1 = []

            # Save model checkpoint
            ckpt_path = os.path.join(self.save_path, f"model_{self.num_timesteps}.zip")
            self.model.save(ckpt_path)
            self.saved_checkpoints.append(ckpt_path)
            if self.verbose > 0:
                print(f"Saved checkpoint: {ckpt_path}")

            # Delete oldest checkpoints if more than 5 are saved
            if len(self.saved_checkpoints) > 5:
                old_ckpt = self.saved_checkpoints.pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)
                    if self.verbose > 0:
                        print(f"Deleted old checkpoint: {old_ckpt}")

        return True

    def _on_rollout_end(self):
        """Called at the end of each rollout to trigger evaluation."""
        self.eval_callback.on_rollout_end()

    def _on_training_end(self):
        """Called when training ends to properly close TensorBoard logging."""
        if self.writer:
            self.writer.close()
        self.eval_callback.on_training_end()
