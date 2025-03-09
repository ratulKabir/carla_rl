import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from torch.utils.tensorboard import SummaryWriter

class SaveBestAndManageCallback(BaseCallback):
    """
    Custom callback that:
      - Saves a checkpoint every `save_freq` timesteps.
      - Keeps only the most recent 5 checkpoint files.
      - Saves the best 5 models based on training rewards.
      - Adds reward value to the saved model filename.
      - Logs action frequency distributions to TensorBoard every `save_freq` steps.
    """

    def __init__(self, eval_env, save_freq, save_path, n_eval_episodes=5, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.n_eval_episodes = n_eval_episodes
        self.saved_checkpoints = []  # Stores recent checkpoints (last 5)
        self.best_models = []  # Stores best models based on training reward
        self.action_buffer_0 = []
        self.action_buffer_1 = []
        self.episode_rewards = []
        self.writer = None

        # Create an evaluation callback for saving the best model
        self.eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            n_eval_episodes=n_eval_episodes,
            verbose=verbose
        )

    def _on_training_start(self):
        """Called at the beginning of training, initializes TensorBoard writer."""
        self.writer = SummaryWriter(self.logger.dir)

    def _on_step(self) -> bool:
        """Called at every step during training."""
        actions_0 = self.locals["actions"][0, 0]
        actions_1 = self.locals["actions"][0, 1]
        self.action_buffer_0.append(actions_0)
        self.action_buffer_1.append(actions_1)

        # Store episode rewards
        reward = self.locals["rewards"][0]  # Assuming single agent
        self.episode_rewards.append(reward)

        if self.num_timesteps % self.save_freq == 0:
            # Log action distributions
            actions_np_0 = np.array(self.action_buffer_0)
            actions_np_1 = np.array(self.action_buffer_1)
            self.writer.add_histogram("action_distribution/maneuver", actions_np_0, self.num_timesteps)
            self.writer.add_histogram("action_distribution/long_distance", actions_np_1, self.num_timesteps)
            self.action_buffer_0 = []
            self.action_buffer_1 = []

            # Save model checkpoint (for recent models)
            ckpt_path = os.path.join(self.save_path, f"model_{self.num_timesteps}.zip")
            self.model.save(ckpt_path)
            self.saved_checkpoints.append(ckpt_path)

            if self.verbose > 0:
                print(f"Saved checkpoint: {ckpt_path}")

            # Remove oldest checkpoint if more than 5 saved
            if len(self.saved_checkpoints) > 5:
                old_ckpt = self.saved_checkpoints.pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)
                    if self.verbose > 0:
                        print(f"Deleted old checkpoint: {old_ckpt}")

        return True

    def _on_rollout_end(self):
        '''Called at the end of each rollout to evaluate'''
        self.eval_callback.on_rollout_end()
        """Called at the end of each rollout to save the best models based on training rewards."""
        mean_training_reward = np.sum(self.episode_rewards) / len(self.episode_rewards)
        self.episode_rewards = []  # Reset episode rewards for next rollout

        if mean_training_reward is not None:
            # Save best models if reward is among top 5
            if len(self.best_models) < 5 or mean_training_reward > min(self.best_models, key=lambda x: x[0])[0]:
                best_ckpt_path = os.path.join(
                    self.save_path, f"best_model_{mean_training_reward:.2f}_{self.num_timesteps}.zip"
                )
                self.model.save(best_ckpt_path)
                self.best_models.append((mean_training_reward, best_ckpt_path))

                if self.verbose > 0:
                    print(f"Saved new best model: {best_ckpt_path} (Reward: {mean_training_reward:.2f})")

                # Keep only top 5 best models
                if len(self.best_models) > 5:
                    worst_model = min(self.best_models, key=lambda x: x[0])  # Find worst
                    self.best_models.remove(worst_model)
                    if os.path.exists(worst_model[1]):
                        os.remove(worst_model[1])  # Delete worst model file
                        if self.verbose > 0:
                            print(f"Deleted worst best model: {worst_model[1]} (Reward: {worst_model[0]:.2f})")

    def _on_training_end(self):
        """Called when training ends to properly close TensorBoard logging."""
        if self.writer:
            self.writer.close()
        self.eval_callback.on_training_end()
