import os
from stable_baselines3.common.callbacks import BaseCallback

class TrainAndLoggingCallback(BaseCallback):
    """
    A custom callback class for logging and saving the AI model periodically during training.
    Helps in monitoring performance and safeguarding against data loss.
    """
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq  # How often to check and save the model.
        self.save_path = save_path    # Directory for saving model checkpoints.

    def _init_callback(self):
        # Ensure the save directory exists; create it if necessary.
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        # Save the model at regular intervals based on the specified frequency.
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'best_model_{self.n_calls}')
            self.model.save(model_path)
        return True
