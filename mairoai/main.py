from callbacks import TrainAndLoggingCallback
from environment_setup import create_grayscale_stack_env  # Ensure we're using the correct environment
from stable_baselines3 import PPO
import os

# Paths for saving training checkpoints and logging details.
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Set up the Super Mario environment for the AI to train in.
env = create_grayscale_stack_env()  # Use the function that creates the correct environment setup

# Initialize callback for periodic saving during model training.
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

# Configure the PPO model for AI training, including setting the policy, environment,
# verbosity level, log directory, learning rate, and steps per update.
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.00005, n_steps=512)
model.learn(total_timesteps=1000000, callback=callback)  # Begin training the model.
model.save('final_mario_model')  # Save the final model after training.
