import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

def create_mario_env(apply_api_compatibility=True, render_mode='human'):
    """
    Configures and returns the Super Mario Bros environment tailored for AI interaction,
    applying necessary adjustments for control and rendering.
    """
    JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)  # Ensure compatibility with environment resets.
    env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=apply_api_compatibility, render_mode=render_mode)  # Create the Mario environment with specified settings.
    env = JoypadSpace(env, SIMPLE_MOVEMENT)  # Limit the AI's movement options to predefined simple movements.
    return env

def create_grayscale_stack_env():
    """
    Creates and configures a Super Mario Bros environment, converting it into grayscale and stacking
    multiple frames to provide the AI with temporal information.
    """
    env = create_mario_env()  # Start with the basic Mario environment.
    env = GrayScaleObservation(env, keep_dim=True)  # Convert environment images to grayscale.
    env = DummyVecEnv([lambda: env])  # Wrap the environment to work with vectorized actions.
    env = VecFrameStack(env, 4, channels_order='last')  # Stack 4 frames to give the model a sense of motion.
    return env
