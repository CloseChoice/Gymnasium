# TODO: add install cell
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
import numpy as np
import matplotlib.pyplot as plt

# Set the parameters needed to generate the FrozenLake environment
map_size=7
seed=123
is_slippery=False
proba_frozen=0.9


# Create the FrozenLake environment
env = gym.make(
    "FrozenLake-v1",
    is_slippery=is_slippery,
    render_mode="rgb_array",
    desc=generate_random_map(
        size=map_size, p=proba_frozen, seed=seed
    ),
)

# We don't know how long it takes to train the model, but we know when it's good enough:
# If each run out of 100 succeeds we stop. We construct this by defining a reward threshold
# and use the eval callback in order to check the threshold after each batch.
# Stop training when the model reaches the reward threshold
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0.999, verbose=1)
eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)

# Create the A2C model and specifiy the policy: MlpPolicy
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20_000, callback=eval_callback, log_interval=100)

# Get the environment after we finished training
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(map_size * 3):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    if done:
        vec_env.render("human")

# From here on we'll just help to understand how the training progress works.

# In order to display the learning progress we need to create another callback
class CustomLoggingCallback(BaseCallback):
    def __init__(self, interval: int = 100, verbose=0):
        super(CustomLoggingCallback, self).__init__(verbose)
        self.rewards = []
        self.mean_rewards = []
        self.interval = interval

    def _on_step(self) -> bool:
        # Get the rewards for the current step
        reward = self.locals['rewards']
        self.rewards.append(reward)
        
        # Log the mean reward every `interval` steps
        if len(self.rewards) % self.interval == 0:
            mean_reward = np.mean(self.rewards[-self.interval:])
            self.mean_rewards.append(mean_reward)
            print(f"Step: {self.num_timesteps}, Mean Reward: {mean_reward}")
        return True

# Instantiate the custom callback
logging_callback = CustomLoggingCallback()

# Train the model with the custom callback
model.learn(total_timesteps=10_000, callback=[eval_callback, logging_callback], log_interval=100)
idx = range(len(logging_callback.rewards))
plt.plot(idx, logging_callback.rewards)
