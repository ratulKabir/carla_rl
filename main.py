import os
import yaml
from envs.carla_env import CarlaGymEnv
from envs.callbacks import SaveBestAndManageCallback
from envs.carla_env_render import MatplotlibAnimationRenderer
from stable_baselines3 import A2C

# Load configurations from YAML
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "envs/configs/config.yaml")
with open(CONFIG_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)

RENDER_CAMERA = config["RENDER_CAMERA"]
TRAIN = config["TRAIN"]
TEST = config["TEST"]
LOAD_MODEL = config["LOAD_MODEL"]
SAVED_MODEL_PATH = os.path.join(os.path.dirname(__file__), config["SAVED_MODEL_PATH"])

if __name__ == '__main__':
    try:
        env = CarlaGymEnv(render_enabled=False)
        eval_env = CarlaGymEnv(render_enabled=RENDER_CAMERA) 
        eval_env.seed(3)

        if TRAIN:
            # Create a directory for saving models/checkpoints.
            save_dir = "./saved_rl_models/1.1"
            os.makedirs(save_dir, exist_ok=True)

            # Create the custom callback: save a checkpoint every 10,000 timesteps.
            callback = SaveBestAndManageCallback(eval_env=eval_env, save_freq=1000, save_path=save_dir, n_eval_episodes=5, verbose=1)

            # Train a small A2C model to confirm everything runs.
            model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
            model.learn(total_timesteps=100_000, callback=callback)

        if TEST:
            # Load the saved model.
            if LOAD_MODEL:
                model = A2C.load(SAVED_MODEL_PATH, env=eval_env)
            else:
                model = A2C("MultiInputPolicy", env)
            # Now test with a custom action:
            obs = eval_env.reset()
            done = False
            step_count = 0

            renderer = MatplotlibAnimationRenderer()
            step = 0
            while not done:
                # Selelct random action from env
                # action = eval_env.action_space.sample()   # always drive x meters forward
                # Get action from the trained model
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                step_count += 1
                print(f"Step: {step_count}, Reward: {reward}, Sim Time: {info['sim_time']}")

                # ego_data = obs.get('ego')
                # neighbors_data = obs.get('neighbors')
                # map_data = obs.get('map')
                # route_ef = obs.get('global_route')
                # route_ef = route_ef[route_ef[:, 0] > 0]
                
                # Update the renderer with the latest simulation data.
                # renderer.update_data(ego_data, neighbors_data, map_data, None, route_ef)
                # renderer.update_plot(step)
                step += 1

    except KeyboardInterrupt:
        print("Simulation interrupted.")
    finally:
        env.close()
        print("Environment closed.")