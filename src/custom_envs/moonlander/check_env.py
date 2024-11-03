from src.custom_envs.moonlander.moonlander_env import MoonlanderWorldEnv


# @hydra.main(config_name="main", config_path="../../../conf", version_base="1.1.2")
def main():
    environment = MoonlanderWorldEnv()
    # check_env(environment)

    observation, info = environment.reset()
    environment.render()

    terminated = False
    while not terminated:
        action = (
            environment.action_space.sample()
        )  # agent policy that uses the observation and info*.py
        observation, reward, terminated, truncated, info = environment.step(action)
        environment.render()

    environment.close()


if __name__ == "__main__":
    main()
