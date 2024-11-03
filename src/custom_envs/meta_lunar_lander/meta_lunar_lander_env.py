import copy
import math
import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLander
import numpy as np

from matplotlib import pyplot as plt


class MetaLunarLanderEnv(gym.Env):
    render_mode = None
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(self):
        ### ACTION SPACE ###
        # one action to choose the task + four actions for each task
        # action 0 --> do nothing
        # action 1 --> fire left orientation engine
        # action 2 --> fire main engine
        # action 3 --> fire right orientation engine
        # action 4 --> switch action
        self.action_space = gym.spaces.Discrete(5)

        ### OBSERVATION SPACE ###
        # 8-dimensional vector for each task:
        # the coordinates of the lander in x
        # the coordinates of the lander in y
        # its linear velocities in x
        # its linear velocities in y
        # its angle
        # its angular velocity
        # and two booleans that represent whether each leg is in contact with the ground or not.
        low = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                -2.5,  # x coordinate
                -2.5,  # y coordinate
                # velocity bounds is 5x rated speed
                -10.0,
                -10.0,
                -2 * math.pi,
                -10.0,
                -0.0,
                -0.0,
                # second environment
                -2.5,  # x coordinate
                -2.5,  # y coordinate
                -10.0,
                -10.0,
                -2 * math.pi,
                -10.0,
                -0.0,
                -0.0,
            ]
        ).astype(np.float32)
        high = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                2.5,  # x coordinate
                2.5,  # y coordinate
                # velocity bounds is 5x rated speed
                10.0,
                10.0,
                2 * math.pi,
                10.0,
                1.0,
                1.0,
                # second environment
                2.5,  # x coordinate
                2.5,  # y coordinate
                10.0,
                10.0,
                2 * math.pi,
                10.0,
                1.0,
                1.0,
            ]
        ).astype(np.float32)

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(16,), dtype=np.float32)

        # first state
        self.lunar_lander_one = LunarLander()
        self.lunar_lander_two = LunarLander()
        self.state_of_lunar_lander_one, _ = self.lunar_lander_one.reset()
        self.state_of_lunar_lander_two, _ = self.lunar_lander_two.reset()
        # concatenate state with vector of dummy value
        # FIXME: what should be the value of the mask?
        self.mask = np.copy(low[0:8])
        self.current_task = 0
        self.state = np.concatenate((self.state_of_lunar_lander_one, self.mask)).flatten()

        self.rendering_first_time = True
        # counter
        self.episode_counter = 0
        self.step_counter = 0

    def step(self, action: int):
        task_switching_costs = 0
        # action 0,1,2 or 3 for each task
        match action:
            case 0:
                # do nothing
                (
                    self.state_of_lunar_lander_one,
                    reward_lunar_lander_one,
                    is_done_lunar_lander_one,
                    _,
                    info_lunar_lander_one,
                ) = self.lunar_lander_one.step(action=0)

                (
                    self.state_of_lunar_lander_two,
                    reward_lunar_lander_two,
                    is_done_lunar_lander_two,
                    _,
                    info_lunar_lander_two,
                ) = self.lunar_lander_two.step(action=0)
            case 1:
                # fire left orientation engine
                if self.current_task == 0:
                    (
                        self.state_of_lunar_lander_one,
                        reward_lunar_lander_one,
                        is_done_lunar_lander_one,
                        _,
                        info_lunar_lander_one,
                    ) = self.lunar_lander_one.step(action=1)

                    (
                        self.state_of_lunar_lander_two,
                        reward_lunar_lander_two,
                        is_done_lunar_lander_two,
                        _,
                        info_lunar_lander_two,
                    ) = self.lunar_lander_two.step(action=0)
                elif self.current_task == 1:
                    (
                        self.state_of_lunar_lander_one,
                        reward_lunar_lander_one,
                        is_done_lunar_lander_one,
                        _,
                        info_lunar_lander_one,
                    ) = self.lunar_lander_one.step(action=0)

                    (
                        self.state_of_lunar_lander_two,
                        reward_lunar_lander_two,
                        is_done_lunar_lander_two,
                        _,
                        info_lunar_lander_two,
                    ) = self.lunar_lander_two.step(action=1)
            case 2:
                # fire main engine
                if self.current_task == 0:
                    (
                        self.state_of_lunar_lander_one,
                        reward_lunar_lander_one,
                        is_done_lunar_lander_one,
                        _,
                        info_lunar_lander_one,
                    ) = self.lunar_lander_one.step(action=2)

                    (
                        self.state_of_lunar_lander_two,
                        reward_lunar_lander_two,
                        is_done_lunar_lander_two,
                        _,
                        info_lunar_lander_two,
                    ) = self.lunar_lander_two.step(action=0)
                elif self.current_task == 1:
                    (
                        self.state_of_lunar_lander_one,
                        reward_lunar_lander_one,
                        is_done_lunar_lander_one,
                        _,
                        info_lunar_lander_one,
                    ) = self.lunar_lander_one.step(action=0)

                    (
                        self.state_of_lunar_lander_two,
                        reward_lunar_lander_two,
                        is_done_lunar_lander_two,
                        _,
                        info_lunar_lander_two,
                    ) = self.lunar_lander_two.step(action=2)
            case 3:
                if self.current_task == 0:
                    (
                        self.state_of_lunar_lander_one,
                        reward_lunar_lander_one,
                        is_done_lunar_lander_one,
                        _,
                        info_lunar_lander_one,
                    ) = self.lunar_lander_one.step(action=3)

                    (
                        self.state_of_lunar_lander_two,
                        reward_lunar_lander_two,
                        is_done_lunar_lander_two,
                        _,
                        info_lunar_lander_two,
                    ) = self.lunar_lander_two.step(action=0)
                elif self.current_task == 1:
                    (
                        self.state_of_lunar_lander_one,
                        reward_lunar_lander_one,
                        is_done_lunar_lander_one,
                        _,
                        info_lunar_lander_one,
                    ) = self.lunar_lander_one.step(action=0)

                    (
                        self.state_of_lunar_lander_two,
                        reward_lunar_lander_two,
                        is_done_lunar_lander_two,
                        _,
                        info_lunar_lander_two,
                    ) = self.lunar_lander_two.step(action=3)
            case 4:
                # switch
                # TODO: switch only possible every 0.5 second = 5 frames?

                # still to step so that episode does not run forever
                (
                    self.state_of_lunar_lander_one,
                    reward_lunar_lander_one,
                    is_done_lunar_lander_one,
                    _,
                    info_lunar_lander_one,
                ) = self.lunar_lander_one.step(action=0)

                (
                    self.state_of_lunar_lander_two,
                    reward_lunar_lander_two,
                    is_done_lunar_lander_two,
                    _,
                    info_lunar_lander_two,
                ) = self.lunar_lander_two.step(action=0)

                if self.current_task == 0:
                    self.current_task = 1
                elif self.current_task == 1:
                    self.current_task = 0

                ### TODO: TASK-SWITCHING COSTS ###
                task_switching_costs = -10
                reward_lunar_lander_one -= 5
                reward_lunar_lander_two -= 5

            # If an exact match is not confirmed, this last case will be used if provided
            case _:
                raise ValueError("action must be 0, 1, 2, 3 or 4")

        if self.current_task == 0:
            self.state = np.concatenate((self.state_of_lunar_lander_one, self.mask)).flatten()
        elif self.current_task == 1:
            self.state = np.concatenate((self.mask, self.state_of_lunar_lander_two)).flatten()

        self.step_counter += 1
        info = {"lunar_lander_one": info_lunar_lander_one, "lunar_lander_two": info_lunar_lander_two,
                "task_switching_costs": task_switching_costs}
        return (
            self.state,
            reward_lunar_lander_one + reward_lunar_lander_two,
            is_done_lunar_lander_one or is_done_lunar_lander_two,
            False,
            info,
        )

    def render(self):
        self.lunar_lander_one.render_mode = "rgb_array"
        self.lunar_lander_two.render_mode = "rgb_array"
        # get rgb image of current lunar landers
        if self.current_task == 0:
            img_lunar_lander_one = self.lunar_lander_one.render()
            img_lunar_lander_two = np.zeros_like(img_lunar_lander_one)
        elif self.current_task == 1:
            img_lunar_lander_two = self.lunar_lander_two.render()
            img_lunar_lander_one = np.zeros_like(img_lunar_lander_two)
        if self.rendering_first_time:
            plt.ion()
            self.fig, self.ax = plt.subplots(1, 2)
            self.im_0 = self.ax[0].imshow(img_lunar_lander_one)
            self.im_1 = self.ax[1].imshow(img_lunar_lander_two)
            self.rendering_first_time = False
        self.im_0.set_data(img_lunar_lander_one)
        self.im_1.set_data(img_lunar_lander_two)
        self.fig.canvas.draw()
        if self.render_mode == "rgb_array":
            return np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                self.fig.canvas.get_width_height()[::-1] + (3,))

    def reset(self, seed=None, options=None):
        self.state_of_lunar_lander_one, _ = self.lunar_lander_one.reset()
        self.state_of_lunar_lander_two, _ = self.lunar_lander_two.reset()
        # concatenate state with vector of dummy value
        # FIXME: what should be the value of the mask?
        self.current_task = 0
        self.state = np.concatenate((self.state_of_lunar_lander_one, self.mask)).flatten()

        self.rendering_first_time = True

        # counter
        self.episode_counter += 1
        self.step_counter = 0
        return self.state, {}
