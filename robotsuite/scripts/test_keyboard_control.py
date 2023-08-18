import argparse

import numpy as np
import matplotlib.pyplot as plt

import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=str, default="square")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--controller", type=str, default="ik", help="Choice of controller. Can be 'ik' or 'osc'")
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=0.005, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=0.001, help="How much to scale rotation user inputs")
    parser.add_argument("--print-joints", action="store_true", help="Visualize the action space of the robot")
    args = parser.parse_args()

    controller_name = "IK_POSE"

    colors = ['r', 'g', 'b']

    num_series = 3
    idx2label = {0: "x", 1: "y", 2: "z"}

    # Get controller config
    controller_config = load_controller_config(default_controller=controller_name)

    # Create argument configuration
    robots = f"SoftUR5e{args.shape}"
    config = {
        "env_name": "SoftPegInHole",
        "robots": robots,
        "controller_configs": controller_config,
    }

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=10,
        hard_reset=False,
    )

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback(device.on_press)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    while True:
        # Reset the environment
        obs = env.reset()

        env.render()

        # Initialize device control
        device.start_control()

        timestep = 0

        fig, ax = plt.subplots()
        lines = []

        thismanager = plt.get_current_fig_manager()
        thismanager.window.wm_geometry("+1400+0")

        # Set the x-axis limits
        ax.set_xlim(0, 10)
        # Set the y-axis limits
        ax.set_ylim(-10, 10.0)

        x_data = [[] for _ in range(num_series)]
        y_data = [[] for _ in range(num_series)]

        # Initialize the lines for each time series
        for i in range(num_series):
            prefix = 'f-'
            line, = ax.plot([], [], colors[i] + '-', label=prefix + idx2label[i])
            lines.append(line)

        ax.legend()

        while True:
            timestep += 1

            # Set active robot
            active_robot = env.robots[0] if args.config == "bimanual" else env.robots[args.arm == "left"]

            # Get the newest action
            action, grasp = input2action(
                device=device, robot=active_robot, active_arm=args.arm, env_configuration=args.config
            )

            # If action is none, then this a reset so we should break
            if action is None:
                plt.close()
                break

            # Step through the simulation and render
            obs, reward, done, info = env.step(action)
            env.render()

            if args.print_joints:
                print(env.robots[0].sim.data.qpos[:7])
            # forces = env.robots[0].get_sensor_measurement("robot0_force")
            # torques = env.robots[0].get_sensor_measurement("robot0_torque")
            obs = obs["all_sensors"][-9:]
            forces = obs[3:6] * 50.0
            torques = obs[6:9] * 5.0

            for j in range(num_series):
                # Append the new data point to the respective lists
                x_data[j].append(timestep)

                y_data[j].append(forces[j])

                # Update the line plot for each time series with the new data
                lines[j].set_data(x_data[j], y_data[j])

                ax.set_xlim(max(0, timestep - 10), timestep + 1)

            # print(env.robots[0].sim.data.qpos[:7])

            plt.pause(0.005)
plt.show()