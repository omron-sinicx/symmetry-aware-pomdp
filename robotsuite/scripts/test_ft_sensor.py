import argparse

import numpy as np
import matplotlib.pyplot as plt

import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="SoftPegInHole")
    parser.add_argument("--viz-torque", action='store_true', help="Whether to visualize torque or not")
    parser.add_argument("--robots", type=str, default="SoftUR5eRound", help="Which robot(s) to use")
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--controller", type=str, default="ik", help="Choice of controller. Can be 'ik' or 'osc'")
    parser.add_argument("--device", type=str, default="joystick")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    args = parser.parse_args()

    # Initialize the figure and axis

    colors = ['b', 'g', 'r']

    num_series = 3

    # Import controller config for EE IK or OSC (pos/ori)
    if args.controller == "ik":
        controller_name = "IK_POSE"
    elif args.controller == "osc":
        controller_name = "OSC_POSE"
    else:
        print("Error: Unsupported controller specified. Must be either 'ik' or 'osc'!")
        raise ValueError

    # Get controller config
    controller_config = load_controller_config(default_controller=controller_name)

    # Create argument configuration
    config = {
        "env_name": args.task,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    args.config = None

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
    )

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    idx2label = {0: "x", 1: "y", 2: "z"}

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard
        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "joystick":
        from robosuite.devices import Joystick
        device = Joystick(pos_xy_scale=args.pos_sensitivity,
                          pos_z_scale=args.pos_sensitivity,
                          rot_scale=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    while True:
        # Reset the environment
        obs = env.reset()

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
            if args.viz_torque:
                prefix = 'm-'
            else:
                prefix = 'f-'
            line, = ax.plot([], [], colors[i] + '-', label=prefix + idx2label[i])
            lines.append(line)

        ax.legend()

        # Setup rendering
        env.render()

        # Initialize device control
        device.start_control()

        while True:
            timestep += 1

            # Set active robot
            active_robot = env.robots[0]

            # Get the newest action
            if args.device == "keyboard":
                action, _ = input2action(
                    device=device, robot=active_robot, active_arm=args.arm, env_configuration
                    =args.config
                )
            else:
                action_dict = device.get_controller_state()

                action = np.zeros(7)
                action[0] = action_dict["front_back"]
                # action[1] = action_dict["left_right"]
                action[2] = action_dict["up_down"]

                if action_dict["reset"]:
                    action = None

            if action is None:
                plt.close()
                break

            forces = env.robots[0].get_sensor_measurement("robot0_force")
            torques = env.robots[0].get_sensor_measurement("robot0_torque")

            for j in range(num_series):
                # Append the new data point to the respective lists
                x_data[j].append(timestep)

                if args.viz_torque:
                    y_data[j].append(torques[j])
                else:
                    y_data[j].append(forces[j])

                # Update the line plot for each time series with the new data
                lines[j].set_data(x_data[j], y_data[j])

                ax.set_xlim(max(0, timestep - 10), timestep + 1)

            # Step through the simulation and render
            obs, reward, done, info = env.step(action)
            env.render()

            plt.pause(0.01)


plt.show()