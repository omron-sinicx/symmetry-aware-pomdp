import argparse

import numpy as np
import matplotlib.pyplot as plt

import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import VisualizationWrapper
from robosuite.devices import Joystick

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=str, default="square")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--viz-torque", action='store_true', help="Whether to visualize torque or not")
    parser.add_argument("--controller", type=str, default="ik", help="Choice of controller. Can be 'ik' or 'osc'")
    parser.add_argument("--pos-xy-scale", type=float, default=0.02, help="How much to scale position user inputs")
    parser.add_argument("--pos-z-scale", type=float, default=0.02, help="How much to scale position user inputs")
    parser.add_argument("--rot-scale", type=float, default=0.05, help="How much to scale rotation user inputs")
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

    step_cnt = 0
    true_step_cnt = 0

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # initialize device
    device = Joystick(pos_xy_scale=args.pos_xy_scale,
                      pos_z_scale=args.pos_z_scale,
                      rot_scale=args.rot_scale)

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
            if args.viz_torque:
                prefix = 'm-'
            else:
                prefix = 'f-'
            line, = ax.plot([], [], colors[i] + '-', label=prefix + idx2label[i])
            lines.append(line)

        ax.legend()

        while True:
            timestep += 1

            # Set active robot
            active_robot = env.robots[0] if args.config == "bimanual" else env.robots[args.arm == "left"]

            # Get the newest action
            action_dict = device.get_controller_state()

            action = np.zeros(6)
            action[0] = action_dict["front_back"]
            action[1] = action_dict["left_right"]
            action[2] = action_dict["up_down"]
            # print(action[2])
            # action[3] = action_dict["rot_left_right"][-1]

            reset = action_dict["reset"]

            if np.linalg.norm(action) > 0.0:
                step_cnt += 1

            true_step_cnt += 1

            if args.print_joints:
                print(env.robots[0].sim.data.qpos[:7])

            # Step through the simulation and render
            obs, reward, done, info = env.step(action)
            env.render()

            obs = obs["all_sensors"][-9:]
            forces = obs[3:6] * 50.0
            torques = obs[6:9] * 5.0

            # forces = env.robots[0].get_sensor_measurement("robot0_force")
            # torques = env.robots[0].get_sensor_measurement("robot0_torque")

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

            # print(env.robots[0].sim.data.qpos[:7])

            if reset or done or reward > 0:
                plt.close()
                print(step_cnt, true_step_cnt, reward)
                step_cnt = 0
                true_step_cnt = 0
                if reward > 0:
                    print("Success!")
                break
            plt.pause(0.005)
plt.show()