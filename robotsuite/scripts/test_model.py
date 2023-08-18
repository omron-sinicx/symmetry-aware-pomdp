import time

import mujoco
import mujoco.viewer

import argparse

parser = argparse.ArgumentParser(description='Robosuite test')
parser.add_argument('--model', type=str)

args = parser.parse_args()

m = mujoco.MjModel.from_xml_path(args.model)
# m = mujoco.MjModel.from_xml_path("robosuite/models/assets/objects/square-hole.xml")
# m = mujoco.MjModel.from_xml_path("robosuite/models/assets/objects/round-hole.xml")
# m = mujoco.MjModel.from_xml_path("robosuite/models/assets/objects/plate-with-hole.xml")
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 1e6:
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)