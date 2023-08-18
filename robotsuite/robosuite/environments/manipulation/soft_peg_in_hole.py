from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import RoundHoleObject, SquareHoleObject
from robosuite.models.objects import SquareHoleOldObject
from robosuite.models.objects import TriangleHoleObject, PentagonHoleObject, HexagonHoleObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat


class SoftPegInHole(SingleArmEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        if 'SquareOld' in robots:
            self.peg_type = 'square_old'
        elif 'Square' in robots:
            self.peg_type = 'square'
        elif 'Triangle' in robots:
            self.peg_type = 'triangle'
        elif 'Pentagon' in robots:
            self.peg_type = 'pentagon'
        elif 'Hexagon' in robots:
            self.peg_type = 'hexagon'
        elif 'Round' in robots:
            self.peg_type = 'round'
        else:
            raise ValueError(f"Invalid robot type: {robots}")

        # parameters for each type of peg
        self.limit_xy = 0.15
        self.limit_torque = 5
        self.limit_force = 50
        if self.peg_type in ['square']:
            self.peg_rel_offsetz = -0.00
            self.arm_tip_rel_offsetz = -0.186
            self.hole_offset = 0.81
        elif self.peg_type == 'round':
            self.peg_rel_offsetz = 0
            self.arm_tip_rel_offsetz = -0.211
            self.hole_offset = 0.81
        elif self.peg_type == 'pentagon':
            self.peg_rel_offsetz = 0
            self.arm_tip_rel_offsetz = -0.191
            self.hole_offset = 0.81
        elif self.peg_type == 'hexagon':
            self.peg_rel_offsetz = 0
            self.arm_tip_rel_offsetz = -0.191
            self.hole_offset = 0.81
        elif self.peg_type == 'triangle':
            self.peg_rel_offsetz = 0
            self.arm_tip_rel_offsetz = -0.191
            self.hole_offset = 0.81
        elif self.peg_type == 'square_old':
            self.peg_rel_offsetz = 0.025
            self.arm_tip_rel_offsetz = -0.186
            self.hole_offset = 0.83
        else:
            raise ValueError(f"Invalid peg type: {self.peg_type}")

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0

        # Right location and angle
        if self._check_success():
            reward = 1.0

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        assert 'SoftUR5e' in self.robots[0].name, 'Must be SoftUR5e robot!'

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        if self.peg_type == 'round':
            self.hole = RoundHoleObject(name="hole")
        elif self.peg_type == 'square':
            self.hole = SquareHoleObject(name="hole")
        elif self.peg_type == 'square_old':
            self.hole = SquareHoleOldObject(name="hole")
        elif self.peg_type == 'triangle':
            self.hole = TriangleHoleObject(name="hole")
        elif self.peg_type == 'pentagon':
            self.hole = PentagonHoleObject(name="hole")
        elif self.peg_type == 'hexagon':
            self.hole = HexagonHoleObject(name="hole")
        else:
            raise ValueError(f"Invalid peg type: {self.peg_type}")
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.hole)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.hole,
                x_range=[-0.1, -0.1],  # fix hole position
                y_range=[0.0, 0.0],
                rotation=0.0,  # fix orientation
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.hole
        )

        # Make sure to add relevant assets from peg and hole objects
        self.model.merge_assets(self.hole)

    def _setup_references(self):
        """
        Ses up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.hole_body_id = self.sim.model.body_name2id(self.hole.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # Get robot prefix and define observables modality
        modality = "object"

        # centralized sensor hub
        @sensor(modality=modality)
        def all_sensors(obs_cache):
            # World frame
            hole_pos_in_world = self.sim.data.body_xpos[self.hole_body_id]
            hole_rot_in_world = self.sim.data.body_xmat[self.hole_body_id].reshape((3, 3))
            hole_pose_in_world = T.make_pose(hole_pos_in_world, hole_rot_in_world)

            # 1. Relative pose of peg's tip in the hole coordiante
            # Peg in world frame
            peg_pos_in_world = self.sim.data.get_site_xpos("robot0_end_peg")
            peg_rot_in_world = self.sim.data.get_site_xmat("robot0_end_peg").reshape((3, 3))
            peg_pose_in_world = T.make_pose(peg_pos_in_world, peg_rot_in_world)

            # Peg in hole frame
            world_pose_in_hole = T.pose_inv(hole_pose_in_world)
            peg_pose_in_hole = T.pose_in_A_to_pose_in_B(peg_pose_in_world, world_pose_in_hole)
            peg_rel_pos, peg_rel_quat = T.mat2pose(peg_pose_in_hole)

            peg_rel_angles = T.mat2euler(T.quat2mat(peg_rel_quat))
            peg_rel_sin_euler = np.sin(peg_rel_angles)
            peg_rel_cos_euler = np.cos(peg_rel_angles)

            # 2. Relative pose of the tip of the arm in the hole coordinate (grip_site in the gripper model)
            arm_tip_pos_in_world = self.sim.data.get_site_xpos("gripper0_grip_site")
            arm_tip_rot_in_world = self.sim.data.get_site_xmat("gripper0_grip_site").reshape((3, 3))
            arm_tip_pose_in_world = T.make_pose(arm_tip_pos_in_world, arm_tip_rot_in_world)

            # Grip_site in hole frame
            arm_tip_pose_in_hole = T.pose_in_A_to_pose_in_B(arm_tip_pose_in_world, world_pose_in_hole)
            arm_tip_rel_pos, arm_tip_rel_quat = T.mat2pose(arm_tip_pose_in_hole)

            arm_tip_rel_angles = T.mat2euler(T.quat2mat(arm_tip_rel_quat))
            arm_tip_rel_sin_euler = np.sin(arm_tip_rel_angles)
            arm_tip_rel_cos_euler = np.cos(arm_tip_rel_angles)

            # check within workspace
            should_terminate = False
            if (abs(arm_tip_rel_pos[:2]) >= self.limit_xy).any():
                should_terminate = True

            # 3. Force and torque feedback in the hole coordinate

            # 3D force between site's body and its parent body, in site frame
            forces = self.robots[0].get_sensor_measurement('robot0_force')

            # 3D torque between site's body and its parent body, in site frame
            torques = self.robots[0].get_sensor_measurement('robot0_torque')

            # F/T sensor site in world frame
            site_pos_in_world = self.sim.data.get_site_xpos("robot0_ft_sensor")
            site_rot_in_world = self.sim.data.get_site_xmat("robot0_ft_sensor").reshape((3, 3))
            site_pose_in_world = T.make_pose(site_pos_in_world, site_rot_in_world)
            site_pose_in_hole = T.pose_in_A_to_pose_in_B(site_pose_in_world, world_pose_in_hole)

            new_forces, new_torques = T.force_in_A_to_force_in_B(forces, torques, site_pose_in_hole)

            peg_rel_pos[2] = peg_rel_pos[2] + self.peg_rel_offsetz
            arm_tip_rel_pos[2] = arm_tip_rel_pos[2] + self.arm_tip_rel_offsetz

            # print(peg_rel_pos, arm_tip_rel_pos, arm_tip_rel_angles)
            # print(peg_rel_pos, arm_tip_rel_pos, arm_tip_rel_angles)
            # print(peg_rel_pos, arm_tip_rel_pos, arm_tip_rel_angles)
            # print(peg_rel_pos, arm_tip_rel_pos, arm_tip_rel_angles)

            # peg_rel_pos = np.clip(peg_rel_pos, -self.limit_xy, self.limit_xy)
            peg_rel_pos = peg_rel_pos / self.limit_xy

            # arm_tip_rel_pos = np.clip(arm_tip_rel_pos, -self.limit_xy, self.limit_xy)
            arm_tip_rel_pos = arm_tip_rel_pos / self.limit_xy

            # new_forces = np.clip(new_forces, -self.limit_force, self.limit_force)
            new_forces = new_forces / self.limit_force

            # new_torques = np.clip(new_forces, -self.limit_torque, self.limit_torque)
            new_torques = new_torques / self.limit_torque

            should_terminate = np.array([should_terminate])

            return np.concatenate([peg_rel_pos, peg_rel_sin_euler, peg_rel_cos_euler,
                                   arm_tip_rel_sin_euler, arm_tip_rel_cos_euler, should_terminate,
                                   arm_tip_rel_pos, new_forces, new_torques])

        sensors = [all_sensors]
        names = [s.__name__ for s in sensors]

        # Create observables
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        self.sim.model.body_pos[self.hole_body_id] = (-0.1, 0.0, self.hole_offset)
        self.sim.model.body_quat[self.hole_body_id] = (1, 0, 0, 0)

    def _calculate_peg_pose_in_hole(self):
        """
        calculate the relative pose of peg in the hole coordinate
        return: relative position and euler angles in degree
        """
        # World frame
        peg_pos_in_world = self.sim.data.get_site_xpos("robot0_end_peg")
        peg_rot_in_world = self.sim.data.get_site_xmat("robot0_end_peg").reshape((3, 3))
        peg_pose_in_world = T.make_pose(peg_pos_in_world, peg_rot_in_world)

        hole_pos_in_world = self.sim.data.body_xpos[self.hole_body_id]
        hole_rot_in_world = self.sim.data.body_xmat[self.hole_body_id].reshape((3, 3))
        hole_pose_in_world = T.make_pose(hole_pos_in_world, hole_rot_in_world)

        world_pose_in_hole = T.pose_inv(hole_pose_in_world)

        peg_pose_in_hole = T.pose_in_A_to_pose_in_B(peg_pose_in_world, world_pose_in_hole)

        rel_pos, rel_quat = T.mat2pose(peg_pose_in_hole)

        rel_pos[2] = rel_pos[2] + self.peg_rel_offsetz

        rel_angles = T.mat2euler(T.quat2mat(rel_quat)) * 57.3

        return rel_pos, rel_angles

    def _check_success(self):
        """
        Check if peg is successfully aligned and placed within the hole

        Returns:
            bool: True if peg is placed in hole correctly
        """

        relative_pos, rel_angles = self._calculate_peg_pose_in_hole()
        pos_condition = np.all(abs(relative_pos) <= 0.01)

        # if pos_condition:
        #     print(relative_pos, rel_angles, self.robots[0].sim.data.qpos[:7], pos_condition)

        return pos_condition
