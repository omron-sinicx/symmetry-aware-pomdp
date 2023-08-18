from escnn import gspaces
from escnn import nn as enn
import torch
import numpy as np
from math import cos, sin
from torchkit import pytorch_utils as ptu

class GroupHelper():
    """
    Helper class for group operations
    """
    def __init__(self, group_name, env_name, actor_type, critic_type):
        self.group_name = group_name
        self.env_name = env_name
        self.actor_type = actor_type
        self.critic_type = critic_type

        define_group = self.actor_type == "equi" or self.critic_type == "equi"

        self.pos_x, self.pos_y, self.pos_z = 0, 1, 2
        self.force_x, self.force_y, self.force_z = 3, 4, 5
        self.torque_x, self.torque_y, self.torque_z = 6, 7, 8
        self.act_x, self.act_y, self.act_z = 0, 1, 2

        self.support_domains = ['PegInsertion-Square-XYZ-v0',
                                 'PegInsertion-Square-Old-XYZ-v0',
                                'PegInsertion-Triangle-XYZ-v0', 'PegInsertion-Round-XYZ-v0', 
                                'PegInsertion-Pentagon-XYZ-v0', 'PegInsertion-Hexagon-XYZ-v0'] 

        if self.group_name == 'FlipXY' and define_group:  # D2 group
            if self.env_name in self.support_domains:
                self.grp_act = gspaces.flipRot2dOnR2(2)
                self.irr_repr = [self.grp_act.irrep(1, 1)]
                self.reg_repr = [self.grp_act.regular_repr]
                self.triv_repr = [self.grp_act.trivial_repr]
                self.num_rotations = len(list(self.grp_act.testing_elements))

                # 2 irreducible for tip2hole xy, 1 trivial for tip2hole z
                # 2 irreducible for force2hole xy, 1 trivial for force2hole z
                # 2 irreducible for torque2hole xy, 1 trivial for torque2hole z
                self.obs_in_type = enn.FieldType(self.grp_act,
                                        2*self.irr_repr + self.triv_repr +
                                        2*self.irr_repr + self.triv_repr +
                                        2*self.irr_repr + self.triv_repr
                                        )

                # 2 irreducible for delta xy, 1 trivial for delta z
                self.act_in_type = enn.FieldType(self.grp_act,
                                        2*self.irr_repr + self.triv_repr
                                        )

                # 2 irreducible for delta xy, 1 trivial for delta z
                self.act_out_type = enn.FieldType(self.grp_act,
                                        2*self.irr_repr + self.triv_repr
                                        )

                #obs: 2 irreducible for tip2hole xy, 1 trivial for tip2hole z
                #obs: 2 irreducible for force2hole xy, 1 trivial for force2hole z
                #obs: 2 irreducible for torque2hole xy, 1 trivial for torque2hole z
                #act: 2 irreducible for delta xy, 1 trivial for delta z
                self.obs_act_in_type = enn.FieldType(self.grp_act,
                                        2*self.irr_repr + self.triv_repr +
                                        2*self.irr_repr + self.triv_repr +
                                        2*self.irr_repr + self.triv_repr +
                                        2*self.irr_repr + self.triv_repr
                                        )

                self.mean_out_type = enn.FieldType(self.grp_act,
                                        2*self.irr_repr + self.triv_repr
                                        )
                self.logstds_out_type = enn.FieldType(self.grp_act,
                                        3*self.triv_repr)

        elif  self.group_name in ['RotXY4', 'RotXY3', 'RotXY5', 'RotXY6'] and define_group:  # C4 group
            if self.env_name in self.support_domains:
                n_rots = int(self.group_name[-1])
                self.grp_act = gspaces.rot2dOnR2(n_rots)
                self.irr_repr = [self.grp_act.irrep(1)]
                self.reg_repr = [self.grp_act.regular_repr]
                self.triv_repr = [self.grp_act.trivial_repr]
                self.num_rotations = len(list(self.grp_act.testing_elements))

                # 1 irreducible for tip2hole xy, 1 trivial for tip2hole z
                # 1 irreducible for force2hole xy, 1 trivial for force2hole z
                # 1 irreducible for torque2hole xy, 1 trivial for torque2hole z
                self.obs_in_type = enn.FieldType(self.grp_act,
                                        self.irr_repr + self.triv_repr +
                                        self.irr_repr + self.triv_repr +
                                        self.irr_repr + self.triv_repr
                                        )

                # 1 irreducible for delta xy, 1 trivial for delta z
                self.act_in_type = enn.FieldType(self.grp_act,
                                        self.irr_repr + self.triv_repr
                                        )

                # 1 irreducible for delta xy, 1 trivial for delta z
                self.act_out_type = enn.FieldType(self.grp_act,
                                        self.irr_repr + self.triv_repr
                                        )

                #obs: 1 irreducible for tip2hole xy, 1 trivial for tip2hole z
                #obs: 1 irreducible for force2hole xy, 1 trivial for force2hole z
                #obs: 1 irreducible for torque2hole xy, 1 trivial for torque2hole z
                #act: 1 irreducible for delta xy, 1 trivial for delta z
                self.obs_act_in_type = enn.FieldType(self.grp_act,
                                        self.irr_repr + self.triv_repr +
                                        self.irr_repr + self.triv_repr +
                                        self.irr_repr + self.triv_repr +
                                        self.irr_repr + self.triv_repr
                                        )

                self.mean_out_type = enn.FieldType(self.grp_act,
                                        self.irr_repr + self.triv_repr
                                        )
                self.logstds_out_type = enn.FieldType(self.grp_act,
                                        3*self.triv_repr)

        elif  self.group_name == 'FlipRotXY4' and define_group:  # D4 group
            if self.env_name in self.support_domains:
                self.grp_act = gspaces.flipRot2dOnR2(4)
                self.irr_repr = [self.grp_act.irrep(1, 1)]
                self.reg_repr = [self.grp_act.regular_repr]
                self.triv_repr = [self.grp_act.trivial_repr]
                self.num_rotations = len(list(self.grp_act.testing_elements))

                # 1 irreducible for tip2hole xy, 1 trivial for tip2hole z
                # 1 irreducible for force2hole xy, 1 trivial for force2hole z
                # 1 irreducible for torque2hole xy, 1 trivial for torque2hole z
                self.obs_in_type = enn.FieldType(self.grp_act,
                                        self.irr_repr + self.triv_repr +
                                        self.irr_repr + self.triv_repr +
                                        self.irr_repr + self.triv_repr
                                        )

                # 1 irreducible for delta xy, 1 trivial for delta z
                self.act_in_type = enn.FieldType(self.grp_act,
                                        self.irr_repr + self.triv_repr
                                        )

                # 1 irreducible for delta xy, 1 trivial for delta z
                self.act_out_type = enn.FieldType(self.grp_act,
                                        self.irr_repr + self.triv_repr
                                        )

                #obs: 1 irreducible for tip2hole xy, 1 trivial for tip2hole z
                #obs: 1 irreducible for force2hole xy, 1 trivial for force2hole z
                #obs: 1 irreducible for torque2hole xy, 1 trivial for torque2hole z
                #act: 1 irreducible for delta xy, 1 trivial for delta z
                self.obs_act_in_type = enn.FieldType(self.grp_act,
                                        self.irr_repr + self.triv_repr +
                                        self.irr_repr + self.triv_repr +
                                        self.irr_repr + self.triv_repr +
                                        self.irr_repr + self.triv_repr
                                        )

                self.mean_out_type = enn.FieldType(self.grp_act,
                                        self.irr_repr + self.triv_repr
                                        )
                self.logstds_out_type = enn.FieldType(self.grp_act,
                                        3*self.triv_repr)

    def rotate_obss_acts(self, obss, acts, angle_deg):
        """
        Rotate observations and actions in-place.
        """
        assert angle_deg <= 0
        ptu.rotate_vector(obss, angle_deg, [self.pos_x, self.pos_y])
        ptu.rotate_vector(obss, angle_deg, [self.force_x, self.force_y])
        ptu.rotate_vector(obss, angle_deg, [self.torque_x, self.torque_y])
        ptu.rotate_vector(acts, angle_deg, [self.act_x, self.act_y])

    def rotate_means(self, means, angle_deg):
        """
        Rotate action means in-place.
        """
        assert angle_deg >= 0
        ptu.rotate_vector(means, angle_deg, [self.act_x, self.act_y])

    def xy_2_xmy(self, obss, acts):
        """
        (x, y) --> (x, -y)
        """
        obss[:, :, [self.pos_y, self.force_y, self.torque_y]] *= -1.0
        acts[:, :, self.act_y] *= -1.0

    def xy_2_mxy(self, obss, acts):
        """
        (x, y) --> (-x, y)
        """
        obss[:, :, [self.pos_x, self.force_x, self.torque_x]] *= -1.0
        acts[:, :, self.act_x] *= -1.0

    def xy_2_mxmy(self, obss, acts):
        """
        (x, y) --> (-x, -y)
        """
        obss[:, :, [self.pos_x, self.pos_y]] *= -1.0
        obss[:, :, [self.force_x, self.force_y]] *= -1.0
        obss[:, :, [self.torque_x, self.torque_y]] *= -1.0

        acts[:, :, [self.act_x, self.act_y]] *= -1.0

    def transform_obss_acts(self, obss, acts):
        """Transform observations based for each group element"""
        assert self.critic_type in ['aug-aux', 'aug', 'aux']
        if self.env_name in self.support_domains:
            if self.group_name == 'FlipXY':
                num_copies = 3
                obs_copies = []
                act_copies = []
                for _ in range(num_copies):
                    obs_copies.append(obss.clone())
                    act_copies.append(acts.clone())

                new_obs_copies = []
                new_act_copies = []
                for i, (obs, act) in enumerate(zip(obs_copies, act_copies)):
                    if i == 0:
                        # (x, y) --> (x, -y)
                        self.xy_2_xmy(obs, act)

                    if i == 1:
                        # (x, y) --> (-x, -y)
                        self.xy_2_mxmy(obs, act)

                    if i == 2:
                        # (x, y) --> (-x, y)
                        self.xy_2_mxy(obs, act)

                    new_obs_copies.append(obs)
                    new_act_copies.append(act)

                return_obss = torch.cat(new_obs_copies, dim=1)
                return_acts = torch.cat(new_act_copies, dim=1)

                return return_obss, return_acts, num_copies

            elif self.group_name in ['RotXY3', 'RotXY4', 'RotXY5', 'RotXY6']:
                num_rotations = int(self.group_name[-1])
                num_copies = num_rotations - 1
                angle = 360 // num_rotations
                obs_copies = []
                act_copies = []
                for _ in range(num_copies):
                    obs_copies.append(obss.clone())
                    act_copies.append(acts.clone())

                new_obs_copies = []
                new_act_copies = []
                for i, (obs, act) in enumerate(zip(obs_copies, act_copies)):
                    self.rotate_obss_acts(obs, act, -angle * (i + 1))  # rotate CW
                    new_obs_copies.append(obs)
                    new_act_copies.append(act)

                assert len(new_obs_copies) == num_copies

                return_obss = torch.cat(new_obs_copies, dim=1)
                return_acts = torch.cat(new_act_copies, dim=1)

                return return_obss, return_acts, num_copies

            elif self.group_name in ['FlipXRotXY3', 'FlipRotXY4', 'FlipXRotXY5', 'FlipRotXY6']:
                num_rotations = int(self.group_name[-1])
                num_copies = 2*num_rotations - 1
                angle = 360 // num_rotations
                obs_copies = []
                act_copies = []
                for _ in range(num_rotations):
                    obs_copies.append(obss.clone())
                    act_copies.append(acts.clone())

                new_obs_copies = []
                new_act_copies = []

                # first (num_rotations - 1) elements are rotated ones
                for i, (obs, act) in enumerate(zip(obs_copies, act_copies)):
                    self.rotate_obss_acts(obs, act, -angle * i)  # rotate CW

                    # first component does not change so ingore
                    if i > 0:
                        new_obs_copies.append(obs.clone())  # store the rotated components
                        new_act_copies.append(act.clone())  # store the rotated components

                for obs, act in zip(obs_copies, act_copies):
                    self.xy_2_xmy(obs, act)   # flip over y-axis
                    new_obs_copies.append(obs)  # store both rotated + flipped
                    new_act_copies.append(act)  # store both rotated + flipped

                assert len(new_obs_copies) == num_copies, len(new_obs_copies)

                return_obss = torch.cat(new_obs_copies, dim=1)
                return_acts = torch.cat(new_act_copies, dim=1)

                return return_obss, return_acts, num_copies

            else:
                raise ValueError("Group not supported.")

        else:
            raise ValueError("Domain not supported.")

    def transform_other_means(self, other_means, i):
        """Transform action means based for each group element and the index
        index must match with the index when transforming the observation
        """
        assert self.critic_type in ['aug-aux', 'aug', 'aux']
        if self.env_name in self.support_domains:
            if self.group_name == 'FlipXY':
                assert 0 <= i <= 2
                if i == 0:
                    # (x, y) <-- (x, -y)
                    other_means[:, :, self.act_y] *= -1.0

                if i == 1:
                    # (x, y) <-- (-x, -y)
                    other_means[:, :, [self.act_x, self.act_y]] *= -1.0

                if i == 2:
                    # (x, y) <-- (-x, y)
                    other_means[:, :, self.act_x] *= -1.0

            elif self.group_name in ['RotXY3', 'RotXY4', 'RotXY5', 'RotXY6']:
                num_rotations = int(self.group_name[-1])
                angle = 360 // num_rotations
                assert 0 <= i < num_rotations - 1
                self.rotate_means(other_means, angle * (i + 1))

            elif self.group_name in ['FlipXRotXY3', 'FlipRotXY4', 'FlipXRotXY5', 'FlipRotXY6']:
                num_rotations = int(self.group_name[-1])
                angle = 360 // num_rotations

                # rotated back (CCW)
                if i < num_rotations - 1:
                    self.rotate_means(other_means, angle * (i + 1))
                else:
                    # flip over y-axis and then rotate back (CCW)
                    # (x, -y) --> (x, y)
                    other_means[:, :, self.act_y] *= -1.0
                    self.rotate_means(other_means, angle * (i - num_rotations + 1))

        else:
            raise ValueError("Domain not supported.")