from math import cos, sin
import torch
from torchkit import pytorch_utils as ptu

import numpy as np


pos_x, pos_y, pos_z = 0, 1, 2
force_x, force_y, force_z = 3, 4, 5
torque_x, torque_y, torque_z = 6, 7, 8
act_x, act_y, act_z = 0, 1, 2


def numpy_rotate_vector_x(v, theta_deg):
    theta = np.deg2rad(theta_deg)
    rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    return np.dot(rot, v)

# def rotate_vector_X(vect, theta_deg):
#     "rotate a theta degree around X axis (CCW)"
#     theta = np.deg2rad(theta_deg)
#     rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
#     return torch.matmul(ptu.from_numpy(rot), vect)

def xy_2_xmy(obss):
    """
    (x, y) --> (x, -y)
    """
    obss[:, :, 0] *= -1.0

def rotate_vector(input_tensor, angle_deg, xy_indices):
    """
    Rotate the specified elements of the final channels in a PyTorch tensor.
    This function will change the input_tensor in-place.
    
    Args:
        input_tensor (torch.Tensor): Input tensor of size [batch_size, channels, height, width].
        angle_deg (torch.Tensor): Rotation angle in degrees.
        channel_indices (List[int]): List of channel indices to rotate.
    
    Returns:
        torch.Tensor: Rotated tensor with the same shape as the input_tensor.
    """
    # Extract the final channels of the tensor
    elements_to_rotate = input_tensor[:, :, xy_indices]

    # Extract the first and second elements (x, y coordinates) of the elements to rotate
    x_coordinates = elements_to_rotate[:, :, 0]
    y_coordinates = elements_to_rotate[:, :, 1]

    # Perform rotation
    angle_rad = torch.deg2rad(ptu.from_numpy(np.array([angle_deg])))
    cos_theta = torch.cos(angle_rad)
    sin_theta = torch.sin(angle_rad)

    rotated_x = cos_theta * x_coordinates - sin_theta * y_coordinates
    rotated_y = sin_theta * x_coordinates + cos_theta * y_coordinates

    # Update the rotated elements in each channel
    elements_to_rotate[:, :, 0] = rotated_x
    elements_to_rotate[:, :, 1] = rotated_y

    # Assign the modified elements back to the original channels
    input_tensor[:, :, xy_indices] = elements_to_rotate

def rotate_obss_acts(obss, acts, angle_deg):
    """
    Rotate observations and actions in-place.
    """
    ptu.rotate_vector(obss, angle_deg, [pos_x, pos_y])
    ptu.rotate_vector(obss, angle_deg, [force_x, force_y])
    ptu.rotate_vector(obss, angle_deg, [torque_x, torque_y])
    ptu.rotate_vector(acts, angle_deg, [act_x, act_y])

def rotate_means(means, angle_deg):
    """
    Rotate action means in-place.
    """
    ptu.rotate_vector(means, angle_deg, [act_x, act_y])

obs = torch.tensor([[[1, 2, 0, 4, 5, 0, 7, 8, 9]]]).float()
act = torch.tensor([[[0.1, 0.2, 0.0]]]).float()
means = torch.tensor([[[0.1, 0.2, 0.2]]]).float()

rotate_obss_acts(obs, act, -90)
rotate_means(means, 90)

print(obs)
print(act)
print(means)
