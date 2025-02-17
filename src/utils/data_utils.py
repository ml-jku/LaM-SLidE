from collections import Counter
from typing import Dict

import torch
from mdtraj import Trajectory
from torch import Tensor

from src.utils.constants import ATOM_ENCODING


def random_rotation_matrix():
    """Generate a random 3x3 rotation matrix using PyTorch."""
    theta = 2 * torch.pi * torch.rand(1)  # Random rotation around the z-axis
    phi = torch.acos(2 * torch.rand(1) - 1)  # Random rotation around the y-axis
    psi = 2 * torch.pi * torch.rand(1)  # Random rotation around the x-axis

    Rz = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    Ry = torch.tensor(
        [[torch.cos(phi), 0, torch.sin(phi)], [0, 1, 0], [-torch.sin(phi), 0, torch.cos(phi)]]
    )
    Rx = torch.tensor(
        [[1, 0, 0], [0, torch.cos(psi), -torch.sin(psi)], [0, torch.sin(psi), torch.cos(psi)]]
    )
    R = torch.mm(Rz, torch.mm(Ry, Rx))  # Combined rotation matrix
    return R


def random_rotation_matrix_2D():
    theta = 2 * torch.pi * torch.rand(1)
    R = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]])
    return R


def centre_random_augmentation(tensor: Tensor, R: Tensor, translation: Tensor):
    if tensor.dim() == 2:
        # Single point cloud case (N, 3)
        center = torch.mean(tensor, dim=0)
        tensor = tensor - center
        return torch.mm(tensor, R.T) + translation
    elif tensor.size()[-1] == 3:
        # Batch case (B, N, 3)
        center = torch.mean(tensor, dim=1, keepdim=True)  # (B, 1, 3)
        tensor = tensor - center
        return torch.bmm(tensor, R.T.expand(tensor.size(0), 3, 3)) + translation


def rotate_point_cloud(tensor: Tensor, R: Tensor) -> Tensor:
    """Rotate a tensor or batch of tensors using a rotation matrix.

    Args:
        tensor: Point cloud tensor of shape (N, 3) for single point cloud
               or (B, N, 3) for batch of point clouds
        R: Rotation matrix of shape (3, 3)

    Returns:
        Rotated tensor with same shape as input
    """
    if tensor.dim() == 2:
        # Single point cloud case (N, 3)
        center = torch.mean(tensor, dim=0)
        tensor = tensor - center
        tensor = torch.mm(tensor, R.T)
        tensor = tensor + center
    elif tensor.size()[-1] == 3:
        # Batch case (B, N, 3)
        # TODO: This could be written better that it always goes over dim2 to end
        center = torch.mean(tensor, dim=1, keepdim=True)  # (B, 1, 3)
        tensor = tensor - center
        tensor = torch.bmm(tensor, R.T.expand(tensor.size(0), 3, 3))
        tensor = tensor + center
    elif tensor.size()[-1] == 2:
        # 2D case (B, N, 2)
        center = torch.mean(tensor, dim=1, keepdim=True)  # (B, 1, 2)
        tensor = tensor - center
        tensor = torch.bmm(tensor, R.T.expand(tensor.size(0), 2, 2))
        tensor = tensor + center

    return tensor


def encode_atoms(trajectory: Trajectory, encoding: Dict[str, int] = ATOM_ENCODING) -> Tensor:
    return torch.LongTensor([encoding[atom.element.symbol] for atom in trajectory.top.atoms])


def encode_atom_radius(trajectory: Trajectory, constant_radius=None) -> Tensor:
    if constant_radius is not None:
        return torch.tensor([constant_radius for _ in trajectory.top.atoms])
    else:
        return torch.tensor([atom.element.radius for atom in trajectory.top.atoms])


def scale_to_new_range(x, old_min=-0.5, old_max=0.5, new_min=0.1, new_max=0.9):
    return (x - old_min) * (new_max - new_min) / (old_max - old_min) + new_min


def count_occurance(items: list, encoding: dict):
    encoding_type_count = Counter(items)
    for ae in encoding.values():
        if ae not in encoding_type_count:
            encoding_type_count[ae] = 0

    return encoding_type_count
