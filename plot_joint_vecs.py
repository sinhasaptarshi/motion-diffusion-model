import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import pdb
import json
from os.path import join as pjoin
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed

def swap_left_right(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    # if data.shape[1] > 24:
    #     tmp = data[:, right_hand_chain]
    #     data[:, right_hand_chain] = data[:, left_hand_chain]
    #     data[:, left_hand_chain] = tmp
    return data


bone_connections = [(0,1), (1,4), (4,7), (7, 10), (0,2), (2,5), (5,8), (8,11),
                        (0,3), (3,6), (6,9), (9,12), (12,15), (9,13), (13,16),
                        (16,18), (18,20), (9,14), (14,17), (17,19), (19,21)]

# def compute_auto_view_params(joints_3d):
#     center = joints_3d.reshape(-1, 3).mean(axis=0)
#     max_range = (joints_3d.reshape(-1, 3).max(0) - joints_3d.reshape(-1, 3).min(0)).max()
#     return center, max_range, -90, 70
# def plot_and_save_animation(joints_3d, save_path='smplh_animation.mp4', fps=30, bone_connections=bone_connections):
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     center, max_range, elev, azim = compute_auto_view_params(joints_3d)
#     num_bones = len(bone_connections)
#     colors = plt.cm.get_cmap("tab20", num_bones)
    
    
#     lines = [ax.plot([], [], [], lw=2, c=colors(idx))[0] for idx, _ in enumerate(bone_connections)]

#     def init():
#         half_range = max_range / 2
#         ax.set_xlim(center[0] - half_range, center[0] + half_range)
#         # ax.set_ylim(center[1] - half_range, center[1] + half_range)
#         ax.set_zlim(center[2] - half_range, center[2] + half_range)
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#         ax.view_init(elev=elev, azim=azim)
#         return lines

#     def update(frame):
#         joints = joints_3d[frame]
#         for line, (i, j), c in zip(lines, bone_connections, range(num_bones)):
#             line.set_data([joints[i, 0], joints[j, 0]], [joints[i, 1], joints[j, 1]])
#             line.set_3d_properties([joints[i, 2], joints[j, 2]])
#         ax.set_title(f"Frame {frame + 1}/{len(joints_3d)}")
#         return lines

#     ani = FuncAnimation(fig, update, frames=len(joints_3d), init_func=init, blit=False, interval=1000 / fps)
#     ani.save(save_path, fps=fps, dpi=150)
#     plt.close()


def compute_auto_view_params(joints_3d):
    center = joints_3d.reshape(-1, 3).mean(axis=0)
    max_range = (joints_3d.reshape(-1, 3).max(0) - joints_3d.reshape(-1, 3).min(0)).max()
    return center, max_range, 45, 70 # Default elev, azim

def plot_and_save_animation(joints_3d, save_path='smplh_animation.mp4', fps=20, bone_connections=bone_connections, text_description = ""):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    center, max_range, elev, azim = compute_auto_view_params(joints_3d)

    num_bones = len(bone_connections)
    colors = plt.cm.get_cmap("tab20", num_bones)

    lines = [ax.plot([], [], [], lw=2, c=colors(idx))[0] for idx, _ in enumerate(bone_connections)]

    if text_description:
        fig.suptitle(text_description, fontsize=12, wrap=True)

    def init():
        half_range = max_range / 2

        # Data axes mapping:
        # Original:   joints_3d[:,:,0] -> X-axis, joints_3d[:,:,1] -> Y-axis, joints_3d[:,:,2] -> Z-axis
        # Desired:    joints_3d[:,:,0] -> Plot X, joints_3d[:,:,2] -> Plot Y, joints_3d[:,:,1] -> Plot Z (vertical)

        # Set limits according to the new mapping
        ax.set_xlim(center[0] - half_range, center[0] + half_range) # X remains X
        ax.set_ylim(center[2] - half_range, center[2] + half_range) # Z data goes to plot Y
        ax.set_zlim(center[1] - half_range, center[1] + half_range) # Y data goes to plot Z (vertical)

        # Set labels according to the new mapping
        ax.set_xlabel('X')
        ax.set_ylabel('Z') # Changed from Y to Z
        ax.set_zlabel('Y') # Changed from Z to Y (this is now the vertical axis)

        # Adjust the view angle for the new vertical axis (Y)
        # elev controls vertical rotation, azim controls horizontal rotation
        # A good starting point for Y-vertical might be elev=20 or 30, azim=-60
        ax.view_init(elev=20, azim=-60) # You'll likely need to adjust these

        return lines

    def update(frame):
        joints = joints_3d[frame]
        for line, (i, j), c in zip(lines, bone_connections, range(num_bones)):
            # IMPORTANT: Re-map the coordinates here!
            # X_plot = joints_data_X, Y_plot = joints_data_Z, Z_plot = joints_data_Y (vertical)
            line.set_data([joints[i, 0], joints[j, 0]],    # X-coordinates for plot
                          [joints[i, 2], joints[j, 2]])    # Z-coordinates for plot (now on y-axis)
            line.set_3d_properties([joints[i, 1], joints[j, 1]]) # Y-coordinates for plot (now on z-axis/vertical)
        ax.set_title(f"Frame {frame + 1}/{len(joints_3d)}")
        return lines

    ani = FuncAnimation(fig, update, frames=len(joints_3d), init_func=init, blit=False, interval=1000 / fps)
    ani.save(save_path, fps=fps, dpi=150)
    plt.close()


def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask
def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)
def recover_root_rot_pos(data, downsample_factor=12):
    """
    Recovers root rotation quaternion and root position from RIC data,
    adapted for downsampled input.

    Args:
        data (torch.Tensor): The input RIC data with shape (..., T, C).
                             C=263, where C[0] is root_rot_vel_y,
                             C[1:3] are root_local_pos_xz, C[3] is root_local_pos_y.
        downsample_factor (int): The factor by which the original data was downsampled.
                                 e.g., if downsampled to every 12 frames, this is 12.

    Returns:
        tuple: (r_rot_quat, r_pos)
               r_rot_quat (torch.Tensor): Root rotation quaternions.
               r_pos (torch.Tensor): Root positions.
    """
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)

    # Get Y-axis rotation from rotation velocity
    # If data is downsampled, rot_vel[..., :-1] represents the velocity at the previous
    # downsampled frame. To get the total angle change over the skipped frames,
    # we multiply by the downsample_factor.
    r_rot_ang[..., 1:] = rot_vel[..., :-1] * downsample_factor
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    # Convert Y-axis rotation angle to quaternion (around Y-axis)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    # Relative root position in XZ plane
    # Similar to rot_vel, if downsampled, data[..., :-1, 1:3] represents the displacement
    # at the previous downsampled frame. Multiply by downsample_factor for total displacement.
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3] * downsample_factor

    # Add Y-axis rotation to root position (this rotates the relative XZ displacement)
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    # Accumulate the root positions over time
    r_pos = torch.cumsum(r_pos, dim=-2) # Assuming -2 is the time dimension

    # Set the Y-component of the root position (this is likely an absolute Y-position)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

def recover_from_ric(data, joints_num=22, downsample_factor=12):
    """
    Recovers full joint positions from RIC data, adapted for downsampled input.

    Args:
        data (torch.Tensor): The input RIC data with shape (..., T, C).
                             C=263, containing root motion and relative joint positions.
        joints_num (int): The total number of joints (including root).
        downsample_factor (int): The factor by which the original data was downsampled.

    Returns:
        torch.Tensor: The recovered absolute joint positions.
    """
    # Call the modified recover_root_rot_pos with the downsample_factor
    r_rot_quat, r_pos = recover_root_rot_pos(data, downsample_factor)

    # Extract local joint positions (excluding root)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    # Add Y-axis rotation to local joints
    # Expand r_rot_quat to match the shape of positions for batch quaternion multiplication
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    # Add root XZ translation to joints
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    # Concatenate root position with local joint positions
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions
# actions = open(f'assets/nymeria_gt_examples_test.txt').readlines()
# pose = np.load('dataset/t2m_test.npy', allow_pickle=True)[None][0]
# data_dict = pose['data_dict']
# name_list = pose['name_list']

# for i, action in enumerate(actions):
#     action = action.strip('\n')
#     text_lines = open(f'dataset/Nymeria/texts/{action}.txt', 'r').readlines()
#     final_name = None
#     for name in name_list:
#         if action in name:
#             final_name = name
#             break
#     if final_name is None:
#         continue            
#     full_text_description = " ".join([line.strip() for line in text_lines if line.strip()]).split('#')[0]

#     # joint_vec = torch.Tensor(np.load(f'../motion-diffusion-model/dataset/Nymeria/new_joint_vecs/{action}.npy'))
#     joint_vec = data_dict[name]['motion']
#     # import pdb
#     # pdb.set_trace()
#     positions = recover_from_ric(torch.Tensor(joint_vec), downsample_factor=12).numpy()
    # positions = joint_vec.numpy()



# plot_and_save_animation(positions, save_path=f'gt_{action}_{i}.mp4', text_description=full_text_description)
final_name = '20231201_s1_daniel_wiley_act2_9qypr0_90713_91913'
positions = np.load(f'../TMR/predictions_1/{final_name}.npy')
# positions[..., 0] *= -1
positions = swap_left_right(positions)
annotations = json.load(open('../TMR/datasets/annotations/nymeria/annotations.json'))
text = annotations[final_name]['annotations'][0]['text']
plot_and_save_animation(positions, save_path=f'motion_example.mp4', text_description=text)