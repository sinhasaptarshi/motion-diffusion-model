import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import pdb
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
        ax.set_zlim(-0.01, 1.8)
        # ax.set_zlim(center[1] - half_range, center[1] + half_range) # Y data goes to plot Z (vertical)

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
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num=22):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

actions = open(f'assets/nymeria_gt_examples_test.txt').readlines()

samples = np.load('save/my_humanml_trans_dec_bert_512_nymeria_v3/samples_my_humanml_trans_dec_bert_512_nymeria_v3_000600000_seed10_nymeria_examples_test/results.npy', allow_pickle=True)[None][0]
motions = samples['motion']
texts = samples['text']

for i, text in enumerate(texts): 
# action = 'M20230823_s0_evelyn_moody_act4_c5qn8i_7198_8398'
# text_lines = open(f'../motion-diffusion-model/dataset/Nymeria/texts/{action}.txt', 'r').readlines()
# full_text_description = " ".join([line.strip() for line in text_lines if line.strip()]).split('#')[0]

# joint_vec = torch.Tensor(np.load(f'../motion-diffusion-model/dataset/Nymeria/new_joint_vecs/{action}.npy'))
# positions = recover_from_ric(joint_vec).numpy()
# positions = joint_vec.numpy()
    
    full_text_description = text
    positions = motions[i].transpose(2, 0, 1)
    action = actions[i%len(actions)].strip('\n')
    plot_and_save_animation(positions, save_path=f'sample_results/prediction_{action}_{i}_pretrained.mp4', text_description=full_text_description)
