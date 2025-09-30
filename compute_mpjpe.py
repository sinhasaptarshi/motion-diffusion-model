
import numpy as np 
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import pdb
from smplx import SMPLH, SMPL
from scipy.ndimage import uniform_filter
import torch
from scipy.spatial.transform import Rotation as R
# from plot_hdepic_predictions import plot_and_save_animation

def calculate_mpjpe(sequence1: np.ndarray, sequence2: np.ndarray, align_root: bool = True) -> float:
    """
    Calculates the Mean Per Joint Position Error (MPJPE) between two 3D joint sequences.
    (This function is assumed to be defined as in the previous response)
    """
    # 1. Input dimension checks
    N1, J1, D1 = sequence1.shape
    N2, J2, D2 = sequence2.shape

    # Ensure the number of joints and dimensions match the specified format (22, 3)
    if not (J1 == J2 == 22 and D1 == D2 == 3):
        raise ValueError(
            f"Input sequences must have shape (N, 22, 3). "
            f"Got sequence1 shape {sequence1.shape} and sequence2 shape {sequence2.shape}."
        )

    # Ensure the number of frames are the same for direct comparison
    if N1 != N2:
        raise ValueError(
            f"Number of frames must be the same for both sequences to calculate MPJPE directly. "
            f"Got {N1} for sequence1 and {N2} for sequence2."
        )

    # Handle empty sequences
    if N1 == 0:
        return 0.0 # Or raise an error, depending on desired behavior for empty input

    # Create copies to avoid modifying original arrays during alignment
    seq1_processed = np.copy(sequence1)
    seq2_processed = np.copy(sequence2)

    # 2. Optional: Root Joint Alignment
    if align_root:
        # Assuming joint 0 is the root joint (e.g., pelvis)
        # Subtract the root joint's position from all other joints for each frame
        root_joint_seq1 = seq1_processed[:, 0:1, :]  # Shape (N, 1, 3)
        root_joint_seq2 = seq2_processed[:, 0:1, :]  # Shape (N, 1, 3)

        seq1_processed = seq1_processed - root_joint_seq1
        seq2_processed = seq2_processed - root_joint_seq2

    # 3. Calculate Euclidean Distances for each corresponding joint in each frame
    # The difference between the two sequences: (N, 22, 3)
    differences = seq1_processed - seq2_processed

    # Square the differences: (N, 22, 3)
    squared_differences = differences ** 2

    # Sum along the 3D coordinate dimension to get squared Euclidean distances: (N, 22)
    sum_squared_differences = np.sum(squared_differences, axis=-1)

    # Take the square root to get Euclidean distances for each joint in each frame: (N, 22)
    joint_distances = np.sqrt(sum_squared_differences)

    # 4. Compute Mean Per Joint Position Error
    # Calculate the mean across all frames and all joints
    mpjpe = np.mean(joint_distances)

    return mpjpe


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

def calculate_foot_skating(self, motion):
    """
    Calculate foot skating based on feet joints
    its the same as calculate_foot_skating_percentage
    """
    vertices = motion['joints']
    # get lowest vertex
    min_vertex_idx = vertices[:-1, ..., 2].argmin(axis=-1)
    # calculate min_vertex velocity
    vertex_diff = np.diff(vertices[..., :2], axis=0)
    # get only velocities of the minimum vertex that are above a threshold
    i, j = np.indices(min_vertex_idx.shape)
    min_vertex_diff = vertex_diff[i, j, min_vertex_idx]
    # get the norm of diffs
    min_vertex_diff_norm = np.linalg.norm(min_vertex_diff, axis=-1)
    return (min_vertex_diff_norm > ((2/3)*0.01)).sum(0) / min_vertex_diff_norm.shape[0]


def calculate_foot_skating_percentage(
    motion: dict, # Expects {'joints': np.ndarray of shape (N, 22, 3)}
    foot_joint_indices: list = [7, 8, 10, 11], # Left Ankle, Right Ankle, Left Toe, Right Toe
    up_axis_idx: int = 1, # Index for the vertical (up) axis: 0=X, 1=Y, 2=Z (1 for Y-up is common in SMPL)
    ground_height_threshold: float = 0.05, # Max height above ground for contact (in meters)
    horizontal_velocity_threshold: float = (2/3) * 0.01 # User's original velocity threshold (m/frame)
) -> float:
    """
    Calculates the percentage of frames where at least one specified foot joint is sliding.
    A foot joint is considered 'sliding' if it's near the ground AND moving horizontally
    above a given velocity threshold.

    Args:
        motion (dict): Dictionary containing 'joints' key with motion data
                       of shape (N, 22, 3).
        foot_joint_indices (list): List of integer indices corresponding to the foot joints
                                   (e.g., [7, 8, 10, 11] for left ankle, right ankle, left toe, right toe).
        up_axis_idx (int): The index of the vertical (up) axis in the 3D coordinates (0=X, 1=Y, 2=Z).
        ground_height_threshold (float): Maximum height a foot joint can be above the ground
                                         to be considered a potential ground contact point.
        horizontal_velocity_threshold (float): Maximum horizontal velocity magnitude for a foot joint
                                               to be considered stationary/in proper contact.
                                               Movement above this means it's sliding.

    Returns:
        float: The percentage (as a fraction, 0.0 to 1.0) of frames that exhibit foot skating.
    """
    vertices = motion # (N, 22, 3)
    num_frames = vertices.shape[0]

    if num_frames < 2:
        return 0.0 # Not enough frames to calculate movement

    # Determine horizontal component indices
    horizontal_components_idx = [i for i in [0, 1, 2] if i != up_axis_idx]

    # Calculate horizontal velocities for all joints between consecutive frames
    # shape: (N-1, 22, 2) for horizontal (x,z or x,y) components
    all_joint_horizontal_diff = np.diff(vertices[..., horizontal_components_idx], axis=0)
    
    # Calculate the norm (magnitude) of horizontal velocity for each joint for each frame difference
    # shape: (N-1, 22)
    all_joint_horizontal_vel_norm = np.linalg.norm(all_joint_horizontal_diff, axis=-1)

    # Initialize a boolean array to track which frames contain any sliding foot
    frames_exhibiting_skating = np.zeros(num_frames - 1, dtype=bool)
    # pdb.set_trace()

    # Iterate through each specified foot joint
    for foot_idx_in_all_joints in foot_joint_indices:
        # Get height for this specific foot joint across frames (N-1)
        # We use vertices[1:, ...] for height to align with velocity calculation,
        # which is difference from previous frame. This height is for the 'current' frame of the diff.
        foot_heights_current_frame = vertices[1:, foot_idx_in_all_joints, up_axis_idx]

        # Get the horizontal velocity norm for this specific foot joint
        foot_vel_norm = all_joint_horizontal_vel_norm[:, foot_idx_in_all_joints] # shape (N-1,)

        # Condition for sliding:
        # 1. Foot joint is near the ground (vertical position)
        # 2. Foot joint's horizontal velocity is above the threshold
        # is_this_foot_sliding = (foot_heights_current_frame < ground_height_threshold) & \
                            #    (foot_vel_norm > horizontal_velocity_threshold)
        
        is_this_foot_sliding = (foot_vel_norm > horizontal_velocity_threshold)
        
        # If this foot is sliding in any frame, mark that frame as having skating
        frames_exhibiting_skating = frames_exhibiting_skating | is_this_foot_sliding

    # The total number of frame differences is num_frames - 1
    total_frame_diffs = num_frames - 1
    if total_frame_diffs == 0:
        return 0.0

    # Calculate the percentage of frames where at least one foot was sliding
    skating_percentage = frames_exhibiting_skating.sum() / total_frame_diffs
    return skating_percentage, frames_exhibiting_skating.sum(), total_frame_diffs

foot_skating_p = []

# hdepic_folder = 'models_to_upload/humanml_trans_dec_512_bert/samples_humanml_trans_dec_512_bert_000600000_seed10/results.npy'
hdepic_folder = 'save_latest/my_humanml_trans_dec_bert_512_nymeria_final_V2/samples_my_humanml_trans_dec_bert_512_nymeria_final_V2_000600000_seed10/results.npy'
# hdepic_folder = 'save_latest/my_humanml_trans_dec_bert_512_hdepic_v4_CA_lossawaresampler/samples_my_humanml_trans_dec_bert_512_hdepic_v4_CA_lossawaresampler_000550000_seed10/results.npy'
# hdepic_folder = 'save_latest/my_humanml_trans_dec_bert_512_hdepic_v4_CA_lossawaresampler_humanML3D_pretrained/samples_my_humanml_trans_dec_bert_512_hdepic_v4_CA_lossawaresampler_humanML3D_pretrained_000540000_seed10/results.npy'
# folder = 'save_latest/my_humanml_trans_dec_bert_512_hot3d_humanML3D_pretrained/samples_my_humanml_trans_dec_bert_512_hot3d_humanML3D_pretrained_000630000_seed10/results.npy'
# folder = 'save_latest/my_humanml_trans_dec_bert_512_nymeria_final_V2/samples_my_humanml_trans_dec_bert_512_nymeria_final_V2_000600000_seed10/results.npy'
# folder = 'save_latest/my_humanml_trans_dec_bert_512_nymeria_final_V2/samples_my_humanml_trans_dec_bert_512_nymeria_final_V2_000600000_seed10/results.npy'
predictions = np.load(hdepic_folder, allow_pickle=True)[None][0]
motions = predictions['motion'].transpose(0,3,1,2)
# pdb.set_trace()
gt = np.load('dataset/HDEPIC_test.npy', allow_pickle=True)[None][0]
name_list = gt['name_list']
# motion = gt['data_dict'].keys()
# pdb.set_trace()
trans_matrix = np.linalg.inv(np.array([[1,0,0],[0,0,-1],[0,1,0]]))
names = np.array(predictions['names'])
MPJPES = []
start_MPJPES = []
goal_MPJPES = []
motion_trans = []
total_skatings = 0
total_gt_skatings = 0
total_frames = 0
total_gt_frames = 0

text = predictions['text']
mean = np.load('dataset/HDEPIC/Mean.npy')
std = np.load('dataset/HDEPIC/Std.npy')
for name in name_list:
# for i in range(len(motions)):
    orig_name = name[2:]
    # orig_name = name
    motion = gt['data_dict'][name]['motion']
    motion_norm = (motion - mean)/std 
    pelvis_disp = np.linalg.norm(motion_norm[-1] - motion_norm[0], axis=-1)
    gt_motion = recover_from_ric(torch.Tensor(motion), downsample_factor=1).numpy()
    # pdb.set_trace()
    
    # plot_and_save_animation(np.dot(gt_motion, trans_matrix), 'gt1.mp4', fps=20)
   
    index = np.where(names == orig_name)[0][0]
    pred_motion = motions[index]
    if gt_motion.shape[0] < pred_motion.shape[0]:
        indices = np.linspace(0, pred_motion.shape[0], gt_motion.shape[0]+1).astype(int)
        # pred_motion = pred_motion[indices[:-1]]
        pred_motion = pred_motion[:gt_motion.shape[0]]
    # elif gt_motion.shape[0] > pred_motion.shape[0]:
    #     indices = np.linspace(0, gt_motion.shape[0], pred_motion.shape[0]+1).astype(int)
    #     # gt_motion = gt_motion[indices[:-1]]
    #     gt_motion = gt_motion[:pred_motion.shape[0]]
    
    foot_skating, skatings, frames = calculate_foot_skating_percentage(pred_motion, [7,8,10,11])
    foot_skating_gt, skatings_gt, frames_gt = calculate_foot_skating_percentage(gt_motion, [7,8,10,11])
    foot_skating_p.append(foot_skating)
    total_skatings += skatings
    total_frames += frames
    total_gt_skatings += skatings_gt
    total_gt_frames += frames_gt
    # pdb.set_trace()

    MPJPES.append(calculate_mpjpe(gt_motion, pred_motion, align_root=False))
    start_MPJPES.append(calculate_mpjpe(gt_motion[0:1], pred_motion[0:1], align_root=False))
    goal_MPJPES.append(calculate_mpjpe(gt_motion[-1:], pred_motion[-1:], align_root=False))
    motion_trans.append(pelvis_disp)

start_MPJPES = np.array(start_MPJPES)
goal_MPJPES = np.array(goal_MPJPES)
motion_trans = np.array(motion_trans)
print(start_MPJPES[motion_trans > 100].mean())
print(goal_MPJPES[motion_trans > 100].mean())
print('MPJPES', sum(MPJPES)/len(MPJPES))
print('start MPJPEs', start_MPJPES.sum()/len(start_MPJPES))
print('goal MPJPEs', goal_MPJPES.sum()/len(goal_MPJPES))
print('Foot skating %', total_skatings/total_frames)
print('GT Foot skating %', total_gt_skatings/total_gt_frames)
idx = np.argsort(np.array(motion_trans))

plt.plot(np.array(motion_trans)[idx], uniform_filter(np.array(MPJPES)[idx], size=10, mode='nearest'))
plt.plot(np.array(motion_trans)[idx], uniform_filter(np.array(start_MPJPES)[idx], size=10, mode='nearest'))
plt.plot(np.array(motion_trans)[idx], uniform_filter(np.array(goal_MPJPES)[idx], size=10, mode='nearest'))

plt.legend(['MPJPE', 'Start pose MPJPE', 'Goal pose MPJPE'])
plt.xlabel('GT pelvis displament between start and goal')
plt.title('MPJPEs v/s pelvis translation')
plt.savefig('MPJPES_pelvis_translation.png')

pdb.set_trace()



# pdb.set_trace()