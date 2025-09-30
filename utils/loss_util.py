from diffusion.nn import mean_flat, sum_flat
import torch
import numpy as np
import pdb


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


def angle_l2(angle1, angle2):
    a = angle1 - angle2
    a = (a + (torch.pi/2)) % torch.pi - (torch.pi/2)
    return a ** 2

def diff_l2(a, b):
    return (a - b) ** 2

def masked_l2(a, b, mask, loss_fn=diff_l2, epsilon=1e-8, entries_norm=True, dataset=None):
    # assuming a.shape == b.shape == bs, J, Jdim, seqlen
    # assuming mask.shape == bs, 1, 1, seqlen
    if dataset is not None:
        orig_motion = dataset.t2m_dataset.inv_transform(a.cpu().permute(0, 2, 3, 1)).float()
        orig_motion = recover_from_ric(orig_motion, downsample_factor=1).cuda()
        start_motion = orig_motion[:,:,0:1].squeeze(1).squeeze(1)
        # pdb.set_trace()
        length = mask.sum(axis=-1, keepdims=True).unsqueeze(1).repeat(1,1,1,22,3)
        last_motion = torch.gather(orig_motion, dim=2, index=length-1 ).squeeze(1).squeeze(1)
        # pdb.set_trace()
        displacement = (torch.norm(last_motion - start_motion, dim=-1))[:,0]
        
    loss = loss_fn(a, b)
    weights = torch.arange(1,mask.shape[-1]+1).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(mask.shape[0],1,1,1).cuda()
    
    scaled_weights = weights/mask.sum(-1, keepdims=True)*200 + 0.5

    scaled_weights[displacement < 1.5] = 1

    # start_motion = a[:,:,:,0:1].squeeze(-1).squeeze(-1)
    # length = mask.sum(axis=-1, keepdims=True).repeat(1,263,1,1)
    # last_motion = torch.gather(a, dim=3, index=length-1 ).squeeze(-1).squeeze(-1)

    # displacement = (torch.norm(last_motion - start_motion, dim=1)/45)**2
    
    
    
    loss = loss * scaled_weights
    # pdb.set_trace()
    # start_motion = a[:,:,:,0:1].squeeze(-1).squeeze(-1)
    # length = mask.sum(axis=-1, keepdims=True).repeat(1,263,1,1)
    # last_motion = torch.gather(a, dim=3, index=length-1 ).squeeze(-1).squeeze(-1)
    # displacement = (torch.norm(last_motion - start_motion, dim=1)/45)**2
    # displacement = displacement/displacement.sum() * displacement.shape[0]
    # weights = displacement.reshape(-1, 1, 1, 1).repeat(1, 263, 1, 400)
    # loss = loss * weights
    # pdb.set_trace()
    
    loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
    
    n_entries = a.shape[1]
    if len(a.shape) > 3:
        n_entries *= a.shape[2]
    non_zero_elements = sum_flat(mask)
    if entries_norm:
        # In cases the mask is per frame, and not specifying the number of entries per frame, this normalization is needed,
        # Otherwise set it to False
        non_zero_elements *= n_entries
    # print('mask', mask.shape)
    # print('non_zero_elements', non_zero_elements)
    # print('loss', loss)
    mse_loss_val = loss / (non_zero_elements + epsilon)  # Add epsilon to avoid division by zero
    # print('mse_loss_val', mse_loss_val)
    return mse_loss_val


def masked_goal_l2(pred_goal, ref_goal, cond, all_goal_joint_names):
    all_goal_joint_names_w_traj = np.append(all_goal_joint_names, 'traj')
    target_joint_idx = [[np.where(all_goal_joint_names_w_traj == j)[0][0] for j in sample_joints] for sample_joints in cond['target_joint_names']]
    loc_mask = torch.zeros_like(pred_goal[:,:-1], dtype=torch.bool)
    for sample_idx in range(loc_mask.shape[0]):
        loc_mask[sample_idx, target_joint_idx[sample_idx]] = True
    loc_mask[:, -1, 1] = False  # vertical joint of 'traj' is always masked out
    loc_loss = masked_l2(pred_goal[:,:-1], ref_goal[:,:-1], loc_mask, entries_norm=False)
    
    heading_loss = masked_l2(pred_goal[:,-1:, :1], ref_goal[:,-1:, :1], cond['is_heading'].unsqueeze(1).unsqueeze(1), loss_fn=angle_l2, entries_norm=False)

    loss =  loc_loss + heading_loss
    return loss
