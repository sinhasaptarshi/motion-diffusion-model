#!/usr/bin/env python
# coding: utf-8

# # Rerun Visualization

# In[ ]:


from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np
import rerun as rr
import trimesh
import pdb

# Configuration
DATA_DIR = Path("./temp_data")

# Bone connections for skeleton visualization
BONE_CONNECTIONS = np.array([
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20),
    (9, 14), (14, 17), (17, 19), (19, 21),
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
])

# Apply additional 90-degree rotation around X-axis if needed
Rx_90 = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
])

print("✓ All imports loaded successfully")
print(f"✓ Data directory: {DATA_DIR}")


# ## Utility Functions

# In[ ]:


def initialize_rerun(app_name: str, output_file: str) -> None:
    """Initialize Rerun with application name and output file."""
    rr.init(app_name)
    
    # Create output directory if it doesn't exist
    output_dir = Path("rerun_output")
    output_dir.mkdir(exist_ok=True)
    
    rr.save(output_dir / output_file)
    print(f"✓ Rerun initialized: {app_name}")
    print(f"✓ Output file: {output_dir / output_file}")


def load_pose_data(filename_joints: Union[str, None]=None, filename_vertices: Union[str, None]=None, data_dir: Path=DATA_DIR) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pose data including joints, vertices, and faces."""
    joints, vertices, faces = None, None, None

    if filename_joints is not None and (data_dir / filename_joints).exists():
        joints = np.load(data_dir / filename_joints)
        print(f"✓ Loaded joints: {joints.shape}")
    if filename_vertices is not None and (data_dir / filename_vertices).exists():
        vertices = np.load(data_dir / filename_vertices)
        print(f"✓ Loaded vertices: {vertices.shape}")
    if (data_dir / "smplx_faces.npy").exists():
        faces = np.load(data_dir / "smplx_faces.npy")
        print(f"✓ Loaded faces: {faces.shape}")
    
    return joints, vertices, faces


def load_vertex_colors(num_vertices: int, default_color: List[int] = [50, 80, 220]) -> np.ndarray:
    """Load vertex colors from file or create default colors."""
    color_path = DATA_DIR / "smplx_verts_colors.txt"
    
    if color_path.exists():
        vertex_colors = np.loadtxt(color_path)
        # Ensure colors are in uint8 format (0-255 range)
        if vertex_colors.max() > 1.0:
            vertex_colors = np.clip(vertex_colors, 0, 255).astype(np.uint8)
        else:
            vertex_colors = (vertex_colors * 255).astype(np.uint8)
        
        # Ensure we have the right number of vertices
        if vertex_colors.shape[0] != num_vertices:
            print(f"⚠️ Color file has {vertex_colors.shape[0]} colors, but mesh has {num_vertices} vertices")
            # Repeat or truncate as needed
            if vertex_colors.shape[0] < num_vertices:
                repeats = (num_vertices // vertex_colors.shape[0]) + 1
                vertex_colors = np.tile(vertex_colors, (repeats, 1))[:num_vertices]
            else:
                vertex_colors = vertex_colors[:num_vertices]
        
        print(f"✓ Loaded vertex colors from file: {vertex_colors.shape}")
    else:
        vertex_colors = np.full((num_vertices, 3), default_color, dtype=np.uint8)
        print(f"✓ Created default vertex colors: {vertex_colors.shape}")
    
    return vertex_colors


def load_pose_rotations(filename_rotations: str="body_rotations.npy", data_dir: Path=DATA_DIR) -> np.ndarray:
    """Load pose rotation data."""
    # Try different possible files for pose data
    pose_file = data_dir / filename_rotations
    pose_data = np.load(pose_file)
    print(f"✓ Loaded pose rotations from {pose_file}: {pose_data.shape}")
    return pose_data


def create_bone_lines(joints: np.ndarray, bone_connections: np.ndarray) -> List[List[np.ndarray]]:
    """Create bone line segments from joint positions."""
    bones = []
    for start_idx, end_idx in bone_connections:
        start_pos = joints[start_idx]
        end_pos = joints[end_idx]
        bones.append([start_pos, end_pos])
    return bones


def log_skeleton(entity_path: str, joints: np.ndarray, 
                joint_color: List[int] = [255, 255, 0],
                bone_color: List[int] = [128, 128, 128],
                joint_radius: float = 0.02) -> None:
    """Log skeleton joints and bones to Rerun."""
    # Log joints
    rr.log(
        f"{entity_path}/joints",
        rr.Points3D(
            positions=joints,
            radii=joint_radius,
            colors=joint_color
        )
    )
    
    # Log bones
    bones = create_bone_lines(joints, BONE_CONNECTIONS)
    rr.log(
        f"{entity_path}/bones", 
        rr.LineStrips3D(bones, colors=bone_color)
    )


def log_mesh(entity_path: str, vertices: np.ndarray, faces: np.ndarray,
             vertex_colors: Optional[np.ndarray] = None, alpha: Optional[float] = None) -> None:
    """Log 3D mesh to Rerun with optional transparency."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    vertex_normals = mesh.vertex_normals
    
    mesh_args = {
        "vertex_positions": vertices,
        "triangle_indices": faces,
        "vertex_normals": vertex_normals
    }
    
    # Handle vertex colors with transparency
    if vertex_colors is not None:
        if alpha is not None and alpha < 1.0:
            # Convert to RGBA format for transparency
            if vertex_colors.shape[1] == 3:  # RGB format
                num_vertices = vertex_colors.shape[0]
                alpha_values = np.full((num_vertices, 1), int(alpha * 255), dtype=np.uint8)
                vertex_colors = np.hstack([vertex_colors, alpha_values])
            elif vertex_colors.shape[1] == 4:  # Already RGBA
                vertex_colors[:, 3] = int(alpha * 255)
        mesh_args["vertex_colors"] = vertex_colors
    elif alpha is not None and alpha < 1.0:
        # Create semi-transparent default colors
        num_vertices = vertices.shape[0]
        default_color = [150, 150, 200]  # Light blue-gray
        alpha_value = int(alpha * 255)
        vertex_colors = np.full((num_vertices, 4), 
                               default_color + [alpha_value], dtype=np.uint8)
        mesh_args["vertex_colors"] = vertex_colors
    
    rr.log(entity_path, rr.Mesh3D(**mesh_args))


def rotation_matrix_to_axis_angle(rotation_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """Convert rotation matrix to axis-angle representation."""
    # Extract rotation axis and angle from rotation matrix
    trace = np.trace(rotation_matrix)
    angle = np.arccos((trace - 1) / 2)
    
    if np.abs(angle) < 1e-6:  # No rotation
        return np.array([0, 0, 1]), 0.0
    
    if np.abs(angle - np.pi) < 1e-6:  # 180 degree rotation
        # Special case for 180 degree rotation
        diagonal = np.diag(rotation_matrix)
        max_idx = np.argmax(diagonal + 1)
        axis = np.zeros(3)
        axis[max_idx] = np.sqrt((diagonal[max_idx] + 1) / 2)
        return axis, angle
    
    # General case
    axis = np.array([
        rotation_matrix[2, 1] - rotation_matrix[1, 2],
        rotation_matrix[0, 2] - rotation_matrix[2, 0],
        rotation_matrix[1, 0] - rotation_matrix[0, 1]
    ]) / (2 * np.sin(angle))
    
    return axis / np.linalg.norm(axis), angle


def pose_vector_to_rotation_matrices(pose_vector: np.ndarray) -> np.ndarray:
    """Convert pose vector to rotation matrices using Rodrigues formula."""
    # Assuming pose_vector contains axis-angle representations
    # Reshape to (num_joints, 3) if needed
    if pose_vector.ndim == 1:
        pose_vector = pose_vector.reshape(-1, 3)
    
    num_joints = pose_vector.shape[0]
    rotation_matrices = np.zeros((num_joints, 3, 3))
    
    for i, axis_angle in enumerate(pose_vector):
        angle = np.linalg.norm(axis_angle)
        if angle < 1e-6:
            rotation_matrices[i] = np.eye(3)
        else:
            axis = axis_angle / angle
            # Rodrigues formula
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            rotation_matrices[i] = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    return rotation_matrices


def log_rotation_axes(entity_path: str, joint_positions: np.ndarray, 
                     rotation_matrices: np.ndarray, axis_length: float = 0.1,
                     colors: List[List[int]] = None, yz_swap: bool = False) -> None:
    """Log rotation axes for each joint."""
    if colors is None:
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # Red, Green, Blue for X, Y, Z
    
    num_joints = joint_positions.shape[0]
    
    for joint_idx in range(num_joints):
        joint_pos = joint_positions[joint_idx]
        rot_matrix = rotation_matrices[joint_idx]
        
        # Extract X, Y, Z axes from rotation matrix
        axes = rot_matrix.T  # Columns are the rotated axes
        if yz_swap:
            axes[:, 0] = -axes[:, 0]
            axes[:, [1, 2]] = axes[:, [2, 1]]
        
        for axis_idx, (axis, color) in enumerate(zip(axes, colors)):
            start_pos = joint_pos
            end_pos = joint_pos + axis * axis_length
            
            # Log as arrow
            rr.log(
                f"{entity_path}/joint_{joint_idx}/axis_{axis_idx}",
                rr.Arrows3D(
                    origins=[start_pos],
                    vectors=[axis * axis_length],
                    colors=[color]
                )
            )

print("✓ Utility functions defined successfully")


# ### utility function for recovering joints

# In[ ]:


import torch
from quaternion import qinv, qrot, quaternion_to_cont6d

# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
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


def recover_from_ric(data, joints_num):
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

def cont6d_to_rotation_matrix(cont6d):
    """
    Convert continuous 6D rotation representation to rotation matrix.
    
    Args:
        cont6d: (..., 6) continuous 6D rotation representation
        
    Returns:
        rotation_matrix: (..., 3, 3) rotation matrices
    """
    if isinstance(cont6d, np.ndarray):
        cont6d = torch.from_numpy(cont6d).float()
    
    # Reshape to ensure we have the right dimensions
    original_shape = cont6d.shape[:-1]
    cont6d = cont6d.view(-1, 6)
    
    # Extract the two 3D vectors
    a1 = cont6d[:, :3]  # First column
    a2 = cont6d[:, 3:]  # Second column
    
    # Normalize first vector
    b1 = a1 / torch.norm(a1, dim=1, keepdim=True)
    
    # Gram-Schmidt process to get orthogonal second vector
    b2 = a2 - torch.sum(b1 * a2, dim=1, keepdim=True) * b1
    b2 = b2 / torch.norm(b2, dim=1, keepdim=True)
    
    # Third vector is cross product
    b3 = torch.cross(b1, b2, dim=1)
    
    # Stack to form rotation matrix
    rotation_matrix = torch.stack([b1, b2, b3], dim=2)  # (..., 3, 3)
    
    # Reshape back to original shape
    rotation_matrix = rotation_matrix.view(original_shape + (3, 3))
    
    return rotation_matrix


def rotation_matrix_to_axis_angle_torch(rotation_matrix):
    """
    Convert rotation matrix to axis-angle representation using PyTorch.
    
    Args:
        rotation_matrix: (..., 3, 3) rotation matrices
        
    Returns:
        axis_angle: (..., 3) axis-angle representation
    """
    if isinstance(rotation_matrix, np.ndarray):
        rotation_matrix = torch.from_numpy(rotation_matrix).float()
    
    original_shape = rotation_matrix.shape[:-2]
    rotation_matrix = rotation_matrix.view(-1, 3, 3)
    batch_size = rotation_matrix.shape[0]
    
    # Calculate trace
    trace = torch.diagonal(rotation_matrix, dim1=-2, dim2=-1).sum(-1)
    
    # Calculate angle
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-6, 1 - 1e-6))
    
    # Handle small angles (no rotation)
    small_angle_mask = torch.abs(angle) < 1e-4
    
    # Handle 180-degree rotations
    large_angle_mask = torch.abs(angle - np.pi) < 1e-4
    
    # Regular case
    regular_mask = ~(small_angle_mask | large_angle_mask)
    
    axis = torch.zeros(batch_size, 3, device=rotation_matrix.device, dtype=rotation_matrix.dtype)
    
    # Small angle case: return zero
    axis[small_angle_mask] = 0
    
    # Large angle case (180 degrees)
    if large_angle_mask.any():
        large_idx = torch.where(large_angle_mask)[0]
        for idx in large_idx:
            R = rotation_matrix[idx]
            # Find the eigenvector corresponding to eigenvalue 1
            eigenvals, eigenvecs = torch.linalg.eig(R)
            real_eigenvals = eigenvals.real
            real_eigenvecs = eigenvecs.real
            
            # Find index of eigenvalue closest to 1
            one_idx = torch.argmin(torch.abs(real_eigenvals - 1))
            axis_unnormalized = real_eigenvecs[:, one_idx]
            axis[idx] = axis_unnormalized / torch.norm(axis_unnormalized)
    
    # Regular case
    if regular_mask.any():
        regular_idx = torch.where(regular_mask)[0]
        R_regular = rotation_matrix[regular_idx]
        angle_regular = angle[regular_idx]
        
        # Extract axis from skew-symmetric part
        skew = (R_regular - R_regular.transpose(-2, -1)) / 2
        axis_unnormalized = torch.stack([
            skew[:, 2, 1],
            skew[:, 0, 2], 
            skew[:, 1, 0]
        ], dim=1)
        
        # Normalize by sin(angle)
        sin_angle = torch.sin(angle_regular).unsqueeze(-1)
        axis[regular_idx] = axis_unnormalized / sin_angle
    
    # Multiply by angle to get axis-angle representation
    axis_angle = axis * angle.unsqueeze(-1)
    
    # Reshape back
    axis_angle = axis_angle.view(original_shape + (3,))
    
    return axis_angle


def cont6d_to_axis_angle(cont6d):
    """
    Convert continuous 6D rotation representation to axis-angle representation.
    
    Args:
        cont6d: (..., 6) continuous 6D rotation representation
        
    Returns:
        axis_angle: (..., 3) axis-angle representation
    """
    # Convert cont6d to rotation matrix
    rotation_matrix = cont6d_to_rotation_matrix(cont6d)
    
    # Convert rotation matrix to axis-angle
    axis_angle = rotation_matrix_to_axis_angle_torch(rotation_matrix)
    
    return axis_angle


# ## Rotation Visualization

# In[ ]:


def create_rotation_visualization(
        save_path: str,
        data_dir: Path,
        filename_joints: Union[str, None],
        filename_vertices: Union[str, None],
        filename_rotations: Union[str, None],
        frame_limit: Union[int, None],
        rotate_Rx_90: bool = False,
    ) -> None:
    """Create visualization showing joint rotations with rotation axes."""
    initialize_rerun("joint_rotation_visualization", save_path)
    
    # Load data
    joints, vertices, faces = load_pose_data(filename_joints=filename_joints, filename_vertices=filename_vertices, data_dir=data_dir)
    if vertices is not None:
        vertex_colors = load_vertex_colors(vertices.shape[1])
    pose_data = load_pose_rotations(filename_rotations, data_dir=data_dir)
    
    if rotate_Rx_90:
        joints = joints @ Rx_90.T
        
    print(f"Creating rotation visualization with {joints.shape[0]} frames...")
    print(f"Pose data shape: {pose_data.shape}")
    
    if frame_limit is None or frame_limit > joints.shape[0]:
        frame_limit = joints.shape[0]

    for t in range(0, frame_limit):
        # Set timeline frame
        rr.set_time("frame", sequence=t)
        
        # Log skeleton
        log_skeleton("pose", joints[t], 
                    joint_color=[255, 255, 0],  # Yellow joints
                    bone_color=[128, 128, 128], # Gray bones
                    joint_radius=0.02)
        
        # Log mesh with transparency so skeleton is visible inside
        if vertices is not None:
            log_mesh("smpl/mesh", vertices[t], faces, vertex_colors)
        
        # Process pose data for this frame
        if pose_data.ndim == 2:
            # Assume it's a flattened representation, reshape to (num_joints, 3)
            pose_params_per_joint = 3
            num_joints_from_pose = pose_data.shape[1] // pose_params_per_joint
            num_joints_to_use = min(joints.shape[1], num_joints_from_pose)
            
            current_pose = pose_data[t, :num_joints_to_use * pose_params_per_joint].reshape(-1, 3)
        elif pose_data.ndim == 3:
            # Already in (num_frames, num_joints, 3) format
            num_joints_to_use = min(joints.shape[1], pose_data.shape[1])
            current_pose = pose_data[t, :num_joints_to_use]
        
        # Convert to rotation matrices
        rotation_matrices = pose_vector_to_rotation_matrices(current_pose)
        if rotate_Rx_90:
            rotation_matrices = Rx_90 @ rotation_matrices
        
        # Log rotation axes
        log_rotation_axes("pose", joints[t, :num_joints_to_use], rotation_matrices)

    print(f"✓ Joint rotation visualization saved to {save_path}")

def create_gt_vs_pred_joints_visualization(
        save_path: str,
        data_dir: Path,
        filename_joints_gt: Union[str, None],
        filename_joints_pred: Union[str, None],
        rotate_Rx_90: bool = True,
    ) -> None:
    """Create visualization showing gt and pred joints."""
    initialize_rerun("gt_vs_pred_joints", save_path)

    # Load data
    joints_gt, _, _ = load_pose_data(filename_joints=filename_joints_gt, data_dir=data_dir)
    joints_pred, _, _ = load_pose_data(filename_joints=filename_joints_pred, data_dir=data_dir)

    if rotate_Rx_90:
        joints_gt = joints_gt @ Rx_90.T
        joints_pred = joints_pred @ Rx_90.T

    print(f"Creating GT vs. Pred joints visualization with {joints_pred.shape[0]} frames...")

    for t in range(0, joints_pred.shape[0]):
        # Set timeline frame
        rr.set_time("frame", sequence=t)
        
        # Log skeleton
        log_skeleton("pose_gt", joints_gt[t], 
                    joint_color=[255, 255, 0],  # Yellow joints
                    bone_color=[128, 128, 128], # Gray bones
                    joint_radius=0.02)
        log_skeleton("pose_pred", joints_pred[t], 
                    joint_color=[255, 0, 0],  # Red joints
                    bone_color=[0, 0, 0], # black bones
                    joint_radius=0.02)

    print(f"✓ Joint rotation visualization saved to {save_path}")


# In[ ]:


def joints_and_rotations_from_data(
    motion_data_path: str,
    data_dir: Path = Path("./data/humanml"),
    filename_joints: str = "body_joints_humanml.npy",
    filename_rotations: str = "body_rotations_humanml.npy"
):
    # Body joints from
    new_joints_vecs = np.load(motion_data_path)
    print(f"✓ Loaded new joints: {new_joints_vecs.shape}")
    rec_ric_data = recover_from_ric(torch.from_numpy(new_joints_vecs).unsqueeze(0).float(), 22).cpu().numpy()[0]
    print(f"✓ Recovered positions from RIC data: {rec_ric_data.shape}")
    rot_data = new_joints_vecs[:, 67:193]  # Assuming 22 joints, each with 6D representation
    print(f"✓ Extracted rotation data: {rot_data.shape}")
    root_rot_quat = recover_root_rot_pos(torch.from_numpy(new_joints_vecs).unsqueeze(0).float())[0]
    root_rot_cont6d = quaternion_to_cont6d(root_rot_quat).numpy()[0]
    print(f"✓ Extracted root rotation data: {root_rot_cont6d.shape}")
    rec_rot_data = np.concatenate([root_rot_cont6d, rot_data], axis=-1).reshape(-1, 22, 6)
    print(f"✓ Extracted rotation data: {rec_rot_data.shape}")
    axis_angle_data = cont6d_to_axis_angle(torch.from_numpy(rec_rot_data)).numpy()
    print(f"Converted axis-angle data shape: {axis_angle_data.shape}")

    np.save(data_dir / filename_joints, rec_ric_data)
    np.save(data_dir / filename_rotations, axis_angle_data)


# ### Run example

# In[ ]:

index = 500

# folder2 = 'save_latest/my_humanml_trans_dec_bert_512_hdepic_v4_CA/samples_my_humanml_trans_dec_bert_512_hdepic_v4_CA_000550000_seed10/results.npy'
folder2 = 'save_latest/my_humanml_trans_dec_bert_512_hot3d/samples_my_humanml_trans_dec_bert_512_hot3d_000280000_seed10/results.npy'

predictions = np.load(folder2, allow_pickle=True)[None][0]
name_list = predictions['names']
preds = predictions['motion'].transpose(0, 3, 1, 2)[:376]
motion = preds[:,-1] - preds[:,0]
motion = np.linalg.norm(motion, axis=-1)[:,0]
index = np.argsort(motion)[-100]
# pdb.set_trace()
gt = np.load('dataset/HOT3D_test.npy', allow_pickle=True)[None][0]
name_list = gt['name_list']
# motion = gt['data_dict'].keys()\
name = name_list[index]

# pdb.set_trace()

pred_idx = np.where(np.array(name_list) == name)[0][0]
pred_motion = predictions['motion'][pred_idx].transpose(2, 0, 1)

motion = gt['data_dict'][name]['motion']
gt_motion = recover_from_ric(torch.Tensor(motion),22)
print(gt_motion.shape)
# if len(gt_motion) < len(pred_motion):
#     gt_motion = np.concatenate([gt_motion, gt_motion[-1:].repeat(len(pred_motion) - len(gt_motion), 1, 1)])
# else:
#     pred_motion = np.concatenate([pred_motion, torch.Tensor(pred_motion[-1:]).repeat(len(gt_motion) - len(pred_motion), 1, 1)])
# pdb.set_trace()


pred_motion = pred_motion[:len(gt_motion)]
np.save(f'temp_data/gt_motion_{index}.npy', gt_motion)
np.save(f'temp_data/pred_motion_{index}.npy', pred_motion)

# seq_id = "Apartment_release_decoration_skeleton_seq138_M1292_269032027457558_269035194124225"

# joints_and_rotations_from_data(
#     motion_data_path=DATA_DIR / f"predictions/{seq_id}.npy",
#     data_dir=DATA_DIR / "adt",
#     filename_joints="body_joints_adt_pred.npy",
#     filename_rotations="body_rotations_adt_pred.npy"
# )

# joints_and_rotations_from_data(
#     motion_data_path=DATA_DIR / f"gt/{seq_id}.npy",
#     data_dir=DATA_DIR / "adt",
#     filename_joints="body_joints_adt.npy",
#     filename_rotations="body_rotations_adt.npy"
# )

create_gt_vs_pred_joints_visualization(
    save_path=f"hot3d_test_{index}.rrd",
    data_dir=DATA_DIR,
    filename_joints_gt=f"gt_motion_{index}.npy",
    filename_joints_pred=f"pred_motion_{index}.npy",
    rotate_Rx_90=True
)


# In[ ]:




