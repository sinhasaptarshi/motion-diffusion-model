import numpy as np 
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import pdb
from smplx import SMPLH, SMPL
import torch
from scipy.spatial.transform import Rotation as R

def quat_to_rotvec(quat_wxyz):
    """Convert quaternion coefficients (w, x, y, z) to rotation vector.

    Args:
        quat_wxyz: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation vector corresponding to the quaternion -- size = [B, 3]
    """
    # Convert wxyz to xyzw for scipy's R.from_quat
    quat_xyzw = torch.cat([quat_wxyz[:, 1:], quat_wxyz[:, :1]], dim=-1)
    # Convert to numpy for scipy
    rot = R.from_quat(quat_xyzw.numpy())
    return torch.from_numpy(rot.as_rotvec()).float()

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

def compute_auto_view_params(joints_3d, sphere_center_data=None):
    # Combine all relevant points for auto-scaling
    all_points = joints_3d.reshape(-1, 3)
    if sphere_center_data is not None:
        # Ensure sphere_center_data is 2D for concatenation if it's a single point
        if sphere_center_data.ndim == 1:
            sphere_points = sphere_center_data[np.newaxis, :]
        else:
            sphere_points = sphere_center_data.reshape(-1, 3) # Reshape in case it's (frames, 3)
        all_points = np.vstack((all_points, sphere_points))
        # pdb.set_trace()

    center = all_points.mean(axis=0)
    max_range = (all_points.max(0) - all_points.min(0)).max()
    return center, max_range, 45, 135

def plot_and_save_animation(joints_3d, save_path='smplh_animation.mp4', fps=30, bone_connections=bone_connections, sphere_center_data=None, sphere_radius_meters=0.05, sphere_color='blue', sphere_alpha=0.3,  camera_pose_7d_data=None, ray_length=2.0, ray_color='green', text=''):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # center, max_range, elev, azim = compute_auto_view_params(joints_3d)
    # pdb.set_trace()
    center, max_range, elev, azim = compute_auto_view_params(joints_3d, sphere_center_data)
    num_bones = len(bone_connections)
    colors = plt.cm.get_cmap("tab20", num_bones)
    
    
    lines = [ax.plot([], [], [], lw=2, c=colors(idx))[0] for idx, _ in enumerate(bone_connections)]
    sphere_plot = None
    base_x_sphere, base_y_sphere, base_z_sphere = (None, None, None)
    if sphere_center_data is not None:
        # Generate sphere data (fixed for all frames, just repositioned)
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        base_x_sphere = sphere_radius_meters * np.outer(np.cos(u), np.sin(v))
        base_y_sphere = sphere_radius_meters * np.outer(np.sin(u), np.sin(v))
        base_z_sphere = sphere_radius_meters * np.outer(np.ones(np.size(u)), np.cos(v))


        # Initial position (will be updated in update function)
        
        initial_sphere_center = sphere_center_data[0] if sphere_center_data.ndim > 1 else sphere_center_data
        
        sphere_plot = ax.plot_surface(base_x_sphere + initial_sphere_center[0],
                                      base_y_sphere + initial_sphere_center[1],
                                      base_z_sphere + initial_sphere_center[2],
                                      color=sphere_color, alpha=sphere_alpha, linewidth=0)
    
    ray_plot = None
    if camera_pose_7d_data is not None:
        # Define camera's forward direction in its local frame (e.g., positive Z-axis)
        camera_forward_local = np.array([0., 0., 1.]) # Consistent with the update function
        
        # Get initial camera pose
        initial_camera_pose_7d = camera_pose_7d_data[0]
        q_xyzw = initial_camera_pose_7d[:4]
        t_xyz = initial_camera_pose_7d[4:]

        R_world_cam = R.from_quat(q_xyzw).as_matrix()
        
        # Calculate initial ray direction in world frame
        ray_direction_world = np.dot(R_world_cam, camera_forward_local)
        
        # Calculate initial ray endpoint
        ray_endpoint_world = t_xyz + ray_direction_world * ray_length
        
        # Plot the initial ray
        ray_plot = ax.plot([t_xyz[0], ray_endpoint_world[0]],
                           [t_xyz[1], ray_endpoint_world[1]],
                           [t_xyz[2], ray_endpoint_world[2]],
                           color=ray_color, lw=2, alpha=0.8)[0]

    def init():
        half_range = max_range / 2
        ax.set_xlim(center[0] - half_range, center[0] + half_range)
        ax.set_ylim(center[1] - half_range, center[1] + half_range)
        ax.set_zlim(center[2] - half_range, center[2] + half_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=elev, azim=azim)
        all_elements = list(lines)
        if sphere_plot:
            all_elements.append(sphere_plot)
        if ray_plot:
            all_elements.append(ray_plot)
        return all_elements

    def update(frame):
        joints = joints_3d[frame]
        for line, (i, j), c in zip(lines, bone_connections, range(num_bones)):
            line.set_data([joints[i, 0], joints[j, 0]], [joints[i, 1], joints[j, 1]])
            line.set_3d_properties([joints[i, 2], joints[j, 2]])
        
        if sphere_plot is not None:
            current_sphere_center = sphere_center_data[frame] if sphere_center_data.ndim > 1 else sphere_center_data
            
            x_shifted = base_x_sphere + current_sphere_center[0]
            y_shifted = base_y_sphere + current_sphere_center[1]
            z_shifted = base_z_sphere + current_sphere_center[2]

            # --- FIX STARTS HERE ---
            # Reconstruct the list of polygons for set_verts
            num_u, num_v = x_shifted.shape
            updated_verts = []
            for i in range(num_u - 1):
                for j in range(num_v - 1):
                    # Define the 4 vertices of the quadrilateral
                    p1 = [x_shifted[i, j], y_shifted[i, j], z_shifted[i, j]]
                    p2 = [x_shifted[i+1, j], y_shifted[i+1, j], z_shifted[i+1, j]]
                    p3 = [x_shifted[i+1, j+1], y_shifted[i+1, j+1], z_shifted[i+1, j+1]]
                    p4 = [x_shifted[i, j+1], y_shifted[i, j+1], z_shifted[i, j+1]]
                    updated_verts.append(np.array([p1, p2, p3, p4])) # Each element is a (4, 3) array

            sphere_plot.set_verts(updated_verts)
            # --- FIX ENDS HERE ---

        if ray_plot is not None:
            # Define camera's forward direction in its local frame (e.g., positive Z-axis)
            camera_forward_local = np.array([0., 0., 1.])
            
            # Extract current frame's camera pose
            current_camera_pose_7d = camera_pose_7d_data[frame]
            q_xyzw = current_camera_pose_7d[:4] # Quaternion (qx, qy, qz, qw)
            t_xyz = current_camera_pose_7d[4:]  # Translation (tx, ty, tz)

            # Convert quaternion to rotation matrix (Camera to World)
            R_world_cam = R.from_quat(q_xyzw).as_matrix()
            
            # Calculate ray direction in world frame
            ray_direction_world = np.dot(R_world_cam, camera_forward_local)
            
            # Calculate ray endpoint
            ray_endpoint_world = t_xyz + ray_direction_world * ray_length
            
            # Update the ray's data
            ray_plot.set_data([t_xyz[0], ray_endpoint_world[0]],
                              [t_xyz[1], ray_endpoint_world[1]])
            ray_plot.set_3d_properties([t_xyz[2], ray_endpoint_world[2]])

        ax.set_title(f"{text} \n Frame {frame + 1}/{len(joints_3d)}")

        all_elements = list(lines)
        if sphere_plot:
            all_elements.append(sphere_plot)
        if ray_plot:
            all_elements.append(ray_plot)
        return all_elements

    ani = FuncAnimation(fig, update, frames=len(joints_3d), init_func=init, blit=False, interval=1000 / fps)
    ani.save(save_path, fps=fps, dpi=150)
    plt.close()


# save_latest/my_humanml_trans_dec_bert_512_hdepic_v2_wo_text/samples_my_humanml_trans_dec_bert_512_hdepic_v2_wo_text_000050000_seed10/results.npy

# folder = 'save_latest/my_humanml_trans_dec_bert_512_hdepic_v1/samples_my_humanml_trans_dec_bert_512_hdepic_v1_000100000_seed10/results.npy'
# folder = 'save_latest/my_humanml_trans_dec_bert_512_hdepic_v2_wo_text/samples_my_humanml_trans_dec_bert_512_hdepic_v2_wo_text_000100000_seed10/results.npy'
folder = 'save_latest/my_humanml_trans_dec_bert_512_nymeria_final_V2/samples_my_humanml_trans_dec_bert_512_nymeria_final_V2_000600000_seed10/results.npy'
folder1 = 'save_latest/my_humanml_trans_dec_bert_512_hdepic_v1/samples_my_humanml_trans_dec_bert_512_hdepic_v1_000300000_seed10/results.npy'
folder2 = 'save_latest/my_humanml_trans_dec_bert_512_hdepic_v3_CA/samples_my_humanml_trans_dec_bert_512_hdepic_v3_CA_000220000_seed10/results.npy'
predictions = np.load(folder, allow_pickle=True)[None][0]
predictions1 = np.load(folder1, allow_pickle=True)[None][0]
predictions2 = np.load(folder2, allow_pickle=True)[None][0]
motions = predictions['motion'].transpose(0,3,1,2)
motions1 = predictions1['motion'].transpose(0,3,1,2)
motions2 = predictions2['motion'].transpose(0,3,1,2)
# pdb.set_trace()
gt = np.load('dataset/HDEPIC_test.npy', allow_pickle=True)[None][0]
name_list = gt['name_list']
motion = gt['data_dict'].keys()
# pdb.set_trace()
trans_matrix = np.linalg.inv(np.array([[1,0,0],[0,0,-1],[0,1,0]]))

text = predictions['text']
index = 700
name = name_list[index]
name = name[2:]
print(name)
gt_motion = np.load(f'../HumanML3D/joints_data_hdepic/{name}.npy')
plot_and_save_animation(np.dot(motions[index], trans_matrix), f'plots/prediction_pretrained_{index}.mp4', fps=20, text=text[index])
plot_and_save_animation(np.dot(motions1[index][:len(gt_motion)], trans_matrix), f'plots/prediction_addition_{index}.mp4', fps=20, text=text[index])
plot_and_save_animation(np.dot(motions2[index][:len(gt_motion)], trans_matrix), f'plots/prediction_CA_{index}.mp4', fps=20, text=text[index])
plot_and_save_animation(np.dot(gt_motion, trans_matrix), f'plots/gt_{index}.mp4', fps=20, text=text[index])
# pdb.set_trace()