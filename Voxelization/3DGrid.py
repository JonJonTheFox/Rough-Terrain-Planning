import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def load_point_cloud(file_path):
    point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    print(f"Loaded point cloud with shape: {point_cloud.shape}")  # Debug print
    return point_cloud

def get_label_colors(sem_labels):
    unique_labels = np.unique(sem_labels)
    num_colors = len(unique_labels)
    colormap = plt.get_cmap('hsv')

    # Initialize an array to store the colors
    colors = np.zeros((sem_labels.shape[0], 3))

    for i, label in enumerate(unique_labels):
        color = colormap(i / num_colors)[:3]  # Get RGB values from the colormap
        colors[sem_labels == label] = color  # Assign color to corresponding points

    return colors


def load_label(file_path):
    label = np.fromfile(file_path, dtype=np.uint32).reshape(-1)
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    print(f"Loaded labels with shape: {sem_label.shape}")  # Debug print
    return sem_label, inst_label


def get_patch_points_and_labels_by_grid(point_cloud, sem_labels, x_index, y_index, patch_size=50):
    # Calculate the boundaries of the patch
    x_min = x_index * patch_size
    x_max = (x_index + 1) * patch_size
    y_min = y_index * patch_size
    y_max = (y_index + 1) * patch_size

    # Filter points within the calculated patch boundaries
    patch_mask = (
        (point_cloud[:, 0] >= x_min) & (point_cloud[:, 0] < x_max) &
        (point_cloud[:, 1] >= y_min) & (point_cloud[:, 1] < y_max)
    )
    patch_points = point_cloud[patch_mask]
    patch_labels = sem_labels[patch_mask]

    return patch_points, patch_labels


def visualize_with_open3d(point_cloud, sem_labels):
    # Get the original label colors
    label_colors = get_label_colors(sem_labels)

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(label_colors)

    # Visualize
    o3d.visualization.draw_geometries([pcd])


# Example usage
point_cloud_path = 'goose_3d_val/lidar/val/2022-12-07_aying_hills/2022-12-07_aying_hills__0000_1670420609181206687_vls128.bin'
label_path = 'goose_3d_val/labels/val/2022-12-07_aying_hills/2022-12-07_aying_hills__0000_1670420609181206687_goose.label'

# Load the point cloud data
point_cloud = load_point_cloud(point_cloud_path)

# Load the semantic labels
sem_labels, inst_labels = load_label(label_path)

# Define the grid indices (e.g., x=0, y=0 corresponds to a patch from x=[0,50), y=[0,50))
x_index, y_index = 0, 0  # Modify these indices to select different patches

# Extract points and labels from the defined grid patch
patch_points, patch_labels = get_patch_points_and_labels_by_grid(point_cloud, sem_labels, x_index, y_index, patch_size=100)

# Visualize the patch with semantic labels
visualize_with_open3d(patch_points, patch_labels)





