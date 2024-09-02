import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def load_point_cloud(file_path):
    point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    print(f"Loaded point cloud with shape: {point_cloud.shape}")  # Debug print
    return point_cloud


def load_label(file_path):
    label = np.fromfile(file_path, dtype=np.uint32).reshape(-1)
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    print(f"Loaded labels with shape: {sem_label.shape}")  # Debug print
    return sem_label, inst_label


def get_patch_points_by_grid(point_cloud, x_min, x_max, y_min, y_max):
    # Filter points within the given x and y boundaries
    patch_mask = (
        (point_cloud[:, 0] >= x_min) & (point_cloud[:, 0] < x_max) &
        (point_cloud[:, 1] >= y_min) & (point_cloud[:, 1] < y_max)
    )
    patch_points = point_cloud[patch_mask]

    return patch_points


def visualize_patch_with_open3d(patch_points):
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(patch_points[:, :3])

    # Visualize
    o3d.visualization.draw_geometries([pcd])


# Example usage
point_cloud_path = '/Users/YehonatanMileguir/GOOSE/goose_3d_val/lidar/val/2022-12-07_aying_hills//2022-12-07_aying_hills__0000_1670420609181206687_vls128.bin'
label_path = '/Users/YehonatanMileguir/GOOSE/goose_3d_val/labels/val/2022-12-07_aying_hills/2022-12-07_aying_hills__0000_1670420609181206687_goose.label'

# Load the point cloud data
point_cloud = load_point_cloud(point_cloud_path)

# Define the patch boundaries (for example x from 100 to 200, y from 200 to 300)
x_min, x_max = 0, 50
y_min, y_max = 0, 50

# Extract points from the defined grid patch
patch_points = get_patch_points_by_grid(point_cloud, x_min, x_max, y_min, y_max)

# Visualize the patch
visualize_patch_with_open3d(patch_points)
