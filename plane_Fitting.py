import numpy as np
import pandas as pd
import open3d as o3d
import random
import matplotlib.pyplot as plt
import os
import subprocess


def load_pointcloud_and_labels(prefix, lidar_dir, labels_dir, csv_file, lidar_suffix='vls128', label_suffix='goose'):
    """
    Load LIDAR point cloud data, corresponding labels, and label metadata.

    :param prefix: Filename prefix for the LIDAR and label files
    :param lidar_dir: Directory where LIDAR data is stored
    :param labels_dir: Directory where label data is stored
    :param csv_file: CSV file containing label metadata
    :param lidar_suffix: Suffix for LIDAR file
    :param label_suffix: Suffix for label file
    :return: Tuple containing LIDAR data, labels, and label metadata
    """
    lidar_file = f"{lidar_dir}/{prefix}_{lidar_suffix}.bin"
    label_file = f"{labels_dir}/{prefix}_{label_suffix}.label"

    lidar_data = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    labels = np.fromfile(label_file, dtype=np.uint32)

    label_metadata = pd.read_csv(csv_file)

    return lidar_data, labels, label_metadata


def random_hex_color():
    """
    Generate a random hex color string.

    :return: A random hex color string (e.g., '#ff0000').
    """
    return f"#{random.randint(0, 0xFFFFFF):06x}"


def visualize_selected_points(lidar_data, labels, label_metadata=None, plane=None):
    """
    Visualize the point cloud data with the given labels and an optional plane.

    :param lidar_data: The point cloud data (N x 4 array where last column is label).
    :param labels: The label data corresponding to each point.
    :param label_metadata: Metadata containing class names and colors.
    :param plane: Optional. A tuple (a, b, c, d) representing the plane equation ax + by + cz + d = 0.
    """
    # Convert label keys to RGB colors
    if label_metadata is not None:
        label_to_color = dict(
            zip(label_metadata['label_key'], label_metadata['hex']))
    else:
        label_to_color = {}
        unique_labels = np.unique(labels)
        for label in unique_labels:
            label_to_color[label] = random_hex_color()

    # Create colors array by mapping labels to their corresponding colors
    colors = np.array(
        [hex_to_rgb(label_to_color.get(label, '#000000')) for label in labels])

    assert len(lidar_data) == len(
        colors), "Number of points and colors do not match"

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_data[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    if plane is not None:
        plane_mesh = create_plane_mesh(plane, lidar_data)
        vis.add_geometry(plane_mesh)

    vis.run()
    vis.destroy_window()


def hex_to_rgb(value):
    """
    Convert a hex color string to an RGB tuple.

    :param value: Hex color string (e.g., '#ff0000').
    :return: A tuple representing the RGB color normalized to [0, 1].
    """
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) / 255.0 for i in range(0, lv, lv // 3))


def create_plane_mesh(plane, points):
    """
    Create a mesh representing the fitted plane.

    :param plane: A tuple (a, b, c, d) representing the plane equation ax + by + cz + d = 0.
    :param points: The point cloud data used to determine the extent of the plane visualization.
    :return: An Open3D TriangleMesh object representing the plane.
    """
    a, b, c, d = plane

    # Determine the bounding box of the points
    min_bound = np.min(points[:, :3], axis=0)
    max_bound = np.max(points[:, :3], axis=0)

    # Create a grid of points on the plane
    xx, yy = np.meshgrid(np.linspace(min_bound[0], max_bound[0], 20),
                         np.linspace(min_bound[1], max_bound[1], 20))
    zz = (-a * xx - b * yy - d) / c

    # Create vertices for the mesh
    vertices = np.stack((xx.flatten(), yy.flatten(), zz.flatten()), axis=-1)

    # Create the mesh object
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    # Define triangles for the mesh
    triangles = []
    for i in range(19):
        for j in range(19):
            triangles.append([i * 20 + j, i * 20 + j + 1, (i + 1) * 20 + j])
            triangles.append([i * 20 + j + 1, (i + 1) *
                             20 + j + 1, (i + 1) * 20 + j])
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))

    # Compute vertex normals for shading
    mesh.compute_vertex_normals()

    # Set the color of the plane
    mesh.paint_uniform_color([0.7, 0.8, 1.0])

    return mesh


def calculate_label_distribution(labels, label_metadata):
    """
    Calculate and print the distribution of labels in the point cloud data.

    :param labels: The label data corresponding to each point.
    :param label_metadata: The label metadata DataFrame containing class names and label keys.
    """
    # Calculate the distribution of labels in the provided data
    label_counts = pd.Series(labels).value_counts().reset_index()
    label_counts.columns = ['label_key', 'count']

    # Merge with label metadata to get the class names
    label_distribution = pd.merge(label_counts,
                                  label_metadata[['label_key', 'class_name']],
                                  on='label_key',
                                  how='left')

    # Sort the distribution by the most common labels
    label_distribution = label_distribution[['label_key', 'class_name', 'count']].sort_values(by='count',
                                                                                              ascending=False)

    return label_distribution


def extract_selected_points(lidar_data, labels, bbox_min, bbox_max):
    """
    Extract points within a bounding box.

    :param lidar_data: The original LiDAR data (including x, y, z, intensity).
    :param labels: The label data corresponding to each point.
    :param bbox_min: The minimum x, y, z coordinates of the bounding box.
    :param bbox_max: The maximum x, y, z coordinates of the bounding box.
    :return: The selected points and corresponding labels.
    """
    # Select points within the bounding box
    selected_indices = np.where(
        (lidar_data[:, 0] >= bbox_min[0]) & (lidar_data[:, 0] <= bbox_max[0]) &
        (lidar_data[:, 1] >= bbox_min[1]) & (lidar_data[:, 1] <= bbox_max[1]) &
        (lidar_data[:, 2] >= bbox_min[2]) & (lidar_data[:, 2] <= bbox_max[2])
    )[0]
    selected_points = lidar_data[selected_indices]
    selected_labels = labels[selected_indices]

    # Debugging: Print the number of selected points
    print(f"Number of points in the bounding box: {len(selected_points)}")

    return selected_points, selected_labels


def filter_labels(lidar_data, labels, keep_labels):
    """
    Filter out all points from the LiDAR data whose labels are not in the keep_labels list.

    :param lidar_data: The original LiDAR data (N x 4 array where last column is label).
    :param labels: The label data corresponding to each point.
    :param keep_labels: List of labels to keep.

    :return: Filtered lidar_data and labels containing only the points with labels in keep_labels.
    """
    keep_labels_set = set(keep_labels)

    is_in_keep_labels = np.isin(labels, list(keep_labels_set))

    filtered_lidar_data = lidar_data[is_in_keep_labels]
    filtered_labels = labels[is_in_keep_labels]

    # Debugging: Print label information
    print(f"Labels to keep: {keep_labels_set}")
    print(f"Original number of points: {len(lidar_data)}")
    print(f"Number of points after filtering: {len(filtered_lidar_data)}")
    print(f"Unique labels in filtered data: {np.unique(filtered_labels)}")

    return filtered_lidar_data, filtered_labels


def fit_plane_least_squares(points):
    """
    Fit a plane to a set of 3D points using the least squares method and compute the Root Mean Squared Error (RMSE).

    :param points: numpy array of shape (n, 3) where n is the number of points
    :return: tuple (plane_coefficients, average_residual)
             - plane_coefficients: numpy array of shape (4,) representing the plane equation coefficients [a, b, c, d]
               where ax + by + cz + d = 0
             - RMSE: Root Mean Squared Error
    """
    # Ensure we have at least 3 points
    if points.shape[0] < 3:
        raise ValueError("At least 3 points are required to fit a plane")

    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Center the points by subtracting the centroid
    centered_points = points - centroid

    # Compute the covariance matrix of the centered points
    cov_matrix = np.cov(centered_points.T)

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # The normal vector of the plane is the eigenvector corresponding to the smallest eigenvalue
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]

    # Ensure the normal vector points "upward"
    if normal_vector[2] < 0:
        normal_vector = -normal_vector

    # Calculate the d coefficient
    d = -np.dot(normal_vector, centroid)

    # The plane equation is now given by: ax + by + cz + d = 0, where [a, b, c] is the normal_vector
    plane_coefficients = np.append(normal_vector, d)

    # Calculate the residuals (perpendicular distance of each point from the plane)
    distances = np.abs(np.dot(points, normal_vector) + d) / \
        np.linalg.norm(normal_vector)

    # Calculate the average residual
    MSE = np.mean(distances ** 2)
    RMSE = np.sqrt(MSE)

    return plane_coefficients, RMSE


def compute_voxel_planes(lidar_data, voxel_labels):
    point_cloud = lidar_data[:, 0:3]
    voxel_planes = dict()
    rmse = dict()
    for label in np.unique(voxel_labels, axis=0):
        points = point_cloud[voxel_labels == label]
        try:
            voxel_planes[label], rmse[label] = fit_plane_least_squares(
                points)

        except Exception as e:
            print(f'Error fitting plane for voxel {label}: {e}')

    return voxel_planes, rmse


def plot_voxel_map(voxel_map, value_map, save_and_open=False, output_file='voxel_map.png', dpi=300):
    # Extract coordinates from voxel_map
    voxel_coords = np.array(list(voxel_map.values()))  # 2D coordinates (x, y)

    # Get the interesting values from the value_map, using NaN for missing values
    voxel_values = np.array([value_map.get(key, np.nan)
                            for key in voxel_map.keys()])

    # Determine grid size based on voxel coordinates
    x_min, y_min = voxel_coords.min(axis=0)
    x_max, y_max = voxel_coords.max(axis=0)

    # Create a 2D grid for the image (initialize with NaNs)
    grid = np.full((x_max - x_min + 1, y_max - y_min + 1), np.nan)

    # Fill the grid with values from the value_map
    for (x, y), value in zip(voxel_coords, voxel_values):
        # Shift coordinates to fit in the grid
        grid[x - x_min, y - y_min] = value

    # Create a masked array to handle NaN values
    masked_grid = np.ma.masked_invalid(grid)

    # Plot the 2D image
    plt.figure(figsize=(12, 10), dpi=dpi)
    im = plt.imshow(masked_grid.T, cmap='viridis', origin='lower', extent=[
                    x_min-0.5, x_max+0.5, y_min-0.5, y_max+0.5], interpolation='nearest')
    cbar = plt.colorbar(im, label='Values', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    plt.title('2D Visualization of Voxel Map Values', fontsize=16)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)

    # Add text annotations only for N/A values
    for (x, y), value in zip(voxel_coords, voxel_values):
        if np.isnan(value):
            pass
            # plt.text(x, y, 'N/A', ha='center', va='center', color='red', fontweight='bold', fontsize=8)

    plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
    plt.tight_layout()

    if save_and_open:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        open_image(output_file)
    else:
        plt.show()
        plt.close()


def open_image(file_path):
    """
    Open the image file with the default application.
    """
    if os.name == 'nt':  # For Windows
        os.startfile(file_path)
    elif os.name == 'posix':  # For macOS and Linux
        try:
            subprocess.call(('open', file_path))  # macOS
        except:
            subprocess.call(('xdg-open', file_path))  # Linux


def apply_threshold(pointcloud, labels, z_treshold):
    ground_mask = pointcloud[:, 2] < z_treshold
    pointcloud_ = pointcloud[ground_mask]
    labels_ = labels[ground_mask]

    # Debugging: Check label distribution before and after thresholding
    print("Label distribution before threshold:")
    print(np.unique(labels, return_counts=True))

    print("Label distribution after threshold:")
    print(np.unique(labels_, return_counts=True))

    print(
        f"Removed {len(pointcloud) - len(pointcloud_)} points below z-threshold {z_treshold}, remaining {len(pointcloud_)} points")
    print(f"This represents {100 * (len(pointcloud_) / len(pointcloud)):.2f}% of the original point cloud data")
    return pointcloud_, labels_

