import numpy as np
import pandas as pd
import open3d as o3d



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

# The following paths require the "testing data" to be at the same level as the script
prefix = '2022-07-22_flight__0071_1658494234334310308'
lidar_dir = 'goose_3d_val/lidar/val/2022-07-22_flight'
labels_dir = 'goose_3d_val/labels/val/2022-07-22_flight'
csv_file = 'goose_3d_val/goose_label_mapping.csv'

lidar_data, labels, label_metadata = load_pointcloud_and_labels(prefix, lidar_dir, labels_dir, csv_file)


def visualize_selected_points(lidar_data, labels, label_metadata, plane=None):
    """
    Visualize the point cloud data with the given labels and an optional plane.

    :param lidar_data: The point cloud data (N x 4 array where last column is label).
    :param labels: The label data corresponding to each point.
    :param label_metadata: Metadata containing class names and colors.
    :param plane: Optional. A tuple (a, b, c, d) representing the plane equation ax + by + cz + d = 0.
    """
    # Convert label keys to RGB colors
    label_to_color = dict(zip(label_metadata['label_key'], label_metadata['hex']))

    # Create colors array by mapping labels to their corresponding colors
    colors = np.array([hex_to_rgb(label_to_color.get(label, '#000000')) for label in labels])

    assert len(lidar_data) == len(colors), "Number of points and colors do not match"

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
            triangles.append([i * 20 + j + 1, (i + 1) * 20 + j + 1, (i + 1) * 20 + j])
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))

    # Compute vertex normals for shading
    mesh.compute_vertex_normals()

    # Set the color of the plane
    mesh.paint_uniform_color([0.7, 0.8, 1.0])

    return mesh

# Create a dictionary to map label_key to class_name
label_to_class = dict(zip(label_metadata['label_key'], label_metadata['class_name']))

# Map each label to its class name
mapped_labels = [label_to_class.get(label, 'Unknown') for label in labels]

[print(f"Point {i}: {lidar_data[i]} -> Class: {mapped_labels[i]}") for i in range(5)]

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


print(calculate_label_distribution(labels, label_metadata).head(10))

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

bbox_min = [2, 0, -4]  # Define the minimum x, y, z coordinates of the bounding box
bbox_max = [8, 6, 0]  # Define the maximum x, y, z coordinates of the bounding box

selected_points, selected_labels = extract_selected_points(lidar_data, labels, bbox_min, bbox_max)





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


keep_labels = [23]  # 50 = Low grass

filtered_points, filtered_labels = filter_labels(selected_points, selected_labels, keep_labels)

print(f"Original number of points: {len(selected_points)}")
print(f"Filtered number of points: {len(filtered_points)}")
print(f"Filtered labels: {np.unique(filtered_labels)}")


def fit_plane_least_squares(points):
    """
    Fit a plane to a set of 3D points using the least squares method.

    :param points: numpy array of shape (n, 3) where n is the number of points
    :return: numpy array of shape (4,) representing the plane equation coefficients [a, b, c, d]
             where ax + by + cz + d = 0
    """
    # Ensure we have at least 3 points
    if points.shape[0] < 3:
        raise ValueError("At least 3 points are required to fit a plane")

    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Center the points by subtracting the centroid
    centered_points = points - centroid

    cov_matrix = np.cov(centered_points.T)

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # The normal vector of the plane is the eigenvector corresponding to the smallest eigenvalue
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]

    # Ensure the normal vector points "upward"
    if normal_vector[2] < 0:
        normal_vector = -normal_vector

    # Calculate the d coefficient
    d = -np.dot(normal_vector, centroid)

    # Return the coefficients of the plane equation ax + by + cz + d = 0
    return np.append(normal_vector, d)


plane = fit_plane_least_squares(filtered_points[:, :3])
print(f"Fitted plane coefficients: {plane}")

visualize_selected_points(filtered_points, filtered_labels, label_metadata, plane)