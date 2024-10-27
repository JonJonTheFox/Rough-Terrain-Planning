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


def voxelize_point_cloud_2d(point_cloud, voxel_size=10):
    # Calculate the 2D voxel grid coordinates (considering only x and y)
    voxel_indices = np.floor(point_cloud[:, :2] / voxel_size).astype(int)

    # Create unique voxel indices
    unique_voxel_indices = np.unique(voxel_indices, axis=0)
    voxel_labels = np.zeros(voxel_indices.shape[0], dtype=int)
    voxel_map = {}

    for label, unique_voxel in enumerate(unique_voxel_indices):
        mask = np.all(voxel_indices == unique_voxel, axis=1)
        voxel_labels[mask] = label
        voxel_map[label] = unique_voxel  # Map voxel label to its coordinates

    return voxel_labels, voxel_map



def visualize_with_open3d(point_cloud, sem_labels, voxel_labels=None, selected_voxels=None):
    # Get the original label colors
    label_colors = get_label_colors(sem_labels)

    if selected_voxels is not None and voxel_labels is not None:
        # Adjust colors based on whether the point belongs to a selected voxel
        visibility = np.isin(voxel_labels, selected_voxels)
        adjusted_colors = np.where(visibility[:, None], label_colors, np.array([0.5, 0.5, 0.5]))
    else:
        adjusted_colors = label_colors

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(adjusted_colors)

    # Visualize
    o3d.visualization.draw_geometries([pcd])


def voxelize_point_cloud_2d_with_label_filter(point_cloud, labels, voxel_size=10, min_points=10,
                                              proportion_threshold=0.9):
    """
    Voxelizes the point cloud in 2D (x, y) and filters the voxels based on the minimum number of points and the proportion of the majority label.

    Parameters:
    - point_cloud (numpy array): The point cloud data (N x 3 or N x 4).
    - labels (numpy array): Array of labels corresponding to each point.
    - voxel_size (int): Size of the voxels for the grid.
    - min_points (int): Minimum number of points required in a voxel to keep it.
    - proportion_threshold (float): Minimum proportion of the majority label in a voxel to keep it.

    Returns:
    - filtered_voxel_labels (numpy array): Array containing the voxel label for each point after filtering.
    - voxel_map (dict): Mapping of voxel labels to their 2D coordinates.
    - unique_voxel_labels (numpy array): Array of unique voxel labels after filtering.
    """

    # Calculate the 2D voxel grid coordinates (considering only x and y)
    voxel_indices = np.floor(point_cloud[:, :2] / voxel_size).astype(int)

    # Create unique voxel indices
    unique_voxel_indices, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)

    # Initialize storage for voxel labels
    voxel_labels = np.zeros(voxel_indices.shape[0], dtype=int)
    voxel_map = {}

    # Prepare the lists for the filtered results
    filtered_voxel_labels = []
    filtered_voxel_map = {}
    valid_voxel_labels = []

    # Iterate over each unique voxel
    for label, unique_voxel in enumerate(unique_voxel_indices):
        mask = (inverse_indices == label)

        # Points and labels in this voxel
        points_in_voxel = point_cloud[mask]
        labels_in_voxel = labels[mask]

        # Skip voxels with fewer points than the threshold
        if len(points_in_voxel) < min_points:
            continue

        # Get the majority label and its proportion
        unique_labels, counts = np.unique(labels_in_voxel, return_counts=True)
        majority_label = unique_labels[np.argmax(counts)]
        majority_proportion = counts[np.argmax(counts)] / len(labels_in_voxel)

        # Skip voxels where the majority label does not meet the threshold
        if majority_proportion < proportion_threshold:
            continue

        # Assign voxel label and save to the map
        voxel_labels[mask] = label
        filtered_voxel_map[label] = unique_voxel
        valid_voxel_labels.append(label)

    # Convert valid voxel labels and filtered voxel map to numpy arrays
    filtered_voxel_labels = np.array(
        [voxel_labels[i] for i in range(len(voxel_labels)) if voxel_labels[i] in valid_voxel_labels])
    unique_voxel_labels = np.array(valid_voxel_labels)

    return filtered_voxel_labels, filtered_voxel_map, unique_voxel_labels


# Example usage
point_cloud_path = '/Users/YehonatanMileguir/GOOSE/goose_3d_val/lidar/val/2022-12-07_aying_hills//2022-12-07_aying_hills__0000_1670420609181206687_vls128.bin'
label_path = '/Users/YehonatanMileguir/GOOSE/goose_3d_val/labels/val/2022-12-07_aying_hills/2022-12-07_aying_hills__0000_1670420609181206687_goose.label'

# Load the point cloud data
point_cloud = load_point_cloud(point_cloud_path)

# Load the labels
sem_labels, inst_labels = load_label(label_path)

# Voxelize the point cloud
voxel_labels = voxelize_point_cloud_2d(point_cloud, voxel_size=50)

# Visualize only certain voxels, for example voxels with indices 1 and 2
selected_voxels = list(range(5, 10))
visualize_with_open3d(point_cloud, sem_labels, voxel_labels=voxel_labels, selected_voxels=selected_voxels)
