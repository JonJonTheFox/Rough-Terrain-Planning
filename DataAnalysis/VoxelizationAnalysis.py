import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Your existing function to load the point cloud and labels
def load_pointcloud_and_labels(prefix, lidar_dir, labels_dir, csv_file, lidar_suffix='vls128', label_suffix='goose'):
    lidar_file = f"{lidar_dir}/{prefix}_{lidar_suffix}.bin"
    label_file = f"{labels_dir}/{prefix}_{label_suffix}.label"

    print(f"Loading LiDAR data from {lidar_file}")
    lidar_data = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    print(f"Loading label data from {label_file}")
    labels = np.fromfile(label_file, dtype=np.uint32)

    print(f"Loading label metadata from {csv_file}")
    label_metadata = pd.read_csv(csv_file)

    print(f"Loaded {len(lidar_data)} points and {len(labels)} labels")
    return lidar_data, labels, label_metadata


# Function to voxelize point cloud (splitting into grid)
def voxelize_pointcloud(lidar_data, voxel_size=10):
    """
    Divide the point cloud into voxels of the specified size.
    Assigns a voxel label to each point based on its spatial location.
    """
    print(f"Voxelizing point cloud with voxel size {voxel_size}")
    voxel_indices = np.floor(lidar_data[:, :3] / voxel_size).astype(int)  # Get voxel indices
    unique_voxels, voxel_labels = np.unique(voxel_indices, axis=0, return_inverse=True)  # Assign unique voxel IDs

    print(f"Generated {len(unique_voxels)} unique voxels")
    return voxel_labels, unique_voxels


# The existing preprocessing function for voxels
def preprocess_voxels(voxel_labels, pointcloud, labels, min_len=10, proportion_threshold=0.7):
    print(f"Preprocessing voxels with min_len={min_len}, proportion_threshold={proportion_threshold}")

    voxel_labels = np.array(voxel_labels)
    labels = np.array(labels)

    map_to_majority = {}
    voxel_pointclouds = {}
    voxel_ids_after_preprocessing = set()

    unique_voxel_labels = np.unique(voxel_labels)
    print(f"Processing {len(unique_voxel_labels)} unique voxel labels")

    # Keep track of label distribution per voxel for analysis
    majority_count = 0

    for vox_id in unique_voxel_labels:
        mask = voxel_labels == vox_id
        voxel_points = pointcloud[mask]
        voxel_lidar_labels = labels[mask]

        if len(voxel_points) < min_len:
            continue

        unique, counts = np.unique(voxel_lidar_labels, return_counts=True)
        majority_index = np.argmax(counts)
        majority_label = unique[majority_index]
        majority_proportion = counts[majority_index] / np.sum(counts)

        # Print some details for inspection
        if majority_proportion == 1.0:
            print(f"Voxel {vox_id} has 100% majority label: {majority_label} with {len(voxel_points)} points.")
            majority_count += 1

        if majority_proportion < proportion_threshold:
            continue

        map_to_majority[vox_id] = majority_label
        voxel_pointclouds[vox_id] = voxel_points
        voxel_ids_after_preprocessing.add(vox_id)

    print(f"After filtering: {len(voxel_ids_after_preprocessing)} voxels remain")
    print(f"Total voxels with 100% majority label: {majority_count}")
    return map_to_majority, voxel_pointclouds, voxel_ids_after_preprocessing


# Function to test different thresholds
def test_voxel_thresholds(voxel_labels, pointcloud, labels, min_lengths, proportion_thresholds):
    result_grid = np.zeros((len(min_lengths), len(proportion_thresholds)))

    for i, min_len in enumerate(min_lengths):
        for j, proportion_threshold in enumerate(proportion_thresholds):
            print(f"Testing combination min_len={min_len}, proportion_threshold={proportion_threshold:.2f}")
            _, _, voxel_ids_after_preprocessing = preprocess_voxels(
                voxel_labels, pointcloud, labels, min_len=min_len, proportion_threshold=proportion_threshold)
            usable_percentage = len(voxel_ids_after_preprocessing) / len(np.unique(voxel_labels)) * 100
            result_grid[i, j] = usable_percentage
            print(f"Usable percentage: {usable_percentage:.2f}%")

    return result_grid


# Plot the results
def plot_voxel_threshold_results(min_lengths, proportion_thresholds, result_grid):
    print("Plotting the results")
    plt.figure(figsize=(10, 8))
    plt.imshow(result_grid, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(label='% Usable Voxels')

    plt.xticks(np.arange(len(proportion_thresholds)), [f'{t:.2f}' for t in proportion_thresholds])
    plt.yticks(np.arange(len(min_lengths)), min_lengths)

    plt.xlabel('Majority Proportion Threshold')
    plt.ylabel('Minimum Length of Points')
    plt.title('Percentage of Usable Voxels for Different Thresholds')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Directories for the LiDAR data
    lidar_dir1 = 'goose_3d_val/lidar/val/2022-07-22_flight'
    labels_dir1 = 'goose_3d_val/labels/val/2022-07-22_flight'
    csv_file = '../goose_3d_val/goose_label_mapping.csv'

    # LiDAR file prefixes
    image_list1 = [
        '2022-07-22_flight__0071_1658494234334310308',
        '2022-07-22_flight__0072_1658494235988100385',
        '2022-07-22_flight__0073_1658494238675704025',
        '2022-07-22_flight__0075_1658494242083534022',
        '2022-07-22_flight__0077_1658494244047191404',
    ]

    # Load LiDAR data and labels for the first file in the list
    prefix = image_list1[0]
    lidar_data, labels, label_metadata = load_pointcloud_and_labels(prefix, lidar_dir1, labels_dir1, csv_file)

    # Voxelizing the point cloud with a smaller voxel size
    voxel_labels, unique_voxels = voxelize_pointcloud(lidar_data[:, :3], voxel_size=10)

    # Test parameters
    min_lengths = list(range(5, 16))  # Test min length from 5 to 15
    proportion_thresholds = np.linspace(0.1, 1.0, 11)  # Test proportion thresholds from 0.5 to 1.0

    # Run the experiment
    result_grid = test_voxel_thresholds(voxel_labels, lidar_data, labels, min_lengths, proportion_thresholds)

    # Plot the results
    plot_voxel_threshold_results(min_lengths, proportion_thresholds, result_grid)
