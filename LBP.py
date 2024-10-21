import csv
import voxel3d as v3d
import plane_Fitting as pf
import numpy as np
from scipy.stats import ttest_ind
import random
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.neighbors import NearestNeighbors


def apply_dynamic_lbp_to_voxels(pointcloud, labels, voxel_labels, high_grass_label, low_grass_label, k_neighbors=6):
    """
    Applies Label Binary Pattern (LBP) to voxels labeled as high grass vs low grass by dynamically defining neighborhoods
    using K-Nearest Neighbors (KNN) from the actual data.
    """
    # Initialize storage for LBP patterns for each voxel
    lbp_patterns = {}

    # Use a KNN model to find nearest neighbors
    knn = NearestNeighbors(n_neighbors=k_neighbors)
    knn.fit(pointcloud)

    # Iterate through all unique voxel labels
    for voxel_label in np.unique(voxel_labels):
        # Get points and labels for the current voxel
        points_in_voxel = pointcloud[voxel_labels == voxel_label]
        labels_in_voxel = labels[voxel_labels == voxel_label]

        # Determine if this voxel is high grass or low grass
        voxel_type = 'high_grass' if high_grass_label in labels_in_voxel else 'low_grass' if low_grass_label in labels_in_voxel else None

        if voxel_type is None or len(points_in_voxel) == 0:
            continue  # Skip voxels that are neither high grass nor low grass

        # Get the centroid of the voxel for neighborhood comparison
        voxel_centroid = np.mean(points_in_voxel, axis=0).reshape(1, -1)

        # Find K nearest neighbors to the voxel centroid
        distances, indices = knn.kneighbors(voxel_centroid)

        # Calculate LBP for the current voxel by comparing with its nearest neighbors
        lbp_value = 0
        for i, neighbor_idx in enumerate(indices[0]):
            # Get neighbor point
            neighbor_point = pointcloud[neighbor_idx]

            # Compare height (z-coordinate) with the current voxel centroid
            height_diff = voxel_centroid[0][2] - neighbor_point[2]

            # If the height difference is positive, set the corresponding LBP bit to 1
            lbp_value |= (height_diff > 0) << i

        # Store the LBP pattern for this voxel
        if voxel_type not in lbp_patterns:
            lbp_patterns[voxel_type] = []

        lbp_patterns[voxel_type].append(lbp_value)

    return lbp_patterns


def process_voxels_with_dynamic_lbp(lidar_data, labels, voxel_labels, high_grass_label, low_grass_label, k_neighbors=6):
    """
    Process the voxels, apply dynamic LBP using K-Nearest Neighbors, and compare high grass vs low grass patterns.
    """
    # Apply dynamic LBP to voxels
    lbp_patterns = apply_dynamic_lbp_to_voxels(
        lidar_data, labels, voxel_labels, high_grass_label, low_grass_label, k_neighbors
    )

    # Compare patterns for high grass vs low grass
    if 'high_grass' in lbp_patterns and 'low_grass' in lbp_patterns:
        high_grass_lbp = lbp_patterns['high_grass']
        low_grass_lbp = lbp_patterns['low_grass']

        # Statistical comparison (e.g., using t-test)
        stat, p_value = ttest_ind(high_grass_lbp, low_grass_lbp)

        print(f"Dynamic LBP pattern comparison between high grass and low grass: t-statistic = {stat}, p-value = {p_value}")

        return high_grass_lbp, low_grass_lbp, stat, p_value
    else:
        print("No valid comparison between high grass and low grass.")
        return None


def test_dynamic_lbp_function(lidar_dir, labels_dir, csv_file, image_list, high_grass_label, low_grass_label, k_neighbors=6):
    """
    Test function to run dynamic LBP comparison on the given images and label types (high grass vs low grass).
    """
    # Create a directory for results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"test_dynamic_lbp_{timestamp}_{labels_dir}"
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each image and process them
    for prefix in image_list:
        # Load pointcloud and labels
        lidar_data, labels, label_metadata = pf.load_pointcloud_and_labels(prefix, lidar_dir, labels_dir, csv_file)

        # Apply z-threshold to pointcloud and labels (if needed in your processing)
        pointcloud, labels = pf.apply_threshold(lidar_data, labels, z_treshold=1)

        # Voxelize the entire point cloud
        voxel_labels, voxel_map, unique_voxel_labels = v3d.voxelize_point_cloud_2d(pointcloud, voxel_size=10)

        # Run the dynamic LBP process and comparison
        high_grass_lbp, low_grass_lbp, stat, p_value = process_voxels_with_dynamic_lbp(
            pointcloud, labels, voxel_labels, high_grass_label, low_grass_label, k_neighbors
        )

        if high_grass_lbp is not None and low_grass_lbp is not None:
            # Save the LBP patterns and statistical comparison results to CSV
            csv_filename = os.path.join(output_dir, f'lbp_comparison_{prefix}.csv')
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['High_Grass_LBP', 'Low_Grass_LBP', 'T-Statistic', 'P-Value'])

                for hg_lbp, lg_lbp in zip(high_grass_lbp, low_grass_lbp):
                    writer.writerow([hg_lbp, lg_lbp, stat, p_value])

            print(f"LBP comparison results for image {prefix} saved to {csv_filename}")

        # Create a simple histogram plot to visualize LBP patterns
        plt.figure(figsize=(10, 6))
        plt.hist(high_grass_lbp, bins=20, alpha=0.5, label='High Grass LBP', color='green')
        plt.hist(low_grass_lbp, bins=20, alpha=0.5, label='Low Grass LBP', color='orange')
        plt.title(f'LBP Pattern Comparison for {prefix}')
        plt.xlabel('LBP Pattern')
        plt.ylabel('Frequency')
        plt.legend()

        # Save the plot
        plot_filename = os.path.join(output_dir, f'lbp_histogram_{prefix}.png')
        plt.savefig(plot_filename)
        plt.close()

        print(f"Histogram of LBP patterns for image {prefix} saved to {plot_filename}")


if __name__ == "__main__":
    # Directories for the lidar and label data
    lidar_dir = 'goose_3d_val/lidar/val/2022-07-22_flight'
    labels_dir = 'goose_3d_val/labels/val/2022-07-22_flight'

    csv_file = 'goose_3d_val/goose_label_mapping.csv'

    # Define the labels for high grass and low grass (these should correspond to the label IDs in your dataset)
    high_grass_label = 51  # Example label ID for high grass
    low_grass_label = 50   # Example label ID for low grass

    # Image lists for comparison
    image_list = [
        '2022-07-22_flight__0071_1658494234334310308',
        '2022-07-22_flight__0072_1658494235988100385',
        '2022-07-22_flight__0073_1658494238675704025',
        '2022-07-22_flight__0075_1658494242083534022',
        '2022-07-22_flight__0077_1658494244047191404',
    ]

    # Run the dynamic LBP testing function
    test_dynamic_lbp_function(lidar_dir, labels_dir, csv_file, image_list, high_grass_label, low_grass_label, k_neighbors=10)
