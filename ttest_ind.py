import csv
import voxel3d as v3d
import plane_Fitting as pf
import numpy as np
from scipy.stats import ttest_ind
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime


def process_image_with_majority_label_comparison(prefix, lidar_dir, labels_dir, csv_file, z_threshold=1, voxel_size=15):
    """
    Processes a single image and returns a dictionary of RMSE values comparing all points in the voxel
    with those of the majority label, excluding majority labels > 100.
    Also extracts the name of the majority label.
    """
    # Load pointcloud and labels
    lidar_data, labels, label_metadata = pf.load_pointcloud_and_labels(prefix, lidar_dir, labels_dir, csv_file)

    # Apply z-threshold to pointcloud and labels
    pointcloud, labels = pf.apply_threshold(lidar_data, labels, z_threshold)

    # Initialize dictionaries to store RMSE values
    rmse_total_by_label = {}
    rmse_majority_by_label = {}

    # Iterate over each unique label in the labels
    for label in np.unique(labels):
        # Filter the point cloud based on the current label
        filtered_pointcloud = pointcloud[labels == label]

        # Voxelize the filtered point cloud
        voxel_labels_, voxel_map_, unique_voxel_labels_ = v3d.voxelize_point_cloud_2d(filtered_pointcloud,
                                                                                      voxel_size=voxel_size)

        labels_in_voxelized_cloud = labels[labels == label][
                                    :len(voxel_labels_)]  # Align the labels with voxelized points

        # Compute the plane for each voxel and the associated RMSE for all points
        voxel_planes_, rmse_total = pf.compute_voxel_planes(filtered_pointcloud, voxel_labels_)

        for voxel_label, rmse_total_value in rmse_total.items():
            # All points RMSE in the voxel
            points_in_voxel = filtered_pointcloud[voxel_labels_ == voxel_label]
            labels_in_voxel = labels_in_voxelized_cloud[voxel_labels_ == voxel_label]

            if len(points_in_voxel) != len(labels_in_voxel):
                continue  # Skip if there's a mismatch between points and labels

            # Find the majority label in the voxel
            majority_label = np.bincount(labels_in_voxel).argmax()

            # Filter out majority labels greater than 100
            if majority_label > 100:
                continue

            # Get the name of the majority label using label_metadata
            majority_label_name = label_metadata[label_metadata['label_key'] == majority_label]['class_name'].values[0]

            # Get points that belong to the majority label
            majority_label_points = points_in_voxel[labels_in_voxel == majority_label]

            # Compute RMSE for the majority label points
            if len(majority_label_points) >= 3:
                majority_voxel_planes_, rmse_majority_value = pf.fit_plane_least_squares(majority_label_points)

                # Store the RMSE for total points and majority label points, along with the label name
                if majority_label_name not in rmse_total_by_label:
                    rmse_total_by_label[majority_label_name] = []
                    rmse_majority_by_label[majority_label_name] = []

                rmse_total_by_label[majority_label_name].append(rmse_total_value)
                rmse_majority_by_label[majority_label_name].append(rmse_majority_value)

    return rmse_total_by_label, rmse_majority_by_label


def run_experiment_compare_majority_total(lidar_dir1, labels_dir1, csv_file, image_list1, iterations=1):
    """
    Runs the experiment comparing RMSE values for total points and majority label points,
    and generates separate plots and data for each majority label name, excluding labels > 100.
    """
    # Create a directory for results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"experiment_compare_majority_total_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize dictionaries to store RMSE values across all iterations and images
    all_rmse_total_by_label = {}
    all_rmse_majority_by_label = {}

    for i in range(iterations):
        print(f"Running experiment iteration {i + 1}/{iterations}...")

        # Process images and gather RMSE data for each majority label
        for prefix in image_list1:
            rmse_total_by_label, rmse_majority_by_label = process_image_with_majority_label_comparison(
                prefix, lidar_dir1, labels_dir1, csv_file)

            # Combine RMSE values across images for each majority label
            for label_name, rmse_values in rmse_total_by_label.items():
                if label_name not in all_rmse_total_by_label:
                    all_rmse_total_by_label[label_name] = []
                    all_rmse_majority_by_label[label_name] = []

                all_rmse_total_by_label[label_name].extend(rmse_values)
                all_rmse_majority_by_label[label_name].extend(rmse_majority_by_label[label_name])

    # Create separate plots and CSVs for each majority label name
    for label_name in all_rmse_total_by_label:
        # Create a separate CSV for this label
        csv_filename = os.path.join(output_dir, f'rmse_comparison_majority_label_{label_name}.csv')
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['RMSE_Total', 'RMSE_Majority'])
            for rmse_total_value, rmse_majority_value in zip(all_rmse_total_by_label[label_name],
                                                             all_rmse_majority_by_label[label_name]):
                writer.writerow([rmse_total_value, rmse_majority_value])

        print(f"RMSE comparison for majority label {label_name} has been saved to {csv_filename}")

        # Create a violin plot comparing total and majority RMSE for this label
        data = {
            'RMSE': all_rmse_total_by_label[label_name] + all_rmse_majority_by_label[label_name],
            'Type': ['Total'] * len(all_rmse_total_by_label[label_name]) + ['Majority'] * len(
                all_rmse_majority_by_label[label_name])
        }
        plt.figure(figsize=(8, 6))
        sns.violinplot(x='Type', y='RMSE', data=data)
        plt.title(f'Violin Plot of RMSE Comparison for Majority Label {label_name}')
        plt.ylabel('RMSE')

        # Save the plot
        plot_filename = os.path.join(output_dir, f'violin_plot_comparison_majority_label_{label_name}.png')
        plt.savefig(plot_filename)
        plt.close()

        print(f"Violin plot for majority label {label_name} has been saved to {plot_filename}")


if __name__ == "__main__":
    # Directories for first set of images
    lidar_dir1 = 'goose_3d_val/lidar/val/2022-12-07_aying_hills'
    labels_dir1 = 'goose_3d_val/labels/val/2022-12-07_aying_hills'

    csv_file = 'goose_3d_val/goose_label_mapping.csv'

    # Image lists for comparison
    image_list1 = [
        '2022-12-07_aying_hills__0006_1670420708448844860',
        '2022-12-07_aying_hills__0009_1670420878948219746',
        '2022-12-07_aying_hills__0010_1670420972132205304',
        '2022-12-07_aying_hills__0011_1670420979760256580',
        '2022-12-07_aying_hills__0012_1670420985739069345',
    ]

    # Run the experiment comparing RMSE values for total points and majority label points
    run_experiment_compare_majority_total(lidar_dir1, labels_dir1, csv_file, image_list1, iterations=1)



