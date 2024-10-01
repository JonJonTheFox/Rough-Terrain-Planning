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


def process_image(prefix, lidar_dir, labels_dir, csv_file, target_class=None, z_threshold=1, voxel_size=15, num_voxels=5):
    """
    Processes a single image and returns RMSE values for the target class in each voxel,
    limiting the number of voxels per label.
    """
    # Load the pointcloud and label metadata
    lidar_data, lidar_labels, label_metadata = pf.load_pointcloud_and_labels(prefix, lidar_dir, labels_dir, csv_file)

    # Apply z-threshold to pointcloud and labels
    pointcloud, labels = pf.apply_threshold(lidar_data, lidar_labels, z_threshold)

    # If a target class is provided, filter for that class
    if target_class:
        # Get the label_key associated with the target class
        label_key = label_metadata[label_metadata['class_name'] == target_class]['label_key'].values[0]

        # Apply the filter on both pointcloud and labels to ensure they remain aligned
        valid_indices = (labels == label_key)
        pointcloud = pointcloud[valid_indices]
        labels = labels[valid_indices]

    # Ensure pointcloud and labels still match after filtering
    assert len(pointcloud) == len(labels), "Filtered pointcloud and labels must have the same length."

    # Initialize a list to store RMSE data for each label
    data = []

    # Iterate over each unique label in the labels
    for label in np.unique(labels):
        # Filter the point cloud based on the current label
        filtered_pointcloud = pointcloud[labels == label]

        # Voxelize the filtered point cloud
        voxel_labels_, voxel_map_, unique_voxel_labels_ = v3d.voxelize_point_cloud_2d(filtered_pointcloud,
                                                                                      voxel_size=voxel_size)

        # Compute the plane for each voxel and the associated RMSE
        voxel_planes_, rmse_ = pf.compute_voxel_planes(filtered_pointcloud, voxel_labels_)

        # Skip if no RMSE data is available
        if len(rmse_) == 0:
            continue

        # Collect RMSE values and their corresponding voxel labels
        rmse_items = list(rmse_.items())

        # If more than the desired number of voxels, sample 5 voxels randomly
        if len(rmse_items) > num_voxels:
            rmse_items = random.sample(rmse_items, num_voxels)

        total_points = 0
        weighted_rmse_sum = 0
        rmse_values = []

        # Compute RMSE statistics for the sampled voxels
        for voxel_label, rmse_value in rmse_items:
            num_points_in_voxel = np.sum(voxel_labels_ == voxel_label)
            weighted_rmse_sum += rmse_value * num_points_in_voxel
            total_points += num_points_in_voxel
            rmse_values.append(rmse_value)

        # Average RMSE weighted by the number of points in each voxel
        average_rmse_ = weighted_rmse_sum / total_points if total_points > 0 else 0

        # Append the results for the current label to the data list
        data.append({
            'label': label,
            'average_rmse': average_rmse_,
            'rmse_values': rmse_values,
            'number_of_voxels': len(rmse_items),
            'total_points': total_points,
        })

    # Return the RMSE statistics for all labels in the image
    return data


def process_multiple_images(image_list, lidar_dir, labels_dir, csv_file, target_class=None, z_threshold=1, voxel_size=15):
    """
    Processes multiple images and collects RMSE data for the target class across all images.
    """
    all_rmse_values = []

    for prefix in image_list:
        # Process each image and get RMSE statistics for all labels
        image_data = process_image(prefix, lidar_dir, labels_dir, csv_file, target_class, z_threshold, voxel_size)

        # Collect RMSE values for the target class
        for data in image_data:
            all_rmse_values.extend(data['rmse_values'])

    return all_rmse_values  # Return all RMSE values collected for all images


def perform_t_test(rmse_1, rmse_2):
    """
    Performs a t-test on two sets of RMSE values.
    """
    t_stat, p_value = ttest_ind(rmse_1, rmse_2, equal_var=False)  # Welch's t-test for unequal variances
    return t_stat, p_value

def run_experiment_with_violin(lidar_dir1, labels_dir1, lidar_dir2, labels_dir2, csv_file, target_class1,
                                   target_class2, image_list1, image_list2, iterations=10):
    # Create a directory for results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{target_class1}_vs_{target_class2}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Save the image names into a text file
    with open(os.path.join(output_dir, 'image_names.txt'), 'w') as f:
        f.write("Image List 1:\n")
        f.write('\n'.join(image_list1) + '\n\n')
        f.write("Image List 2:\n")
        f.write('\n'.join(image_list2))

    results = []

    for i in range(iterations):
        print(f"Running experiment iteration {i + 1}/{iterations}...")

        # Process images for the first target class and corresponding directory
        rmse_1 = process_multiple_images(image_list1, lidar_dir1, labels_dir1, csv_file, target_class=target_class1)

        # Process images for the second target class and corresponding directory
        rmse_2 = process_multiple_images(image_list2, lidar_dir2, labels_dir2, csv_file, target_class=target_class2)

        t_stat, p_value = perform_t_test(rmse_1, rmse_2)

        # Log results for each iteration
        results.append({
            'Iteration': i + 1,
            'T-Statistic': t_stat,
            'P-Value': p_value
        })

        print(f"Iteration {i + 1}: T-statistic = {t_stat}, P-value = {p_value}")


        # Save RMSE values to a CSV file for the current iteration
        rmse_csv_filename = os.path.join(output_dir, f'iteration_{i + 1}_rmse_values.csv')
        with open(rmse_csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Target_Class', 'RMSE_Value'])
            for rmse_value in rmse_1:
                writer.writerow([target_class1, rmse_value])
            for rmse_value in rmse_2:
                writer.writerow([target_class2, rmse_value])

        print(f"RMSE values for iteration {i + 1} have been saved to {rmse_csv_filename}")

            # Plot violin plots for the RMSE values for this iteration
        plot_violin_rmse_per_iteration(rmse_1, rmse_2, target_class1, target_class2, output_dir, iteration=i + 1)



        # Write the t-test results to a CSV file
        csv_filename = os.path.join(output_dir, f'{target_class1}_vs_{target_class2}_t_test.csv')
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Iteration', 'T-Statistic', 'P-Value'])
            writer.writeheader()
            writer.writerows(results)

        print(f"T-test results have been saved to {csv_filename}")

        plot_p_values(csv_filename, output_dir)

def plot_violin_rmse_per_iteration(rmse_class1, rmse_class2, target_class1, target_class2, output_dir, iteration):
        """
        Generates a violin plot of RMSE values for a single experiment iteration.
        """
        # Combine the RMSE values and class labels into a single data structure
        data = {
            'RMSE': rmse_class1 + rmse_class2,
            'Target_Class': [target_class1] * len(rmse_class1) + [target_class2] * len(rmse_class2)
        }

        # Create a violin plot of the RMSE values for each class for this iteration
        plt.figure(figsize=(8, 6))
        sns.violinplot(x='Target_Class', y='RMSE', data=data)
        plt.title(f'Violin Plot of RMSE for {target_class1} and {target_class2} (Iteration {iteration})')
        plt.ylabel('RMSE')

        # Save the plot for this iteration
        plt.savefig(os.path.join(output_dir, f'violin_plot_rmse_iteration_{iteration}.png'))
        plt.close()

        print(f"Violin plot for iteration {iteration} has been saved.")





def plot_p_values(csv_filename, output_dir):
    p_values = []

    with open(csv_filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            p_values.append(float(row['P-Value']))

    # Plot 1: Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=p_values)
    plt.title("Boxplot of P-Values")
    plt.ylabel("P-Values")
    plt.savefig(os.path.join(output_dir, 'boxplot.png'))
    plt.close()

    # Plot 2: Violin plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=p_values)
    plt.title("Violin Plot of P-Values")
    plt.ylabel("P-Values")
    plt.savefig(os.path.join(output_dir, 'violin_plot.png'))
    plt.close()

    # Plot 3: Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(p_values, bins=20, edgecolor='black')
    plt.axvline(0.05, color='r', linestyle='--', label='p = 0.05')
    plt.axvline(0.01, color='g', linestyle='--', label='p = 0.01')
    plt.title('Histogram of P-Values')
    plt.xlabel('P-Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'histogram.png'))
    plt.close()

    # Plot 4: ECDF (Empirical Cumulative Distribution Function)
    plt.figure(figsize=(8, 6))
    sns.ecdfplot(p_values)
    plt.axvline(0.05, color='r', linestyle='--', label='p = 0.05')
    plt.axvline(0.01, color='g', linestyle='--', label='p = 0.01')
    plt.title('ECDF of P-Values')
    plt.xlabel('P-Value')
    plt.ylabel('Proportion <= P-Value')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'ecdf.png'))
    plt.close()


if __name__ == "__main__":
    # Directories for first set of images
    lidar_dir1 = 'goose_3d_val/lidar/val/2022-12-07_aying_hills'
    labels_dir1 = 'goose_3d_val/labels/val/2022-12-07_aying_hills'

    # Directories for second set of images
    lidar_dir2 = 'goose_3d_val/lidar/val/2022-12-07_aying_hills'
    labels_dir2 = 'goose_3d_val/labels/val/2022-12-07_aying_hills'

    csv_file = 'goose_3d_val/goose_label_mapping.csv'

    # Prompt for the class labels to compare
    target_class1 = input("Enter the class for image set 1 (e.g., 'low_grass'): ")
    target_class2 = input("Enter the class for image set 2 (e.g., 'bush'): ")

    # Image lists for comparison
    image_list1 = [
        '2022-12-07_aying_hills__0006_1670420708448844860',
        '2022-12-07_aying_hills__0009_1670420878948219746',
        '2022-12-07_aying_hills__0010_1670420972132205304',
        '2022-12-07_aying_hills__0011_1670420979760256580',
        '2022-12-07_aying_hills__0012_1670420985739069345',
    ]

    image_list2 = [
        '2022-12-07_aying_hills__0006_1670420708448844860',
        '2022-12-07_aying_hills__0009_1670420878948219746',
        '2022-12-07_aying_hills__0010_1670420972132205304',
        '2022-12-07_aying_hills__0011_1670420979760256580',
        '2022-12-07_aying_hills__0012_1670420985739069345',
    ]

    run_experiment_with_violin(lidar_dir1, labels_dir1, lidar_dir2, labels_dir2, csv_file, target_class1, target_class2, image_list1, image_list2, iterations=5)

    image_list2 = [
        '2023-03-03_garching_2__0114_1677850404544550420',
        '2023-03-03_garching_2__0115_1677850409391907174',
        '2023-03-03_garching_2__0116_1677850414554513656',
        '2023-03-03_garching_2__0117_1677850415819237733',
        '2023-03-03_garching_2__0118_1677850418032002589',
    ]
