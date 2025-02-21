import csv

from Voxelization import voxel3d as v3d
import plane_Fitting as pf
import numpy as np
from scipy.stats import ttest_ind
import random


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
        voxel_labels_, voxel_map_ = v3d.voxelize_point_cloud_2d(filtered_pointcloud, voxel_size=voxel_size)

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

def run_experiment(lidar_dir, labels_dir, csv_file, target_class1, target_class2,  image_list1, image_list2, iterations=10):
    results = []

    for i in range(iterations):
        rmse_1 = process_multiple_images(image_list1, lidar_dir, labels_dir, csv_file, target_class=target_class1)
        rmse_2 = process_multiple_images(image_list2, lidar_dir, labels_dir, csv_file, target_class=target_class2)

        t_stat, p_value = perform_t_test(rmse_1, rmse_2)

        # Log results for each iteration
        results.append({
            'Iteration': i + 1,
            'T-Statistic': t_stat,
            'P-Value': p_value
        })

        print(f"Iteration {i + 1}: T-statistic = {t_stat}, P-value = {p_value}")

    # Write the results to a CSV file
    with open(f'{target_class1}&{target_class2}.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Iteration', 'T-Statistic', 'P-Value'])
        writer.writeheader()
        writer.writerows(results)

    print("Results have been saved to {target_class1}&{target_class2}.csv")


if __name__ == "__main__":
    # Directories
    lidar_dir = '../goose_3d_val/lidar/val/2022-12-07_aying_hills'
    labels_dir = '../goose_3d_val/labels/val/2022-12-07_aying_hills'
    csv_file = '../goose_3d_val/goose_label_mapping.csv'

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

    run_experiment(lidar_dir, labels_dir, csv_file, target_class1, target_class2, image_list1, image_list2, iterations=10)

image_list = [
        '2022-12-07_aying_hills__0000_1670420609181206687',
        '2022-12-07_aying_hills__0001_1670420652887861538',
        '2022-12-07_aying_hills__0002_1670420663298382187',
        '2022-12-07_aying_hills__0003_1670420665566527858',
        '2022-12-07_aying_hills__0004_1670420689172563910',
        '2022-12-07_aying_hills__0005_1670420698140436007',
        '2022-12-07_aying_hills__0006_1670420708448844860',
        '2022-12-07_aying_hills__0008_1670420875957530235',
        '2022-12-07_aying_hills__0009_1670420878948219746',
        '2022-12-07_aying_hills__0010_1670420972132205304',
        '2022-12-07_aying_hills__0011_1670420979760256580',
        '2022-12-07_aying_hills__0012_1670420985739069345',
        #'2022-12-07_aying_hills__0013_1670420991614950614',
        #'2022-12-07_aying_hills__0014_1670421005324235752',
        #'2022-12-07_aying_hills__0015_1670421010066267238',
        #'2022-12-07_aying_hills__0016_1670421031919145646',
        #'2022-12-07_aying_hills__0017_1670421057895607851',
        #'2022-12-07_aying_hills__0018_1670421079232429484',
        #'2022-12-07_aying_hills__0019_1670421095003492677',
        #'2022-12-07_aying_hills__0020_1670421106754371527',
        #'2022-12-07_aying_hills__0021_1670421115000885398',
        #'2022-12-07_aying_hills__0022_1670421125720729310',
        #'2022-12-07_aying_hills__0023_1670421136338979710',
        #'2022-12-07_aying_hills__0024_1670421147263892369',
        #'2022-12-07_aying_hills__0025_1670421148708137583',
        #'2022-12-07_aying_hills__0026_1670421149222734552',
        #'2022-12-07_aying_hills__0027_1670421152520763486',
        #'2022-12-07_aying_hills__0028_1670421156954085022',
        #'2022-12-07_aying_hills__0029_1670421161282631132',
        #'2022-12-07_aying_hills__0030_1670421161488766857',
        #'2022-12-07_aying_hills__0031_1670421164065826921',
        #'2022-12-07_aying_hills__0032_1670421167674448570',
        #'2022-12-07_aying_hills__0033_1670421176951435802',
        #'2022-12-07_aying_hills__0034_1670421194782947030',
        #'2022-12-07_aying_hills__0035_1670421216118815596',
        #'2022-12-07_aying_hills__0036_1670421219933042864',
        #'2022-12-07_aying_hills__0037_1670421230548919820',
        #'2022-12-07_aying_hills__0038_1670421236630781967',
        #'2022-12-07_aying_hills__0039_1670421255286672290',
        #'2022-12-07_aying_hills__0040_1670421276003323059',
        #'2022-12-07_aying_hills__0041_1670421306821351925',
        #'2022-12-07_aying_hills__0042_1670421312284207423',
        #'2022-12-07_aying_hills__0043_1670421315376691070',
        #'2022-12-07_aying_hills__0044_1670421321354124635',
        #'2022-12-07_aying_hills__0045_1670421354439414871',
        #'2022-12-07_aying_hills__0046_1670421360005772609',
        #2022-12-07_aying_hills__0047_1670421370621698009',
        #2022-12-07_aying_hills__0048_1670421387318425961',
        #'2022-12-07_aying_hills__0049_1670421413189702923',
        #'2022-12-07_aying_hills__0050_1670421419166945886',
        #'2022-12-07_aying_hills__0051_1670421422259751319',
        #'2022-12-07_aying_hills__0052_1670421426175744005',
        #'2022-12-07_aying_hills__0053_1670421437616759245',
        #'2022-12-07_aying_hills__0054_1670421438647357603',
        #'2022-12-07_aying_hills__0055_1670421442048655345',
        #'2022-12-07_aying_hills__0056_1670421446481393352',
        #'2022-12-07_aying_hills__0057_1670421449778652057',
        #'2022-12-07_aying_hills__0058_1670421454932264554',
        #'2022-12-07_aying_hills__0059_1670421456993279654',
        #'2022-12-07_aying_hills__0060_1670421460086112887',
        #'2022-12-07_aying_hills__0061_1670421460704308119',
        '2022-12-07_aying_hills__0062_1670421463383944594',
        '2022-12-07_aying_hills__0063_1670421463796196650',
        '2022-12-07_aying_hills__0064_1670421474001823741',
        '2022-12-07_aying_hills__0065_1670421475958264241',
        '2022-12-07_aying_hills__0066_1670421481936547901'
    ]