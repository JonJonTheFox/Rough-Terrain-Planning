from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import csv
import voxel3d as v3d
import plane_Fitting as pf
import numpy as np
import random
import os
from datetime import datetime

def process_image_with_metrics(prefix, lidar_dir, labels_dir, csv_file, target_classes, z_threshold=1, voxel_size=5, num_voxels=100000):
    """
    Processes a single image and returns RMSE values along with number of points, density, and majority label
    for each voxel, limited by num_voxels, and only for specified target classes.
    """
    # Load pointcloud and labels
    lidar_data, labels, label_metadata = pf.load_pointcloud_and_labels(prefix, lidar_dir, labels_dir, csv_file)

    # Map target classes to corresponding label keys
    target_label_keys = label_metadata[label_metadata['class_name'].isin(target_classes)]['label_key'].values

    # Apply z-threshold to pointcloud and labels
    pointcloud, labels = pf.apply_threshold(lidar_data, labels, z_threshold)

    # Initialize lists to store RMSE values, number of points, density, and majority labels
    all_voxel_rmse = []
    num_points_list = []
    density_list = []
    majority_labels = []

    voxel_volume = voxel_size ** 3  # Assuming a cubic voxel

    # Iterate over each unique label in the labels
    for label in np.unique(labels):
        if label not in target_label_keys:
            continue  # Skip labels that are not in the target classes

        # Filter the point cloud based on the current label
        filtered_pointcloud = pointcloud[labels == label]

        # Voxelize the filtered point cloud
        voxel_labels_, voxel_map_, unique_voxel_labels_ = v3d.voxelize_point_cloud_2d(filtered_pointcloud, voxel_size=voxel_size)

        # Check if the number of voxels exceeds the limit
        if len(unique_voxel_labels_) > num_voxels:
            # Randomly sample num_voxels from the list of unique voxel labels
            sampled_voxel_labels = random.sample(list(unique_voxel_labels_), num_voxels)
        else:
            sampled_voxel_labels = unique_voxel_labels_

        labels_in_voxelized_cloud = labels[labels == label][:len(voxel_labels_)]  # Align the labels with voxelized points

        # Compute the plane for each voxel and the associated RMSE
        voxel_planes_, rmse_ = pf.compute_voxel_planes(filtered_pointcloud, voxel_labels_)

        for voxel_label in sampled_voxel_labels:
            # Get RMSE for the voxel
            if voxel_label not in rmse_:
                continue  # Skip if RMSE was not computed for the voxel

            rmse_value = rmse_[voxel_label]
            all_voxel_rmse.append(rmse_value)

            points_in_voxel = filtered_pointcloud[voxel_labels_ == voxel_label]
            labels_in_voxel = labels_in_voxelized_cloud[voxel_labels_ == voxel_label]

            if len(points_in_voxel) != len(labels_in_voxel):
                continue

            # Number of points and density
            num_points = len(points_in_voxel)
            num_points_list.append(num_points)
            density_list.append(num_points / voxel_volume)

            # Calculate the majority label in the voxel and store it
            majority_label = np.bincount(labels_in_voxel).argmax()
            majority_labels.append(majority_label)

    return all_voxel_rmse, num_points_list, density_list, majority_labels


def run_single_image_list_test(lidar_dir, labels_dir, csv_file, image_list, target_classes, iterations=1, num_voxels=100000):
    """
    Runs a real data test case for the SVM experiment using actual LiDAR data from the provided image list.
    Limits the number of voxels per image to num_voxels and only classifies the target classes.
    """
    # Create a directory for results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"classificationTest_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for i in range(iterations):
        print(f"Running real data test iteration {i + 1}/{iterations}...")

        # Process images in the image_list
        all_voxel_rmse, num_points, density, majority_labels = [], [], [], []
        for prefix in image_list:
            rmse, num_points_voxel, density_voxel, majority_labels_voxel = process_image_with_metrics(
                prefix, lidar_dir, labels_dir, csv_file, target_classes=target_classes, num_voxels=num_voxels
            )
            all_voxel_rmse.extend(rmse)
            num_points.extend(num_points_voxel)
            density.extend(density_voxel)
            majority_labels.extend(majority_labels_voxel)

        # Combine the data into one DataFrame
        data = {
            'RMSE': all_voxel_rmse,
            'Num_Points': num_points,
            'Density': density,
            'Class': majority_labels  # The majority labels for each voxel
        }

        df = pd.DataFrame(data)

        # Split the data for SVM classification
        X = df[['RMSE', 'Num_Points', 'Density']]
        y = df['Class']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train an SVM classifier
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)

        # Predict and evaluate the model
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Iteration {i + 1}: Accuracy = {accuracy}")
        print(f"Classification Report:\n {classification_report(y_test, y_pred)}")

        # Log results for each iteration
        results.append({
            'Iteration': i + 1,
            'Accuracy': accuracy,
            'Classification Report': classification_report(y_test, y_pred, output_dict=True)
        })

        # Save the dataframe and results
        df.to_csv(os.path.join(output_dir, f'iteration_{i + 1}_real_data.csv'), index=False)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'svm_real_data_results.csv'), index=False)

    print(f"SVM results have been saved to {output_dir}")


if __name__ == "__main__":
    # Directories for the images
    lidar_dir = 'goose_3d_val/lidar/val/2022-12-07_aying_hills'
    labels_dir = 'goose_3d_val/labels/val/2022-12-07_aying_hills'

    csv_file = 'goose_3d_val/goose_label_mapping.csv'

    # Target classes for classification (asphalt, bush, low_grass)
    target_classes = ['asphalt', 'bush', 'low_grass']

    # Image list for classification
    image_list = [
        '2022-12-07_aying_hills__0006_1670420708448844860',
        '2022-12-07_aying_hills__0009_1670420878948219746',
    ]

    # Run the real data test with one iteration and limit the number of voxels to 10
    run_single_image_list_test(lidar_dir, labels_dir, csv_file, image_list, target_classes=target_classes, iterations=1, num_voxels=100000)

