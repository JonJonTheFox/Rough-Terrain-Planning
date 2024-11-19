# UNUSED SCRIPT
# This script is intended for comparing segmentation analysis.
# It is currently not utilized in the main pipeline and serves as a placeholder 
# for potential future work on segmentation model comparison.

import time
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pointnet2.models # Make sure to have PointNet++ installed
from Voxelization import voxel3d as v3d
import plane_Fitting as pf
import random


# Load .bin data
def load_bin_data(bin_path):
    """ Loads a .bin file with LiDAR point cloud data and returns the points as a numpy array. """
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  # (x, y, z, intensity)
    return point_cloud

# RandomForest feature extraction (from your earlier code)
def process_image_with_metrics(prefix, lidar_dir, labels_dir, csv_file, target_classes, z_threshold=1, voxel_size=5, num_voxels=100000):
    """
    Processes the LiDAR point cloud to extract RMSE values, number of points, density, and majority label.
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

    for label in np.unique(labels):
        if label not in target_label_keys:
            continue

        # Filter the point cloud based on the current label
        filtered_pointcloud = pointcloud[labels == label]

        # Voxelize the filtered point cloud
        voxel_labels_, voxel_map_, unique_voxel_labels_ = v3d.voxelize_point_cloud_2d(filtered_pointcloud, voxel_size=voxel_size)

        if len(unique_voxel_labels_) > num_voxels:
            sampled_voxel_labels = random.sample(list(unique_voxel_labels_), num_voxels)
        else:
            sampled_voxel_labels = unique_voxel_labels_

        labels_in_voxelized_cloud = labels[labels == label][:len(voxel_labels_)]

        # Compute the plane for each voxel and the associated RMSE
        voxel_planes_, rmse_ = pf.compute_voxel_planes(filtered_pointcloud, voxel_labels_)

        for voxel_label in sampled_voxel_labels:
            if voxel_label not in rmse_:
                continue

            rmse_value = rmse_[voxel_label]
            all_voxel_rmse.append(rmse_value)

            points_in_voxel = filtered_pointcloud[voxel_labels_ == voxel_label]
            labels_in_voxel = labels_in_voxelized_cloud[voxel_labels_ == voxel_label]

            if len(points_in_voxel) != len(labels_in_voxel):
                continue

            num_points = len(points_in_voxel)
            num_points_list.append(num_points)
            density_list.append(num_points / voxel_volume)

            majority_label = np.bincount(labels_in_voxel).argmax()
            majority_labels.append(majority_label)

    return all_voxel_rmse, num_points_list, density_list, majority_labels

# Run RandomForest model
def run_random_forest(point_cloud, lidar_dir, labels_dir, csv_file, target_classes, image_list, iterations=1):
    """ Trains and tests a RandomForestClassifier on the processed LiDAR data and returns the runtime. """
    start_time = time.time()

    # Extract metrics for RandomForest using your function
    all_voxel_rmse, num_points, density, majority_labels = [], [], [], []
    for prefix in image_list:
        rmse, num_points_voxel, density_voxel, majority_labels_voxel = process_image_with_metrics(
            prefix, lidar_dir, labels_dir, csv_file, target_classes, num_voxels=100000)
        all_voxel_rmse.extend(rmse)
        num_points.extend(num_points_voxel)
        density.extend(density_voxel)
        majority_labels.extend(majority_labels_voxel)

    # Prepare data for RandomForest
    data = {'RMSE': all_voxel_rmse, 'Num_Points': num_points, 'Density': density}
    df = pd.DataFrame(data)
    X = df[['RMSE', 'Num_Points', 'Density']]
    y = majority_labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train RandomForest
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    # Test RandomForest
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"RandomForest Accuracy: {accuracy}")

    runtime = time.time() - start_time
    print(f"RandomForest Runtime: {runtime:.4f} seconds")
    return runtime

# Load PointNet++ model
def load_pointnet_model():
    """ Loads the PointNet++ segmentation model. """
    model = pointnet2.models.pointnet2_ssg_cls.PointnetSAModule(input_channels=3, num_classes=64)
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()
    return model

# Run PointNet++ inference
def run_pointnet_inference(point_cloud, model):
    """ Runs inference on the point cloud data using PointNet++ and returns the runtime. """
    start_time = time.time()

    # Prepare point cloud
    point_cloud = torch.tensor(point_cloud[:, :3]).float()  # Extract x, y, z
    point_cloud = point_cloud.unsqueeze(0)  # Add batch dimension
    if torch.cuda.is_available():
        point_cloud = point_cloud.cuda()

    # Run inference
    with torch.no_grad():
        segmentation_output = model(point_cloud)
        predicted_labels = torch.argmax(segmentation_output, dim=1).cpu().numpy()

    runtime = time.time() - start_time
    print(f"PointNet++ Runtime: {runtime:.4f} seconds")
    return runtime

# Main function to run both models and compare time
def compare_models(bin_file, lidar_dir, labels_dir, csv_file, target_classes, image_list):
    """ Compares the runtime of RandomForest and PointNet++ on the given LiDAR .bin file. """
    print("Loading LiDAR data...")
    point_cloud = load_bin_data(bin_file)

    # Load PointNet++ model
    pointnet_model = load_pointnet_model()

    print("\nRunning RandomForest...")
    rf_runtime = run_random_forest(point_cloud, lidar_dir, labels_dir, csv_file, target_classes, image_list)

    print("\nRunning PointNet++...")
    pn_runtime = run_pointnet_inference(point_cloud, pointnet_model)

    print("\n--- Time Comparison ---")
    print(f"RandomForest Runtime: {rf_runtime:.4f} seconds")
    print(f"PointNet++ Runtime: {pn_runtime:.4f} seconds")


if __name__ == "__main__":
    bin_file_path = 'path/to/your/lidar_data.bin'  # Path to your LiDAR .bin file
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

    # Compare both models
    compare_models(bin_file_path, lidar_dir, labels_dir, csv_file, target_classes, image_list)
