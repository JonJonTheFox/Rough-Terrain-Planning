from matplotlib import pyplot as plt
import seaborn as sns
from Voxelization import voxel3d as v3d
import PlaneFitting.plane_Fitting as pf
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# Function to apply LBP on a voxelized point cloud
def apply_dynamic_lbp_to_voxels(pointcloud, labels, voxel_labels, high_grass_label=None, low_grass_label=None,
                                k_neighbors=6):
    """
    Applies Label Binary Pattern (LBP) to a subset of the voxel point cloud.
    """
    lbp_patterns = {}
    knn = NearestNeighbors(n_neighbors=k_neighbors)

    # Train KNN model on the current subset of the pointcloud
    knn.set_params(n_neighbors=min(k_neighbors, len(pointcloud)))
    knn.fit(pointcloud)

    # Iterate over each unique label in this voxel's subset
    for voxel_label in np.unique(voxel_labels):
        points_in_voxel = pointcloud[voxel_labels == voxel_label]
        if len(points_in_voxel) == 0:
            continue
        voxel_centroid = np.mean(points_in_voxel, axis=0).reshape(1, -1)

        # Find the K nearest neighbors to the centroid
        distances, indices = knn.kneighbors(voxel_centroid)

        lbp_value = 0
        for i, neighbor_idx in enumerate(indices[0]):
            neighbor_point = pointcloud[neighbor_idx]
            height_diff = voxel_centroid[0][2] - neighbor_point[2]
            lbp_value |= (height_diff > 0) << i

        lbp_patterns[voxel_label] = lbp_value

    return lbp_patterns


# Function to map labels to categories
def map_labels_to_categories(label_mapping, label_key):
    obstacle_labels = {'animal', 'barrel', 'barrier_tape', 'bicycle', 'boom_barrier', 'bridge', 'building', 'bus',
                       'car', 'caravan', 'container', 'debris', 'fence', 'guard_rail', 'heavy_machinery', 'hedge',
                       'kick_scooter', 'misc_sign', 'motorcycle', 'obstacle', 'pedestrian_crossing', 'person', 'pole',
                       'rail_track', 'rider', 'road_block', 'rock', 'scenery_vegetation', 'street_light',
                       'traffic_cone',
                       'traffic_light', 'traffic_sign', 'trailer', 'tree_crown', 'tree_root', 'tree_trunk', 'truck',
                       'tunnel', 'wall', 'wire', 'high_grass'}
    passable_labels = {'asphalt', 'cobble', 'gravel', 'sidewalk', 'soil', 'low_grass'}
    if label_key not in label_mapping:
        print(f"Warning: label_key {label_key} not found in label_mapping. Assigning to 'other' category.")
        return 2
    if label_mapping[label_key] in obstacle_labels:
        return 0
    elif label_mapping[label_key] in passable_labels:
        return 1
    else:
        return 2


def process_image_with_metrics_and_lbp_and_obstacle_classification(prefix, lidar_dir, labels_dir, csv_file,
                                                                   label_mapping, z_threshold=1, voxel_size=5,
                                                                   num_voxels=100000, k_neighbors=6,
                                                                   min_len=20, proportion_threshold=0.9):
    """
    Processes a single image and returns RMSE values, LBP values, number of points, density, and a new feature:
    whether the voxel is an 'obstacle', 'passable', 'low_grass', or 'high_grass', while applying voxel preprocessing.
    """
    # Load pointcloud and labels
    lidar_data, labels, label_metadata = pf.load_pointcloud_and_labels(prefix, lidar_dir, labels_dir, csv_file)

    # Apply z-threshold to pointcloud and labels (filter the point cloud and corresponding labels)
    pointcloud, labels = pf.apply_threshold(lidar_data, labels, z_threshold)

    # Voxelize the point cloud
    voxel_labels_, voxel_map_, unique_voxel_labels_ = v3d.voxelize_point_cloud_2d(pointcloud, voxel_size=voxel_size)

    # Store the points for each voxel
    voxel_to_points = {voxel_label: [] for voxel_label in unique_voxel_labels_}
    total_points = len(pointcloud)  # Track initial number of points
    retained_points = 0  # Initialize counter for retained points after filtering

    for idx, voxel_label in enumerate(voxel_labels_):
        voxel_to_points[voxel_label].append(idx)  # Map the point index to its corresponding voxel label

    all_voxel_rmse = []
    num_points_list = []
    density_list = []
    lbp_list = []
    majority_labels = []
    class_categories = []

    voxel_volume = voxel_size ** 3

    # Iterate over each voxel
    for voxel_label, point_indices in voxel_to_points.items():
        points_in_voxel = pointcloud[point_indices]
        labels_in_voxel = labels[point_indices]

        if len(points_in_voxel) < min_len:
            continue

        # Calculate the majority label in the voxel
        unique_labels, label_counts = np.unique(labels_in_voxel, return_counts=True)
        majority_label = unique_labels[np.argmax(label_counts)]
        majority_proportion = label_counts.max() / label_counts.sum()

        if majority_proportion < proportion_threshold:
            continue

        # Increment the count of retained points after passing both filters
        retained_points += len(points_in_voxel)

        # Pass only the points and voxel labels specific to this voxel
        voxel_planes_, rmse_ = pf.compute_voxel_planes(points_in_voxel, np.array([voxel_label] * len(points_in_voxel)))

        # Calculate LBP for the filtered point cloud
        # Modify the LBP calculation call in the main processing function
        voxel_labels_filtered = voxel_labels_[point_indices]  # Only labels relevant to points in the current voxel
        lbp_patterns = apply_dynamic_lbp_to_voxels(points_in_voxel, labels_in_voxel, voxel_labels_filtered,
                                                   # filtered labels
                                                   high_grass_label=None, low_grass_label=None,
                                                   k_neighbors=k_neighbors)

        if voxel_label not in rmse_:
            continue

        rmse_value = rmse_[voxel_label]
        all_voxel_rmse.append(rmse_value)

        # Number of points and density
        num_points = len(points_in_voxel)
        num_points_list.append(num_points)
        density_list.append(num_points / voxel_volume)

        # Store the majority label
        majority_labels.append(majority_label)

        # Get the LBP value for this voxel
        lbp_value = lbp_patterns.get(voxel_label, 0)
        lbp_list.append(lbp_value)

        # Map the majority label to the new class categories (obstacle, passable, or grass)
        class_category = map_labels_to_categories(label_mapping, majority_label)
        class_categories.append(class_category)

    # Calculate and print the percentage of data retained
    retention_percentage = (retained_points / total_points) * 100
    print(f"Total initial points: {total_points}")
    print(f"Points retained after filtering: {retained_points}")
    print(f"Percentage of data retained after filtering: {retention_percentage:.2f}%")

    return all_voxel_rmse, num_points_list, density_list, lbp_list, majority_labels, class_categories




# Run function
def run_single_image_list_test_with_lbp_and_obstacle_classification(lidar_dir, labels_dir, csv_file, image_list,
                                                                    label_mapping, iterations=1, num_voxels=100000,
                                                                    k_neighbors=6):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"classificationTest_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i in range(iterations):
        print(f"Running real data test iteration {i + 1}/{iterations}...")
        all_voxel_rmse, num_points, density, lbp_values, majority_labels, class_categories = [], [], [], [], [], []

        for prefix in image_list:
            rmse, num_points_voxel, density_voxel, lbp_voxel, majority_labels_voxel, class_categories_voxel = process_image_with_metrics_and_lbp_and_obstacle_classification(
                prefix, lidar_dir, labels_dir, csv_file, label_mapping=label_mapping, num_voxels=num_voxels,
                k_neighbors=k_neighbors)
            all_voxel_rmse.extend(rmse)
            num_points.extend(num_points_voxel)
            density.extend(density_voxel)
            lbp_values.extend(lbp_voxel)
            majority_labels.extend(majority_labels_voxel)
            class_categories.extend(class_categories_voxel)

        data = {
            'RMSE': all_voxel_rmse,
            'Num_Points': num_points,
            'Density': density,
            'LBP': lbp_values,
            'Class': majority_labels,
            'Category': class_categories
        }
        df = pd.DataFrame(data)
        X = df[['RMSE','Num_Points', 'Density', 'LBP']]
        y = df['Category']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Iteration {i + 1}: Accuracy = {accuracy}")
        print(f"Classification Report:\n {classification_report(y_test, y_pred)}")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n {conf_matrix}")

        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test),
                    yticklabels=np.unique(y_test))
        plt.title(f'Confusion Matrix - Iteration {i + 1}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plot_filename = os.path.join(output_dir, f'confusion_matrix_{i + 1}.png')
        plt.savefig(plot_filename)
        plt.close()
        print(f"Confusion matrix for iteration {i + 1} saved to {plot_filename}")
        results.append({
            'Iteration': i + 1,
            'Accuracy': accuracy,
            'Classification Report': classification_report(y_test, y_pred, output_dict=True)
        })
        df.to_csv(os.path.join(output_dir, f'iteration_{i + 1}_real_data.csv'), index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'classification_real_data_results.csv'), index=False)


if __name__ == "__main__":
    lidar_dir = '../goose_3d_val/lidar/val/2022-07-22_flight'
    labels_dir = '../goose_3d_val/labels/val/2022-07-22_flight'
    csv_file = '../goose_3d_val/goose_label_mapping.csv'
    target_classes = ['high_grass', 'low_grass', 'asphalt']
    image_list = [
        '2022-07-22_flight__0071_1658494234334310308',
        '2022-07-22_flight__0072_1658494235988100385',
        '2022-07-22_flight__0073_1658494238675704025',
        '2022-07-22_flight__0075_1658494242083534022',
    ]
    label_mapping = {
        33: 'animal', 23: 'asphalt', 60: 'barrel', 48: 'barrier_tape', 13: 'bicycle', 7: 'bikeway',
        25: 'boom_barrier', 43: 'bridge', 38: 'building', 15: 'bus', 17: 'bush', 12: 'car', 36: 'caravan',
        3: 'cobble', 58: 'container', 30: 'crops', 22: 'curb', 29: 'debris', 8: 'ego_vehicle', 41: 'fence',
        16: 'forest', 24: 'gravel', 42: 'guard_rail', 57: 'heavy_machinery', 59: 'hedge', 51: 'high_grass',
        49: 'kick_scooter', 5: 'leaves', 50: 'low_grass', 63: 'military_vehicle', 47: 'misc_sign',
        18: 'moss', 20: 'motorcycle', 4: 'obstacle', 35: 'on_rails', 56: 'outlier', 9: 'pedestrian_crossing',
        14: 'person', 61: 'pipe', 45: 'pole', 26: 'rail_track', 32: 'rider', 10: 'road_block',
        11: 'road_marking', 40: 'rock', 52: 'scenery_vegetation', 21: 'sidewalk', 53: 'sky', 2: 'snow',
        31: 'soil', 6: 'street_light', 1: 'traffic_cone', 19: 'traffic_light', 46: 'traffic_sign',
        37: 'trailer', 27: 'tree_crown', 62: 'tree_root', 28: 'tree_trunk', 34: 'truck', 44: 'tunnel',
        0: 'undefined', 39: 'wall', 54: 'water', 55: 'wire'
    }
    run_single_image_list_test_with_lbp_and_obstacle_classification(lidar_dir, labels_dir, csv_file, image_list,
                                                                    label_mapping, iterations=1, num_voxels=1000,
                                                                    k_neighbors=5)
