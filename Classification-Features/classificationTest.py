import time
from cProfile import label

from matplotlib import pyplot as plt
import seaborn as sns
from Voxelization import voxel3d as v3d
import PlaneFitting.plane_Fitting as pf
import os
from datetime import datetime
import numpy as np
import pandas as pd
import logging
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

location_accuracies = {}

# Modified main function to run classification with the new dataframe
from collections import defaultdict

# Dictionary to hold F1 and recall scores for each label per location
f1_scores = defaultdict(list)
recall_scores = defaultdict(list)


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
                       'kick_scooter', 'misc_sign', 'motorcycle', 'obstacle', 'person', 'pole',
                       'rail_track', 'rider', 'road_block', 'rock', 'scenery_vegetation', 'street_light',
                       'traffic_cone',
                       'traffic_light', 'traffic_sign', 'trailer', 'tree_crown', 'tree_root', 'tree_trunk', 'truck',
                       'tunnel', 'wall', 'wire'}
    passable_labels = {'asphalt', 'cobble', 'gravel', 'sidewalk', 'soil', 'low_grass'}
    if label_key not in label_mapping:
        print(f"Warning: label_key {label_key} not found in label_mapping. Assigning to 'other' category.")
        return 8
    if label_mapping[label_key] in obstacle_labels:
        return 0
    elif label_mapping[label_key] == 'cobble':
        return 1
    elif label_mapping[label_key] == 'gravel':
        return 2
    elif label_mapping[label_key] == 'sidewalk':
        return 3
    elif label_mapping[label_key] == 'soil':
        return 4
    elif label_mapping[label_key] == 'high_grass':
        return 5
    elif label_mapping[label_key] == 'low_grass':
        return 6
    elif label_mapping[label_key] == 'snow':
        return 7
    else:
        return 8




# Additional imports based on existing usage

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Function to center the point cloud
def center_point_cloud(point_cloud):
    centroid = np.mean(point_cloud[:, :3], axis=0)
    centered_cloud = point_cloud.copy()
    centered_cloud[:, :3] -= centroid  # Subtract the centroid from all points (XYZ only)
    return centered_cloud


# Function to compute the convex hull volume
def compute_convex_hull_volume(point_cloud):
    if len(point_cloud) < 4:
        logging.warning("Not enough points to compute convex hull volume. Returning 0.")
        return 0.0
    hull = ConvexHull(point_cloud[:, :3])  # Use only XYZ coordinates
    return hull.volume


# Function to compute the density of points
def compute_density(point_cloud):
    volume = compute_convex_hull_volume(point_cloud)
    if volume == 0:
        logging.warning("Convex hull volume is 0. Cannot compute density.")
        return 0.0
    return len(point_cloud) / volume  # Number of points per unit volume


# Function to perform PCA and compute variance ratios, flatness, and elongation
def compute_pca(point_cloud):
    pca = PCA(n_components=3)
    pca.fit(point_cloud[:, :3])  # Use only XYZ coordinates
    variance_ratios = pca.explained_variance_ratio_
    flatness = variance_ratios[1] / variance_ratios[2] if variance_ratios[2] > 0 else 0
    elongation = variance_ratios[0] / variance_ratios[1] if variance_ratios[1] > 0 else 0
    return variance_ratios, flatness, elongation


# Function to compute surface roughness
def compute_surface_roughness(point_cloud, k=10):
    if len(point_cloud) < k:
        logging.warning("Not enough points to compute surface roughness. Returning 0.")
        return 0.0
    neighbors = NearestNeighbors(n_neighbors=k).fit(point_cloud[:, :3])
    _, indices = neighbors.kneighbors(point_cloud[:, :3])
    roughness = []

    for idx in indices:
        local_points = point_cloud[idx, :3]
        local_pca = PCA(n_components=1)
        local_pca.fit(local_points)
        roughness.append(local_pca.explained_variance_ratio_[0])

    return np.mean(roughness)


# Function to compute height variability and vertical skewness
def compute_height_variability(point_cloud):
    z_values = point_cloud[:, 2]
    return np.std(z_values), skew(z_values)


# Function to compute curvature
def compute_curvature(point_cloud, k=10):
    if len(point_cloud) < k:
        logging.warning("Not enough points to compute curvature. Returning 0.")
        return 0.0
    neighbors = NearestNeighbors(n_neighbors=k).fit(point_cloud[:, :3])
    _, indices = neighbors.kneighbors(point_cloud[:, :3])
    curvatures = []

    for idx in indices:
        local_points = point_cloud[idx, :3]
        pca = PCA(n_components=3)
        pca.fit(local_points)
        curvatures.append(pca.explained_variance_ratio_[2])

    return np.mean(curvatures)


# Function to compute the mean nearest neighbor distance
def compute_mean_nearest_neighbor_distance(point_cloud, k=1):
    if len(point_cloud) <= k:
        logging.warning("Not enough points to compute mean nearest neighbor distance. Returning 0.")
        return 0.0
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(point_cloud[:, :3])
    distances, _ = neighbors.kneighbors(point_cloud[:, :3])
    return np.mean(distances[:, 1:])


# Function to compute intensity-based features (mean, std, skewness)
def compute_intensity_features(point_cloud):
    intensities = point_cloud[:, 3]
    mean_intensity = np.mean(intensities)
    std_intensity = np.std(intensities)
    skew_intensity = skew(intensities)
    return mean_intensity, std_intensity, skew_intensity


# Combined function to compute all properties for a given point cloud
# Other imports and functions remain unchanged

def create_cost_map_with_predictions(voxel_df, predictions, voxel_size=5, passable_cost=1, non_passable_cost=100):
    """
    Creates and visualizes a 2D cost map based on predicted passability values.

    Parameters:
    - voxel_df (pd.DataFrame): DataFrame containing voxel data with 'X', 'Y' coordinates.
    - predictions (np.array or pd.Series): Array of predicted categories (1 for passable, 0 for non-passable).
    - voxel_size (int): Size of each voxel in units. Default is 5.
    - passable_cost (int): Cost value assigned to passable regions. Default is 1.
    - non_passable_cost (int): Cost value assigned to non-passable regions. Default is 100.
    """
    # Extract and scale voxel coordinates
    x_coords = (voxel_df['X'] / voxel_size).astype(int)
    y_coords = (voxel_df['Y'] / voxel_size).astype(int)

    # Define grid size based on coordinate extents
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    grid_shape = (x_max - x_min + 1, y_max - y_min + 1)

    # Initialize cost map with non-passable cost
    cost_map = np.ones(grid_shape) * non_passable_cost

    # Populate the cost map based on predictions
    for x, y, prediction in zip(x_coords, y_coords, predictions):
        if prediction == 1:  # Predicted passable
            cost_map[x - x_min, y - y_min] = passable_cost
        else:  # Predicted non-passable
            cost_map[x - x_min, y - y_min] = non_passable_cost

    # Plot the cost map
    plt.figure(figsize=(10, 10))
    plt.imshow(cost_map.T, cmap='YlGnBu', origin='lower', extent=[x_min, x_max, y_min, y_max])
    plt.colorbar(label="Cost")
    plt.title("Predicted Passability Cost Map")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()


def process_image_with_metrics_and_lbp_and_obstacle_classification(prefix, lidar_dir, labels_dir, csv_file,
                                                                   label_mapping, z_threshold=1, voxel_size=5,
                                                                   num_voxels=100000, k_neighbors=6,
                                                                   min_len=10, proportion_threshold=0.5):
    # Load point cloud and labels
    lidar_data, labels, label_metadata = pf.load_pointcloud_and_labels(prefix, lidar_dir, labels_dir, csv_file)
    pointcloud, labels = pf.apply_threshold(lidar_data, labels, z_threshold)
    voxel_labels_, voxel_map_, unique_voxel_labels_ = v3d.voxelize_point_cloud_2d(pointcloud, voxel_size=voxel_size)

    voxel_to_points = {voxel_label: [] for voxel_label in unique_voxel_labels_}
    total_points = len(pointcloud)
    retained_points = 0

    for idx, voxel_label in enumerate(voxel_labels_):
        voxel_to_points[voxel_label].append(idx)

    # Lists to hold computed features
    all_voxel_rmse, num_points_list, density_list, lbp_list, majority_labels, class_categories = [], [], [], [], [], []
    convex_hull_volumes, densities, pca_var_pc1, pca_var_pc2, pca_var_pc3, flatnesses, elongations, roughnesses, \
        height_variabilities, skewnesses, curvatures, mean_nn_distances, mean_intensities, std_intensities, \
        skew_intensities = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    voxel_volume = voxel_size ** 3

    for voxel_label, point_indices in voxel_to_points.items():
        points_in_voxel = pointcloud[point_indices]
        labels_in_voxel = labels[point_indices]

        if len(points_in_voxel) < min_len:
            continue

        unique_labels, label_counts = np.unique(labels_in_voxel, return_counts=True)
        majority_label = unique_labels[np.argmax(label_counts)]
        majority_proportion = label_counts.max() / label_counts.sum()

        if majority_proportion < proportion_threshold:
            continue

        retained_points += len(points_in_voxel)
        voxel_planes_, rmse_ = pf.compute_voxel_planes(points_in_voxel, np.array([voxel_label] * len(points_in_voxel)))
        lbp_patterns = apply_dynamic_lbp_to_voxels(points_in_voxel, labels_in_voxel, voxel_labels_[point_indices],
                                                   k_neighbors=k_neighbors)

        if voxel_label not in rmse_:
            continue

        rmse_value = rmse_[voxel_label]
        all_voxel_rmse.append(rmse_value)
        num_points = len(points_in_voxel)
        num_points_list.append(num_points)
        density_list.append(num_points / voxel_volume)
        majority_labels.append(majority_label)
        lbp_value = lbp_patterns.get(voxel_label, 0)
        lbp_list.append(lbp_value)
        class_categories.append(map_labels_to_categories(label_mapping, majority_label))

        # Calculate additional features for each voxel
        convex_hull_volumes.append(compute_convex_hull_volume(points_in_voxel))
        densities.append(compute_density(points_in_voxel))
        variance_ratios, flatness, elongation = compute_pca(points_in_voxel)
        # Split PCA variance ratios into separate components
        pca_var_pc1.append(variance_ratios[0])
        pca_var_pc2.append(variance_ratios[1])
        pca_var_pc3.append(variance_ratios[2])
        flatnesses.append(flatness)
        elongations.append(elongation)
        roughnesses.append(compute_surface_roughness(points_in_voxel))
        height_var, skewness = compute_height_variability(points_in_voxel)
        height_variabilities.append(height_var)
        skewnesses.append(skewness)
        curvatures.append(compute_curvature(points_in_voxel))
        mean_nn_distances.append(compute_mean_nearest_neighbor_distance(points_in_voxel))
        mean_intensity, std_intensity, skew_intensity = compute_intensity_features(points_in_voxel)
        mean_intensities.append(mean_intensity)
        std_intensities.append(std_intensity)
        skew_intensities.append(skew_intensity)

    retention_percentage = (retained_points / total_points) * 100
    print(f"Total initial points: {total_points}")
    print(f"Points retained after filtering: {retained_points}")
    print(f"Percentage of data retained after filtering: {retention_percentage:.2f}%")

    # Create dataframe with all features
    data = {
        'RMSE': all_voxel_rmse,
        'Num_Points': num_points_list,
        'Density': density_list,
        'LBP': lbp_list,
        'Class': majority_labels,
        'Category': class_categories,
        'ConvexHullVolume': convex_hull_volumes,
        'PCA_Var_PC1': pca_var_pc1,
        'PCA_Var_PC2': pca_var_pc2,
        'PCA_Var_PC3': pca_var_pc3,
        'Flatness': flatnesses,
        'Elongation': elongations,
        'SurfaceRoughness': roughnesses,
        'HeightVariability': height_variabilities,
        'HeightSkewness': skewnesses,
        'Curvature': curvatures,
        'MeanNN_Distance': mean_nn_distances,
        'MeanIntensity': mean_intensities,
        'StdIntensity': std_intensities,
        'SkewIntensity': skew_intensities
    }
    df = pd.DataFrame(data)
    return df


def run_single_image_list_test_with_lbp_and_obstacle_classification(
        lidar_dir, labels_dir, csv_file, image_list, label_mapping, iterations=1, num_voxels=1000, k_neighbors=5,
        z_threshold=1, min_len=10, proportion_threshold=0.5):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"classificationTest_{timestamp}_{os.path.basename(lidar_dir)}"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare initial parameters text
    params_text = f"""
        Classification Test Parameters:

        1. Voxel Size: 5
        2. Passable Cost: 1
        3. Non-Passable Cost: 100
        4. Number of Iterations: {iterations}
        5. Number of Voxels: {num_voxels}
        6. K Neighbors (for LBP): {k_neighbors}
        7. Z Threshold: {z_threshold}
        8. Minimum Points per Voxel (min_len): {min_len}
        9. Majority Proportion Threshold: {proportion_threshold}

        Directories and Data Files:
        - LiDAR Directory: {lidar_dir}
        - Labels Directory: {labels_dir}
        - CSV Label Mapping File: {csv_file}

        Image List:
        """ + "\n".join([f"- {img}" for img in image_list]) + f"""

        Label Mapping:
        {label_mapping}
        """

    params_file_path = os.path.join(output_dir, "classification_test_parameters_and_metrics.txt")
    with open(params_file_path, "w") as file:
        file.write(params_text)
    os.makedirs(output_dir, exist_ok=True)


    location_name = os.path.basename(lidar_dir)
    if location_name not in location_accuracies:
        location_accuracies[location_name] = []

    # Proceed with the classification and record timing
    results = []
    for i in range(iterations):
        print(f"Running real data test iteration {i + 1}/{iterations}...")
        dfs = []

        for prefix in image_list:
            # Start timing for this image processing
            start_time = time.time()

            df = process_image_with_metrics_and_lbp_and_obstacle_classification(
                prefix, lidar_dir, labels_dir, csv_file, label_mapping, z_threshold=z_threshold,
                voxel_size=5, num_voxels=num_voxels, k_neighbors=k_neighbors, min_len=min_len,
                proportion_threshold=proportion_threshold)

            # End timing for this image processing
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken for image {prefix}: {elapsed_time:.4f} seconds")

            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        X = combined_df.drop(columns=['Class', 'Category'])
        y = combined_df['Category']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        prediction_start_time = time.time()

        y_pred = clf.predict(X_test)

        # End timing for prediction
        prediction_end_time = time.time()
        prediction_elapsed_time = prediction_end_time - prediction_start_time
        print(f"Time taken for Random Forest prediction: {prediction_elapsed_time:.4f} seconds")

        accuracy = accuracy_score(y_test, y_pred)
        location_accuracies[location_name].append(accuracy)

        # Capture and print classification metrics as usual
        class_report = classification_report(y_test, y_pred, output_dict=True)
        print(class_report)
        metrics_text = f"\nIteration {i + 1} Metrics:\n- Accuracy: {accuracy:.4f}\n\n"

        for label, metrics in class_report.items():
            if label in {'accuracy', 'macro avg', 'weighted avg'}:
                continue
            f1_score = metrics['f1-score']
            recall = metrics['recall']
            metrics_text += f"Label {label}:\n  - F1 Score: {f1_score:.4f}\n  - Recall: {recall:.4f}\n\n"
            f1_scores[location_name].append(f1_score)
            recall_scores[location_name].append(recall)

        # Append metrics to text file
        with open(params_file_path, "a") as file:
            file.write(metrics_text)

        # Plot confusion matrix for each iteration
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test),
                    yticklabels=np.unique(y_test))
        plt.title(f'Confusion Matrix - Iteration {i + 1}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        sanitized_lidar_dir = lidar_dir.replace('/', '_').replace('..', '_')
        plot_filename = os.path.join(output_dir, f'confusion_matrix_{i + 1}_{sanitized_lidar_dir}.png')
        plt.savefig(plot_filename)
        plt.close()

        # Append results to overall metrics
        results.append({
            'Iteration': i + 1,
            'Accuracy': accuracy,
            'Classification Report': class_report
        })

        combined_df.to_csv(os.path.join(output_dir, f'iteration_{i + 1}_real_data.csv'), index=False)

    # Save final results as a summary CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'classification_real_data_results.csv'), index=False)


def plot_f1_recall_by_label(f1_scores, recall_scores):
    """
    Plots F1 and Recall scores for each label across different datasets using specific category labels.

    Parameters:
    - f1_scores (dict): Dictionary with dataset names as keys and lists of F1 scores for each category.
    - recall_scores (dict): Dictionary with dataset names as keys and lists of Recall scores for each category.
    """

    # Define specific category labels as mapped in `map_labels_to_categories`
    category_labels = [
        "Obstacle",  # Category 0
        "Cobble",  # Category 1
        "Gravel",  # Category 2
        "Sidewalk",  # Category 3
        "Soil",  # Category 4
        "High Grass",  # Category 5
        "Low Grass",  # Category 6
        "Other"  # Category 7
    ]

    # Extract unique dataset locations for plotting
    datasets = list(f1_scores.keys())

    for dataset in datasets:
        # Get the F1 and recall scores for each category in this dataset, filling missing data with zeros
        f1_values = np.array(f1_scores[dataset] + [0] * (len(category_labels) - len(f1_scores[dataset])))
        recall_values = np.array(recall_scores[dataset] + [0] * (len(category_labels) - len(recall_scores[dataset])))

        # Plot F1 and recall scores
        plt.figure(figsize=(12, 6))
        bar_width = 0.35
        index = np.arange(len(category_labels))

        plt.bar(index, f1_values, bar_width, label='F1 Score', alpha=0.7)
        plt.bar(index + bar_width, recall_values, bar_width, label='Recall', alpha=0.7)

        plt.xlabel('Categories')
        plt.ylabel('Score')
        plt.title(f'F1 and Recall Scores for {dataset}')
        plt.xticks(index + bar_width / 2, category_labels, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()



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

    lidar_dir2 = '../goose_3d_val/lidar/val/2022-08-30_siegertsbrunn_feldwege'
    labels_dir2 = '../goose_3d_val/labels/val/2022-08-30_siegertsbrunn_feldwege'
    # Image lists for comparison
    image_list2 = [
        '2022-08-30_siegertsbrunn_feldwege__0528_1661860582736903436',
        '2022-08-30_siegertsbrunn_feldwege__0529_1661860582946925890',
        '2022-08-30_siegertsbrunn_feldwege__0530_1661860585992333325',
        '2022-08-30_siegertsbrunn_feldwege__0531_1661860589143543478',
        '2022-08-30_siegertsbrunn_feldwege__0532_1661860591874117478',
    ]

    lidar_dir3 = '../goose_3d_val/lidar/val/2022-12-07_aying_hills'
    labels_dir3 = '../goose_3d_val/labels/val/2022-12-07_aying_hills'
    # Image lists for comparison
    image_list3 = [
        '2022-12-07_aying_hills__0006_1670420708448844860',
        '2022-12-07_aying_hills__0009_1670420878948219746',
        '2022-12-07_aying_hills__0010_1670420972132205304',
        '2022-12-07_aying_hills__0011_1670420979760256580',
        '2022-12-07_aying_hills__0012_1670420985739069345',
    ]

    lidar_dir4 = '../goose_3d_val/lidar/val/2022-09-21_garching_uebungsplatz_2'
    labels_dir4 = '../goose_3d_val/labels/val/2022-09-21_garching_uebungsplatz_2'

    image_list4 = [
        '2022-09-21_garching_uebungsplatz_2__0000_1663755178980462982',
        '2022-09-21_garching_uebungsplatz_2__0001_1663755182709213081',
        '2022-09-21_garching_uebungsplatz_2__0002_1663755194931038715',
        '2022-09-21_garching_uebungsplatz_2__0003_1663755204459982406',
        '2022-09-21_garching_uebungsplatz_2__0004_1663755209017252315',
    ]

    lidar_dir5 = '../goose_3d_val/lidar/val/2023-01-20_aying_mangfall_2'
    labels_dir5 = '../goose_3d_val/labels/val/2023-01-20_aying_mangfall_2'

    image_list5 = [
        '2023-01-20_aying_mangfall_2__0402_1674223287010374431',
        '2023-01-20_aying_mangfall_2__0403_1674223290055201300',
        '2023-01-20_aying_mangfall_2__0404_1674223294780692415',
        '2023-01-20_aying_mangfall_2__0405_1674223297089695576',
        '2023-01-20_aying_mangfall_2__0406_1674223303390449467',
    ]

    lidar_dir6 = '../goose_3d_val/lidar/val/2023-03-03_garching_2'
    labels_dir6 = '../goose_3d_val/labels/val/2023-03-03_garching_2'

    image_list6 = [
        '2023-03-03_garching_2__0114_1677850404544550420',
        '2023-03-03_garching_2__0115_1677850409391907174',
        '2023-03-03_garching_2__0116_1677850414554513656',
        '2023-03-03_garching_2__0117_1677850415819237733',
        '2023-03-03_garching_2__0118_1677850418032002589',
    ]

    lidar_dir7 = '../goose_3d_val/lidar/val/2023-05-15_neubiberg_rain'
    labels_dir7 = '../goose_3d_val/labels/val/2023-05-15_neubiberg_rain'

    image_list7 = [
        '2023-05-15_neubiberg_rain__0623_1684157849628053530',
        '2023-05-15_neubiberg_rain__0624_1684157852529822670',
        '2023-05-15_neubiberg_rain__0625_1684157854913211132',
        '2023-05-15_neubiberg_rain__0626_1684157858851042175',
        '2023-05-15_neubiberg_rain__0627_1684157862063584296',
    ]

    lidar_dir8 = '../goose_3d_val/lidar/val/2023-05-17_neubiberg_sunny'
    labels_dir8 = '../goose_3d_val/labels/val/2023-05-17_neubiberg_sunny'

    image_list8 = [
        '2023-05-17_neubiberg_sunny__0379_1684329742550906590',
        '2023-05-17_neubiberg_sunny__0380_1684329745349077704',
        '2023-05-17_neubiberg_sunny__0381_1684329746496937615',
        '2023-05-17_neubiberg_sunny__0382_1684329747629654308',
        '2023-05-17_neubiberg_sunny__0383_1684329748563080364',
    ]

    run_single_image_list_test_with_lbp_and_obstacle_classification(lidar_dir, labels_dir, csv_file, image_list,
                                                                  label_mapping, iterations=1, num_voxels=1000,
                                                                    k_neighbors=5)

    #run_single_image_list_test_with_lbp_and_obstacle_classification(lidar_dir2, labels_dir2, csv_file, image_list2,
    #                                                                label_mapping, iterations=1, num_voxels=1000,
    #                                                                k_neighbors=5)

    #run_single_image_list_test_with_lbp_and_obstacle_classification(lidar_dir3, labels_dir3, csv_file, image_list3,
    #                                                                label_mapping, iterations=1, num_voxels=1000,
    #                                                                k_neighbors=5)

    #run_single_image_list_test_with_lbp_and_obstacle_classification(lidar_dir4, labels_dir4, csv_file, image_list4,
    #                                                                label_mapping, iterations=1, num_voxels=1000, k_neighbors=5)

    #run_single_image_list_test_with_lbp_and_obstacle_classification(lidar_dir5, labels_dir5, csv_file, image_list5,
    #                                                                label_mapping, iterations=1, num_voxels=1000,
    #                                                                k_neighbors=5)
    #run_single_image_list_test_with_lbp_and_obstacle_classification(lidar_dir6, labels_dir6, csv_file, image_list6,
    #                                                                label_mapping, iterations=1, num_voxels=1000,
    #                                                                k_neighbors=5)

    #run_single_image_list_test_with_lbp_and_obstacle_classification(lidar_dir7, labels_dir7, csv_file, image_list7,
    #                                                                label_mapping, iterations=1, num_voxels=1000,
    #                                                                k_neighbors=5)

    #run_single_image_list_test_with_lbp_and_obstacle_classification(lidar_dir8, labels_dir8, csv_file, image_list8,
    #                                                                label_mapping, iterations=1, num_voxels=1000,
    #                                                                k_neighbors=5)

    locations = list(location_accuracies.keys())
    accuracies = list(location_accuracies.values())

    plot_f1_recall_by_label(f1_scores, recall_scores)



