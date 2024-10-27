# Import necessary libraries
from Voxelization import voxel3d as v3d
import plane_Fitting as pf
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Reload modules if necessary (useful for debugging or development)
import importlib
importlib.reload(v3d)
importlib.reload(pf)

# Define file paths
prefix = '2022-07-22_flight__0254_1658494839082804823'
lidar_dir = '../goose_3d_val/lidar/val/2022-07-22_flight'
labels_dir = '../goose_3d_val/labels/val/2022-07-22_flight'
csv_file = '../goose_3d_val/goose_label_mapping.csv'
flight_unique_ids = [
    '0071_1658494234334310308', '0072_1658494235988100385', '0073_1658494238675704025',
    '0075_1658494242083534022', '0077_1658494244047191404', '0078_1658494246011438339',
    '0080_1658494248904312093', '0082_1658494251900354294'
]
prefixes = ['2022-07-22_flight__' + id for id in flight_unique_ids]

# Initialize lists for data storage
all_pointclouds = []
all_lidar_labels = []
all_label_metadata = []

# Load data for each prefix
for prefix in prefixes:
    lidar_data, lidar_labels, label_metadata = pf.load_pointcloud_and_labels(prefix, lidar_dir, labels_dir, csv_file)
    all_pointclouds.append(lidar_data)
    all_lidar_labels.append(lidar_labels)
    all_label_metadata.append(label_metadata)

# Load and process label mapping
df = pd.read_csv('../goose_3d_val/goose_label_mapping.csv')
label_to_class = dict(zip(df['label_key'], df['class_name']))

# Voxelize point clouds
all_voxel_labels = []
all_voxel_maps = []
all_unique_voxel_labels = []

for i, pointcloud in enumerate(all_pointclouds):
    voxel_labels, voxel_map, unique_voxel_labels = v3d.voxelize_point_cloud_2d(pointcloud, voxel_size=1)
    all_voxel_labels.append(voxel_labels)
    all_voxel_maps.append(voxel_map)
    all_unique_voxel_labels.append(unique_voxel_labels)

# Preprocess voxels
all_map_to_majority = []
all_voxel_pointclouds = []
all_voxel_ids_after_preprocessing = []

for i, pointcloud in enumerate(all_pointclouds):
    voxel_labels = all_voxel_labels[i]
    labels = all_lidar_labels[i]
    map_to_majority, voxel_pointclouds, voxel_ids_after_preprocessing = pf.preprocess_voxels(
        voxel_labels=voxel_labels, pointcloud=pointcloud, labels=labels, min_len=10, proportion_threshold=0.7
    )
    all_map_to_majority.append(map_to_majority)
    all_voxel_pointclouds.append(voxel_pointclouds)
    all_voxel_ids_after_preprocessing.append(voxel_ids_after_preprocessing)

# Analyze label frequency
all_freq_df = []
for map_to_majority in all_map_to_majority:
    freq_df = pf.analyze_label_frequency(map_to_majority, label_to_class, sort_order='ascending')
    all_freq_df.append(freq_df)
    pf.print_label_frequency(freq_df)

# Filter by whitelist
whitelist = {'asphalt', 'low_grass', 'high_grass', 'hedge'}
all_voxel_ids = []
all_labels_wl = []

for i, map_to_majority in enumerate(all_map_to_majority):
    filtered_voxel_ids, labels_wl = pf.filter_voxels_by_whitelist(
        map_to_majority=map_to_majority, label_to_class=label_to_class, whitelist=whitelist
    )
    all_voxel_ids.append(filtered_voxel_ids)
    all_labels_wl.append(labels_wl)

# Run plane fitting and compute RMSE
all_plane_params = []
for i, filtered_voxel_ids in enumerate(all_voxel_ids):
    voxel_pointclouds = all_voxel_pointclouds[i]
    plane_params = pf.run_plane_fitting_on_voxels(filtered_voxel_ids=filtered_voxel_ids, voxel_pointclouds=voxel_pointclouds)
    all_plane_params.append(plane_params)

all_rmse_map = []
for i, plane_params in enumerate(all_plane_params):
    map_to_majority = all_map_to_majority[i]
    rmse_map = pf.aggregate_rmse_by_label(plane_params=plane_params, map_to_majority=map_to_majority)
    all_rmse_map.append(rmse_map)

# Plot RMSE values as violin plots
for rmse_map in all_rmse_map:
    pf.plot_rmse_violin(
        rmse_map=rmse_map, label_to_class=label_to_class, figsize=(10, 6), title='Violin Plot of RMSE Values by Label'
    )

# Extract features and targets for model training
class_to_label = {v: k for k, v in label_to_class.items()}
labels = [class_to_label[c] for c in whitelist]
n = len(all_pointclouds)
n_train = int(n * 0.8)

def extract_features_and_targets(labels, pointcloud_rmse_range):
    features = []
    targets = []
    for label in labels:
        for pointcloud_rmse in pointcloud_rmse_range:
            rmse_values = pointcloud_rmse.get(label, [])
            features += rmse_values
            targets += [label] * len(rmse_values)
    return np.array(features).reshape(-1, 1), np.array(targets)

X_train, y_train = extract_features_and_targets(labels, all_rmse_map[:n_train])
X_test, y_test = extract_features_and_targets(labels, all_rmse_map[n_train:])

# Train RandomForest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {accuracy:.4f}")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Calculate and print R² score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# Save training and test datasets
with open('../X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)
with open('../X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)
with open('../Y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)
with open('../Y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)

print("Script completed successfully!")
