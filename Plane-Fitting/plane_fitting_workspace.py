from Voxelization import voxel3d as v3d
import plane_Fitting as pf
import numpy as np

# The following paths require the "testing data" to be at the same level as the script
prefix = '2022-07-22_flight__0254_1658494839082804823'
lidar_dir = 'goose_3d_val/lidar/val/2022-07-22_flight'
labels_dir = 'goose_3d_val/labels/val/2022-07-22_flight'
csv_file = 'goose_3d_val/goose_label_mapping.csv'

lidar_data, lidar_labels, label_metadata = pf.load_pointcloud_and_labels(prefix, lidar_dir, labels_dir, csv_file)

z_threshold = 1
pointcloud, labels = pf.apply_threshold(lidar_data, lidar_labels, z_threshold)

voxel_labels, voxel_map = v3d.voxelize_point_cloud_2d(pointcloud, voxel_size=1)

n_insufficient_voxels = 0
for i in np.unique(voxel_labels):
    if np.sum(voxel_labels == i) < 4:
        n_insufficient_voxels += 1

print(f"Number of voxels with less than 4 points: {n_insufficient_voxels}")
print(f"Proportion of voxels with less than 4 points: {n_insufficient_voxels / len(np.unique(voxel_labels))}")
# TODO : Maybe remove when len(points) == 3 because it's not enough to compute residuals

pf.visualize_selected_points(pointcloud, voxel_labels)
# pf.visualize_selected_points(pointcloud, labels, label_metadata)

voxel_planes, residuals = pf.compute_voxel_planes(pointcloud, voxel_labels)

pf.plot_voxel_map(voxel_map, residuals, save_and_open=True, output_file='high_res_voxel_map.png', dpi=300)

import pandas as pd

# Initialize a list to store the results
data = []
pointcloud = lidar_data

for i in np.unique(lidar_labels):
    # for i in [23, 51]:

    filtered_pointcloud = pointcloud[pointcloud[:, 3] == i, :]
    # print(f"Pointcloud shape for label {i}: {filtered_pointcloud.shape}")
    # print(filtered_pointcloud)
    # print(f"len(filtered_pointcloud): {len(filtered_pointcloud)}")
    # Voxelize point cloud
    voxel_labels_, voxel_map_ = v3d.voxelize_point_cloud_2d(filtered_pointcloud, voxel_size=30)

    # Compute voxel planes and residuals
    voxel_planes_, residuals_ = pf.compute_voxel_planes(filtered_pointcloud, voxel_labels_)

    # Calculate number of points and average residuals
    if len(residuals_) == 0:
        continue
    number_of_points_ = len(filtered_pointcloud)
    average_residual_ = sum(residuals_.values()) / len(residuals_.values())

    # Append results to the list
    data.append({
        'label': i,
        'residuals': residuals_,
        'voxel_map': voxel_map_,
        'voxel_labels': voxel_labels_,
        'number_of_points': number_of_points_,
        'average_residual': average_residual_,
        'pointcloud': filtered_pointcloud,
        'voxel_number': len(np.unique(voxel_labels_))
    })

# Convert the list into a pandas DataFrame
df = pd.DataFrame(data)

# Select the relevant columns: 'label', 'number_of_points', and 'average_residual'
residuals_df = df

# Perform an inner join (merge) with label_metadata on 'label' (equivalent to 'label_key')
merged_df = pd.merge(label_metadata, residuals_df, left_on='label_key', right_on='label')

# Drop the 'has_instance' and 'hex' columns
merged_df = merged_df.drop(columns=['has_instance', 'hex'])

# Reorder the columns: 'label', 'class_name', 'number_of_points', 'average_residual'
merged_df = merged_df[['label', 'class_name', 'number_of_points', 'average_residual', 'voxel_number']]

# Sort by the average residuals
sorted_merged_df = merged_df.sort_values(by='average_residual', ascending=True)

# Display the sorted DataFrame
print(sorted_merged_df)
# TODO : This is incoherent : "alsphalt" scores the worst...

