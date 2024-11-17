import os
import pickle
import logging
import pandas as pd
import numpy as np
from VoxelGrid import VoxelGrid
import import_helper as ih
import voxel_utils as vu
from multiprocessing import Pool, cpu_count
def process_single_dataset(dataset, voxel_size, min_num_points, min_proportion, use_selected_features=False):
    logging.info(f"Processing dataset: {dataset}")
    max_pcs = ih.get_max_pointclouds_count(dataset)
    
    # List to hold all voxel data for this dataset
    dataset_voxel_data = []

    for index in range(max_pcs):
        logging.info(f"Processing point cloud {index + 1}/{max_pcs} from dataset {dataset}")

        # Load point cloud and label data
        try:
            points, labels = ih.import_pc_and_labels(dataset, index)
        except Exception as e:
            logging.error(f"Failed to load point cloud {index} from dataset {dataset}: {e}")
            continue

        # Initialize and process the VoxelGrid
        voxel_grid = VoxelGrid(voxel_size=voxel_size)
        voxel_grid.add_points(points, labels)
        voxel_grid.filter_voxels(min_num_points, min_proportion)
        dominant_labels = voxel_grid.get_dominant_labels_with_proportion()

        # Get statistics based on the selected mode
        if use_selected_features:
            stats = voxel_grid.get_selected_features_statistics()
        else:
            stats = voxel_grid.get_voxel_statistics()

        voxel_centers = voxel_grid.get_voxel_centers()

        # Collect only the required voxel data
        for voxel_key, center in zip(voxel_grid.voxel_dict.keys(), voxel_centers):
            voxel_info = {
                'dataset': dataset,
                'voxel_key': voxel_key,
                'center_x': center[0],
                'center_y': center[1],
                'num_points': len(voxel_grid.voxel_dict[voxel_key]),
            }

            # Add statistics conditionally

            voxel_info.update({
                'dominant_label': dominant_labels.get(voxel_key, {}).get('label', None),
                'dominant_proportion': dominant_labels.get(voxel_key, {}).get('proportion', None),
                'avg_intensity': stats.get('avg_intensity', {}).get(voxel_key, None),
                'elevation_range': stats.get('elevation_range', {}).get(voxel_key, None),
                'plane_coef_3': stats.get('plane_coef_3', {}).get(voxel_key, None),
                'RMSE': stats.get('RMSE', {}).get(voxel_key, None),
                'flatness': stats.get('flatness', {}).get(voxel_key, None),
                'elongation': stats.get('elongation', {}).get(voxel_key, None),
            })
            if not use_selected_features:
                voxel_info.update({
                    'convex_hull_volume': stats.get('convex_hull_volume', {}).get(voxel_key, None),
                    'density': stats.get('density', {}).get(voxel_key, None),
                    'std_intensity': stats.get('std_intensity', {}).get(voxel_key, None),
                    'vertical_skewness': stats.get('vertical_skewness', {}).get(voxel_key, None),
                    'mean_nn_distance': stats.get('mean_nn_distance', {}).get(voxel_key, None),
                })

            # Replace None values with np.nan
            voxel_info = {k: (v if v is not None else np.nan) for k, v in voxel_info.items()}
            dataset_voxel_data.append(voxel_info)

    return dataset_voxel_data






def process_voxel_data(
    voxel_size=1,
    min_num_points=10,
    min_proportion=0.6,
    output_file="processed_voxel_data.local/all_datasets_voxel_data.pkl",
    use_selected_features=False
):
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize logger
    logging.basicConfig(level=logging.INFO)

    # Get datasets from import_helper
    datasets = list(ih.DATASET_MAP.keys())

    # Use multiprocessing to process each dataset in parallel
    with Pool(cpu_count()) as pool:
        results = pool.starmap(
            process_single_dataset,
            [(dataset, voxel_size, min_num_points, min_proportion, use_selected_features) for dataset in datasets]
        )

    # Combine all results into a single list and add metadata
    all_voxel_data = {
        'metadata': {
            'voxel_size': voxel_size,
            'min_num_points': min_num_points,
            'min_proportion': min_proportion,
            'datasets': datasets
        },
        'voxel_data': [voxel_info for dataset_voxel_data in results for voxel_info in dataset_voxel_data]
    }

    # Save to a single pickle file
    with open(output_file, "wb") as f:
        pickle.dump(all_voxel_data, f)

    logging.info(f"All voxel data has been pickled and saved to {output_file}")

if __name__ == "__main__":
    # CHANGE THIS TO True TO USE THE SELECTED FEATURES
    use_selected_features = True
    
        
    file_path = os.path.dirname(os.path.abspath(__file__))
    if use_selected_features:
        output_file = os.path.join(file_path, "processed_voxel_data_reduced.local/all_datasets_voxel_data.pkl")
    else:
        output_file = os.path.join(file_path, "processed_voxel_data.local/all_datasets_voxel_data.pkl")

    process_voxel_data(
        voxel_size=1,
        min_num_points=20,
        min_proportion=1.0,
        output_file=output_file,
        use_selected_features=True
    )
