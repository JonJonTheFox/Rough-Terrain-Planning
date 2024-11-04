import os
import pickle
import logging
import pandas as pd
import numpy as np
from VoxelGrid import VoxelGrid
import import_helper as ih
import voxel_utils as vu
from multiprocessing import Pool, cpu_count

def process_single_dataset(dataset, voxel_size, min_num_points, min_proportion):
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

        # Get statistics and features for each voxel
        stats = voxel_grid.get_voxel_statistics()
        voxel_centers = voxel_grid.get_voxel_centers()
        dominant_labels = voxel_grid.get_dominant_labels_with_proportion()

        # Collect each voxel's data
        for voxel_key, center in zip(voxel_grid.voxel_dict.keys(), voxel_centers):
            voxel_info = {
                'dataset': dataset,
                'voxel_key': voxel_key,
                'center_x': center[0],
                'center_y': center[1],
                'num_points': len(voxel_grid.voxel_dict[voxel_key]),
                'dominant_label': dominant_labels[voxel_key]['label'],
                'dominant_proportion': dominant_labels[voxel_key]['proportion'],
                'RMSE': stats['voxel_plane_rmse'][voxel_key],
                'plane_coefficients': stats['voxel_plane_coefficients'][voxel_key],
                'vertical_skewness': stats['vertical_skewness'][voxel_key],
                'convex_hull_volume': stats['convex_hull_volume'][voxel_key],
                'density': stats['density'][voxel_key],
                'elevation_range': stats['elevation_range'][voxel_key],
                'avg_intensity': stats['avg_intensity'][voxel_key]
            }

            # Replace None values with np.nan
            voxel_info = {k: (v if v is not None else np.nan) for k, v in voxel_info.items()}
            dataset_voxel_data.append(voxel_info)

    return dataset_voxel_data

def process_voxel_data(
    voxel_size=1,
    min_num_points=20,
    min_proportion=0.8,
    output_file="processed_voxel_data.local/all_datasets_voxel_data.pkl"
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
            [(dataset, voxel_size, min_num_points, min_proportion) for dataset in datasets]
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
    file_path = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(file_path, "processed_voxel_data.local/all_datasets_voxel_data.pkl")
    # Example usage with parameters
    process_voxel_data(
        voxel_size=1,
        min_num_points=20,
        min_proportion=0.8,
        output_file=output_file
    )
