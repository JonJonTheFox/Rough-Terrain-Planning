import os
import logging
import numpy as np
import pandas as pd

# Set up logging for production
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Base directory for dataset storage
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Short name mappings for the datasets
DATASET_MAP = {
    'flight': '2022-07-22_flight',
    'siegertsbrunn': '2022-08-30_siegertsbrunn_feldwege',
    'garching': '2022-09-21_garching_uebungsplatz_2',
    'aying_hills': '2022-12-07_aying_hills',
    'aying_mangfall': '2023-01-20_aying_mangfall_2',
    'garching_2': '2023-03-03_garching_2',
    'neubiberg_rain': '2023-05-15_neubiberg_rain',
    'neubiberg_sunny': '2023-05-17_neubiberg_sunny',
}

def resolve_dataset_path(short_name):
    if short_name not in DATASET_MAP:
        logging.error(f"Unknown dataset short name: {short_name}")
        raise ValueError(f"Unknown dataset short name: {short_name}")
    return DATASET_MAP[short_name]

def load_label_metadata():
    csv_file = os.path.join(CURRENT_DIR, 'goose_3d_val/goose_label_mapping.csv')
    try:
        label_metadata = pd.read_csv(csv_file)
        logging.info("Label metadata loaded successfully")
        return label_metadata
    except Exception as e:
        logging.error(f"Error loading label metadata: {e}")
        raise

def get_max_pointclouds_count(short_name):
    """
    Returns the maximum number of point clouds available in the dataset.
    
    Parameters:
    - short_name: Dataset short name (e.g., 'flight')
    
    Returns:
    - Integer count of point cloud files available in the dataset.
    """
    dataset_name = resolve_dataset_path(short_name)
    lidar_dir = os.path.join(CURRENT_DIR, f'goose_3d_val/lidar/val/{dataset_name}')
    
    try:
        all_files = [f for f in os.listdir(lidar_dir) if f.endswith('.bin')]
        pointcloud_count = len(all_files)
        logging.info(f"{pointcloud_count} point clouds found for {short_name}")
        return pointcloud_count
    except Exception as e:
        logging.error(f"Error reading directory {lidar_dir}: {e}")
        raise

def import_pc_and_labels(short_name, index, lidar_suffix='vls128', label_suffix='goose'):
    dataset_name = resolve_dataset_path(short_name)

    lidar_dir = os.path.join(CURRENT_DIR, f'goose_3d_val/lidar/val/{dataset_name}')
    labels_dir = os.path.join(CURRENT_DIR, f'goose_3d_val/labels/val/{dataset_name}')

    try:
        all_files = [f for f in os.listdir(lidar_dir) if f.endswith('.bin')]
        flight_unique_ids = [file_name.split('__')[1].replace('.bin', '') for file_name in all_files]

        if not flight_unique_ids:
            logging.error(f"No point cloud files found in {lidar_dir}")
            raise FileNotFoundError(f"No point cloud files found in {lidar_dir}")

        flight_unique_ids.sort()
    except Exception as e:
        logging.error(f"Error reading {lidar_dir}: {e}")
        raise

    if index >= len(flight_unique_ids):
        logging.error(f"Index {index} out of range. Available indices: 0 to {len(flight_unique_ids) - 1}")
        raise IndexError(f"Index {index} out of range for flight IDs.")

    prefix = f'{dataset_name}__{flight_unique_ids[index]}'

    lidar_file = f"{lidar_dir}/{prefix}.bin"
    label_file = f"{labels_dir}/{prefix.replace(lidar_suffix, label_suffix)}.label"

    try:
        lidar_data = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        labels = np.fromfile(label_file, dtype=np.uint32)
        logging.info(f"Loaded point cloud and labels for {prefix}")
    except Exception as e:
        logging.error(f"Failed to load point cloud or labels for {prefix}: {e}")
        raise

    return lidar_data, labels

def import_multiple_pc_and_labels(short_name, start_index, end_index, lidar_suffix='vls128', label_suffix='goose'):
    """
    Load multiple point clouds and labels between the specified indices (inclusive).
    
    Parameters:
    - short_name: Dataset short name (e.g., 'flight')
    - start_index: Start index for point clouds
    - end_index: End index for point clouds
    - lidar_suffix: Suffix for LIDAR files (default 'vls128')
    - label_suffix: Suffix for label files (default 'goose')
    
    Returns:
    - List of tuples, each containing (pointcloud, labels) for each index.
    """
    dataset_name = resolve_dataset_path(short_name)
    lidar_dir = os.path.join(CURRENT_DIR, f'goose_3d_val/lidar/val/{dataset_name}')

    try:
        all_files = [f for f in os.listdir(lidar_dir) if f.endswith('.bin')]
        flight_unique_ids = [file_name.split('__')[1].replace('.bin', '') for file_name in all_files]

        if not flight_unique_ids:
            logging.error(f"No point cloud files found in {lidar_dir}")
            raise FileNotFoundError(f"No point cloud files found in {lidar_dir}")

        flight_unique_ids.sort()
    except Exception as e:
        logging.error(f"Error reading {lidar_dir}: {e}")
        raise

    if end_index >= len(flight_unique_ids) or start_index < 0:
        logging.error(f"Indices out of range. Available indices: 0 to {len(flight_unique_ids) - 1}")
        raise IndexError(f"Indices out of range for flight IDs.")

    pointclouds_and_labels = []
    for index in range(start_index, end_index + 1):
        prefix = f'{dataset_name}__{flight_unique_ids[index]}'
        lidar_file = f"{lidar_dir}/{prefix}.bin"
        label_file = f"{lidar_dir.replace('lidar', 'labels')}/{prefix.replace(lidar_suffix, label_suffix)}.label"
        
        try:
            lidar_data = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
            labels = np.fromfile(label_file, dtype=np.uint32)
            pointclouds_and_labels.append((lidar_data, labels))
            logging.info(f"Loaded point cloud and labels for {prefix}")
        except Exception as e:
            logging.error(f"Failed to load point cloud or labels for {prefix}: {e}")
            raise

    return pointclouds_and_labels


def get_metadata_map(label_metadata):
    """
    Create a mapping from label_key (int) to class_name (str) from the label metadata DataFrame.

    Parameters:
    - label_metadata: DataFrame containing the label metadata with 'label_key' and 'class_name' columns.

    Returns:
    - metadata_map: A dictionary mapping label_key (int) to class_name (str).
    """
    try:
        metadata_map = pd.Series(label_metadata['class_name'].values, index=label_metadata['label_key']).to_dict()
        logging.info("Metadata map successfully created.")
        return metadata_map
    except KeyError as e:
        logging.error(f"Missing required columns in label_metadata DataFrame: {e}")
        raise
    except Exception as e:
        logging.error(f"Failed to create metadata map: {e}")
        raise
    
def get_hex_map(label_metadata):
    """
    Create a mapping from label_key (int) to hex_color (str) from the label metadata DataFrame.

    Parameters:
    - label_metadata: DataFrame containing the label metadata with 'label_key' and 'hex_color' columns.

    Returns:
    - hex_map: A dictionary mapping label_key (int) to hex_color (str).
    """
    try:
        hex_map = pd.Series(label_metadata['hex'].values, index=label_metadata['label_key']).to_dict()
        logging.info("Hex color map successfully created.")
        return hex_map
    except KeyError as e:
        logging.error(f"Missing required columns in label_metadata DataFrame: {e}")
        raise
    except Exception as e:
        logging.error(f"Failed to create hex color map: {e}")
        raise
    
def hex_to_rgb(hex_color):
    """
    Convert a hex color string to an RGB tuple.

    Parameters:
    - hex_color: Hex color string (e.g., '#FFAAFF')

    Returns:
    - rgb_tuple: A tuple containing RGB values (e.g., (255, 170, 255))
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_rgb_map(label_metadata):
    """
    Create a mapping from label_key (int) to RGB color (tuple) from the label metadata DataFrame.

    Parameters:
    - label_metadata: DataFrame containing the label metadata with 'label_key' and 'hex_color' columns.

    Returns:
    - rgb_map: A dictionary mapping label_key (int) to RGB tuple (e.g., (255, 170, 255)).
    """
    try:
        # Convert hex color to RGB and create the mapping
        rgb_map = pd.Series(
            [hex_to_rgb(hex_color) for hex_color in label_metadata['hex'].values], 
            index=label_metadata['label_key']
        ).to_dict()
        
        logging.info("RGB color map successfully created.")
        return rgb_map
    except KeyError as e:
        logging.error(f"Missing required columns in label_metadata DataFrame: {e}")
        raise
    except Exception as e:
        logging.error(f"Failed to create RGB color map: {e}")
        raise

def get_label_name(label_key, metadata_map):
    """
    Convert a label key (int) to its corresponding class name (str).

    Parameters:
    - label_key: The integer key for the label.
    - metadata_map: Dictionary mapping label_key to class_name.

    Returns:
    - class_name: The string class name corresponding to the label_key, or None if not found.
    """
    try:
        return metadata_map.get(label_key, None)
    except KeyError:
        logging.warning(f"Label key {label_key} not found in metadata map.")
        return None


if __name__ == "__main__":
    # Get maximum number of point clouds for a dataset
    max_pcs = get_max_pointclouds_count('flight')
    print(f"Max point clouds for 'flight': {max_pcs}")

    # Load a point cloud and corresponding labels
    pointcloud, labels = import_pc_and_labels('flight', 2)

    # Load metadata and create a map
    metadata = load_label_metadata()
    metadata_map = get_metadata_map(metadata)

    # Example: Convert a label key to label name
    label_name = get_label_name(3, metadata_map)
    print(f"Label 3 corresponds to: {label_name}")

    # Load multiple point clouds
    pcs_labels = import_multiple_pc_and_labels('flight', 1, 3)
    for i, (pc, label) in enumerate(pcs_labels):
        print(f"Point Cloud {i+1}: {pc.shape}, Labels: {len(label)}")
