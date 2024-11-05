# dataset_analyzer.py

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import logging
from voxel_utils import VoxelGrid
from import_helper import (
    import_multiple_pc_and_labels,
    get_max_pointclouds_count,
    load_label_metadata,
    get_metadata_map
)

class DatasetAnalyzer:
    def __init__(self, voxel_size: float = 1.0):
        """
        Initialize the dataset analyzer.
        
        Parameters:
        - voxel_size: Size of voxel cube edges
        """
        self.voxel_size = voxel_size
        self.label_metadata = load_label_metadata()
        self.metadata_map = get_metadata_map(self.label_metadata)
        self.UNDEFINED_LABEL = "undefined"
        
    def _categorize_label(self, label: int) -> str:
        """
        Categorize a label based on metadata map.
        Returns the label name if it exists in metadata, otherwise returns 'undefined'.
        """
        return self.metadata_map.get(label, self.UNDEFINED_LABEL)

    def analyze_dataset(self, dataset_name: str, start_idx: int = 0, end_idx: Optional[int] = None) -> Dict:
        """
        Analyze a complete dataset and return statistics.
        
        Parameters:
        - dataset_name: Short name of the dataset
        - start_idx: Starting index for analysis
        - end_idx: Ending index for analysis (if None, uses all available point clouds)
        
        Returns:
        - Dictionary containing dataset statistics
        """
        if end_idx is None:
            end_idx = get_max_pointclouds_count(dataset_name) - 1
            
        logging.info(f"Analyzing dataset {dataset_name} from index {start_idx} to {end_idx}")
        
        # Load point clouds and labels
        pointclouds_and_labels = import_multiple_pc_and_labels(
            dataset_name, start_idx, end_idx
        )
        
        # Initialize statistics with defaultdict for label distribution
        dataset_stats = {
            'total_points': 0,
            'total_voxels': 0,
            'label_distribution': defaultdict(int),
            'voxel_statistics': [],
            'point_density': [],
            'label_coverage': defaultdict(int),
            'raw_label_counts': defaultdict(int)  # Keep track of raw label counts for reference
        }
        
        # Process each point cloud
        for idx, (points, labels) in enumerate(pointclouds_and_labels):
            # Create voxel grid for this point cloud
            voxel_grid = VoxelGrid(self.voxel_size)
            voxel_grid.add_points(points, labels)
            
            # Get statistics for this point cloud
            stats = voxel_grid.get_voxel_statistics()
            
            # Update dataset statistics
            dataset_stats['total_points'] += len(points)
            dataset_stats['total_voxels'] += stats['total_voxels']
            
            # Update label distribution using categorized labels
            for label, count in stats['label_distribution'].items():
                # Store raw label counts
                dataset_stats['raw_label_counts'][label] += count
                # Update categorized distribution
                category = self._categorize_label(label)
                dataset_stats['label_distribution'][category] += count
                
            # Store point cloud specific statistics
            dataset_stats['voxel_statistics'].append({
                'point_cloud_idx': start_idx + idx,
                'num_voxels': stats['total_voxels'],
                'points_per_voxel': stats['points_per_voxel']
            })
            
            # Calculate point density
            dataset_stats['point_density'].append(len(points) / stats['total_voxels'])
            
            # Update label coverage with categorized labels
            dominant_labels = voxel_grid.get_dominant_labels()
            for label in dominant_labels.values():
                category = self._categorize_label(label)
                dataset_stats['label_coverage'][category] += 1
        
        return dataset_stats
    
    def generate_report(self, dataset_stats: Dict) -> pd.DataFrame:
        """
        Generate a detailed report from dataset statistics.
        
        Parameters:
        - dataset_stats: Statistics dictionary from analyze_dataset
        
        Returns:
        - DataFrame containing the analysis report
        """
        report_data = []
        
        # Overall statistics
        report_data.append({
            'metric': 'Total Points',
            'value': dataset_stats['total_points']
        })
        report_data.append({
            'metric': 'Total Voxels',
            'value': dataset_stats['total_voxels']
        })
        report_data.append({
            'metric': 'Average Point Density',
            'value': np.mean(dataset_stats['point_density'])
        })
        
        # Label distribution
        for label, count in dataset_stats['label_distribution'].items():
            report_data.append({
                'metric': f'Label Distribution - {label}',
                'value': count,
                'percentage': (count / dataset_stats['total_points']) * 100
            })
            
        # Label coverage
        for label, count in dataset_stats['label_coverage'].items():
            report_data.append({
                'metric': f'Label Coverage (Voxels) - {label}',
                'value': count,
                'percentage': (count / dataset_stats['total_voxels']) * 100
            })
            
        # Add raw label information in a separate section
        report_data.append({
            'metric': '--- Raw Label Information ---',
            'value': '',
            'percentage': ''
        })
        
        for label, count in dataset_stats['raw_label_counts'].items():
            report_data.append({
                'metric': f'Raw Label {label}',
                'value': count,
                'percentage': (count / dataset_stats['total_points']) * 100
            })
            
        return pd.DataFrame(report_data)

    def get_label_distribution_summary(self, dataset_stats: Dict) -> pd.DataFrame:
        """
        Generate a summary of label distribution with both raw and categorized counts.
        
        Parameters:
        - dataset_stats: Statistics dictionary from analyze_dataset
        
        Returns:
        - DataFrame containing label distribution summary
        """
        summary_data = []
        
        # Add categorized label distribution
        for label, count in dataset_stats['label_distribution'].items():
            summary_data.append({
                'label_type': 'Categorized',
                'label': label,
                'count': count,
                'percentage': (count / dataset_stats['total_points']) * 100
            })
            
        # Add raw label counts
        for label, count in dataset_stats['raw_label_counts'].items():
            category = self._categorize_label(label)
            summary_data.append({
                'label_type': 'Raw',
                'label': f'{label} (Category: {category})',
                'count': count,
                'percentage': (count / dataset_stats['total_points']) * 100
            })
            
        return pd.DataFrame(summary_data)