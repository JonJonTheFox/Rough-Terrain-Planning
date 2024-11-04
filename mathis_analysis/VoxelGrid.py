import numpy as np
import pandas as pd
from collections import defaultdict
import logging
from scipy.spatial import ConvexHull
from typing import Tuple, List, Dict, Optional
import voxel_utils as vu
class VoxelGrid:
    def __init__(self, voxel_size: float = 1.0):
        """
        Initialize a voxel grid with given voxel size.
        
        Parameters:
        - voxel_size: Size of voxel cube edges in same units as point cloud
        """
        self.voxel_size = voxel_size
        self.voxel_dict = defaultdict(list)
        self.voxel_labels = defaultdict(list)
        
    def _point_to_voxel_key(self, point: np.ndarray) -> Tuple[int, int, int]:
        """Convert a point to its voxel grid indices."""
        voxel_x = int(np.floor(point[0] / self.voxel_size))
        voxel_y = int(np.floor(point[1] / self.voxel_size))
        return (voxel_x, voxel_y)
    
    def add_points(self, points: np.ndarray, labels: np.ndarray):
        """
        Add points and their labels to the voxel grid.
        
        Parameters:
        - points: Nx4 array of points (x, y, z, intensity)
        - labels: N array of label indices
        """
        for point_idx in range(len(points)):
            point = points[point_idx]
            label = labels[point_idx]
            voxel_key = self._point_to_voxel_key(point)
            self.voxel_dict[voxel_key].append(point)
            self.voxel_labels[voxel_key].append(label)
    
    def filter_voxels(self, number_of_points: int , minimum_majority_proportion: float, whitelist = None):
        dominant_labels = self.get_dominant_labels_with_proportion()
        for voxel_key in list(self.voxel_dict.keys()): # Iterating over a copy
            points = self.voxel_dict[voxel_key]
            if len(points) < number_of_points:
                self.remove_voxel(voxel_key)
                continue
            else :
                label = dominant_labels[voxel_key]['label']
                proportion = dominant_labels[voxel_key]['proportion']
                
                if whitelist:
                    if label not in whitelist : 
                        self.remove_voxel(voxel_key)
                        continue
                
                if proportion < minimum_majority_proportion:
                    self.remove_voxel(voxel_key)
                    continue
                        
                    
    def remove_voxel(self, voxel_key) -> None:
        """Remove the specified voxel from the voxel dictionary."""
        if voxel_key in self.voxel_dict:
            logging.debug(f"Removing Voxel: {voxel_key}")
            del self.voxel_dict[voxel_key]     
                
                

    def get_voxel_statistics(self) -> Dict:
        """Calculate various statistics about the voxels."""
        stats = {
            'total_voxels': len(self.voxel_dict),
            'points_per_voxel': {},
            'label_distribution': defaultdict(int),
            'voxel_label_counts': defaultdict(lambda: defaultdict(int)),
            'voxel_plane_rmse': {},
            'voxel_plane_coefficients': {},
            'vertical_skewness': {},
            'convex_hull_volume': {},
            'density': {},
            'elevation_range': {},
            'avg_intensity': {}
        }
        
        for voxel_key, points in self.voxel_dict.items():
            point_count = len(points)
            stats['points_per_voxel'][point_count] = stats['points_per_voxel'].get(point_count, 0) + 1
            
            points_3d = np.array(points)[:, :3]
            try:
                if len(points) < 3:
                    stats['voxel_plane_rmse'][voxel_key] = None
                    stats['voxel_plane_coefficients'][voxel_key] = None
                else:
                    plane_coefficients, rmse = vu.fit_plane_least_squares(points_3d)
                    stats['voxel_plane_rmse'][voxel_key] = rmse
                    stats['voxel_plane_coefficients'][voxel_key] = plane_coefficients
                
            except ValueError as e:
                logging.warning(f"Voxel {voxel_key}: {e}")
                stats['voxel_plane_rmse'][voxel_key] = None
                stats['voxel_plane_coefficients'][voxel_key] = None
            
            
            stats['vertical_skewness'][voxel_key] = vu.calculate_vertical_skewness(points_3d)
            hull_volume = vu.calculate_convex_hull_volume(points_3d)
            stats['convex_hull_volume'][voxel_key] = hull_volume
        
            if hull_volume == 0:
                stats['density'][voxel_key] = None
            else:     
                stats['density'][voxel_key] = vu.calculate_density(points_3d, hull_volume)
            stats['elevation_range'][voxel_key] = vu.calculate_elevation_range(points_3d)
            
            stats['avg_intensity'][voxel_key] = np.mean(np.array(points)[:, 3])
            
            
            # Label statistics
            labels = self.voxel_labels[voxel_key]
            for label in labels:
                stats['label_distribution'][label] += 1
                stats['voxel_label_counts'][voxel_key][label] += 1
        
        return stats

    def get_voxel_centers(self) -> np.ndarray:
        """Return the center coordinates of all voxels."""
        centers = []
        for voxel_key in self.voxel_dict.keys():
            x, y = voxel_key
            center = np.array([
                (x + 0.5) * self.voxel_size,
                (y + 0.5) * self.voxel_size
            ])
            centers.append(center)
        return np.array(centers)
    
    def get_dominant_labels_with_proportion(self) -> Dict:
        """Return the dominant label and its proportion for each voxel."""
        dominant_labels = {}
        for voxel_key, labels in self.voxel_labels.items():
            unique_labels, counts = np.unique(labels, return_counts=True)
            dominant_label_index = np.argmax(counts)
            dominant_label = unique_labels[dominant_label_index]
            total_count = np.sum(counts)
            dominant_proportion = counts[dominant_label_index] / total_count
            dominant_labels[voxel_key] = {
                'label': dominant_label,
                'proportion': dominant_proportion
            }
        
        return dominant_labels
