import numpy as np
from scipy.spatial import ConvexHull, QhullError
from typing import Tuple, Dict
import logging

def fit_plane_least_squares(points: np.ndarray) -> Tuple[np.ndarray, float]:
    # ... (existing code from your fit_plane_least_squares function) ...
    # This remains the same
    if points.shape[0] < 3:
        raise ValueError("At least 3 points are required to fit a plane")

    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]
    
    if normal_vector[2] < 0:
        normal_vector = -normal_vector

    d = -np.dot(normal_vector, centroid)
    plane_coefficients = np.append(normal_vector, d)

    distances = np.abs(np.dot(points, normal_vector) + d) / np.linalg.norm(normal_vector)
    MSE = np.mean(distances ** 2)
    RMSE = np.sqrt(MSE)

    return plane_coefficients, RMSE

def calculate_vertical_skewness(points: np.ndarray) -> float:
    """
    Calculate the vertical skewness of the point cloud.
    
    Parameters:
    - points: Nx3 numpy array of points in 3D space.
    
    Returns:
    - Vertical skewness value, indicating distribution asymmetry in the z-dimension.
    """
    z_values = points[:, 2]
    mean_z = np.mean(z_values)
    std_dev_z = np.std(z_values)
    skewness = np.mean(((z_values - mean_z) / std_dev_z) ** 3) if std_dev_z > 0 else 0
    return skewness

def calculate_convex_hull_volume(points: np.ndarray) -> float:
        """
        Calculate the volume of the convex hull of the point cloud.
        
        Parameters:
        - points: Nx3 numpy array of points in 3D space.
        
        Returns:
        - Convex hull volume (float) or 0.0 if the points are not full-dimensional.
        """
        if points.shape[0] < 4:  # Need at least 4 points for a 3D hull
            return 0.0
        
        # Check if points are full-dimensional
        if np.linalg.matrix_rank(points - points.mean(axis=0)) < 3:
            # Points are not full-dimensional, i.e., they lie on a plane or a line
            return 0.0

        try:
            hull = ConvexHull(points)
            hull_volume = hull.volume
            if hull_volume < 0:
                logging.warning(f"Convex hull volume is negative: {hull_volume}")
                return hull_volume
            return hull_volume
        except QhullError as e:
            logging.warning(f"Convex hull calculation failed: {e}")
            return 0.0

def calculate_density(points: np.ndarray, voxel_volume: float) -> float:
    """
    Calculate the density of points within a voxel.
    
    Parameters:
    - points: Nx3 numpy array of points in 3D space.
    - voxel_volume: Volume of the voxel.
    
    Returns:
    - Density of points within the voxel.
    """
    if voxel_volume <= 0:
        raise ValueError("Voxel volume must be positive")
    return points.shape[0] / voxel_volume

def calculate_elevation_range(points: np.ndarray) -> float:
    """
    Calculate the elevation range of points within a voxel.
    
    Parameters:
    - points: Nx3 numpy array of points in 3D space.
    
    Returns:
    - Elevation range (max z - min z).
    """
    return np.max(points[:, 2]) - np.min(points[:, 2])
'''
def extract_features(points: np.ndarray, voxel_volume: float) -> Dict[str, float]:
    """
    Extract a variety of features from a point cloud within a voxel.
    
    Parameters:
    - points: Nx3 numpy array of points in 3D space.
    - voxel_volume: Volume of the voxel (to calculate density).
    
    Returns:
    - Dictionary of features.
    """
    features = {}
    
    # Plane fitting
    if points.shape[0] >= 3:
        plane_coefficients, rmse = fit_plane_least_squares(points)
        features['plane_rmse'] = rmse
        features['plane_coefficients'] = plane_coefficients
    else:
        features['plane_rmse'] = None
        features['plane_coefficients'] = None

    # Vertical skewness
    features['vertical_skewness'] = calculate_vertical_skewness(points)
    
    # Convex hull volume
    features['convex_hull_volume'] = calculate_convex_hull_volume(points)
    
    # Density
    features['density'] = calculate_density(points, voxel_volume)
    
    # Elevation range
    features['elevation_range'] = calculate_elevation_range(points)
    
    return features
'''