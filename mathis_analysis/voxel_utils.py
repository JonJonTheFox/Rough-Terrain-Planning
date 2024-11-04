import numpy as np
from scipy.spatial import ConvexHull, QhullError
from typing import Tuple, Dict
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.stats import skew

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

def calculate_avg_z_value(points: np.ndarray) -> float:
    """
    Calculate the average z value of points within a voxel.
    
    Parameters:
    - points: Nx3 numpy array of points in 3D space.
    
    Returns:
    - Average z value.
    """
    return np.mean(points[:, 2])


def apply_dynamic_lbp(points: np.ndarray, k_neighbors: int = 6) -> int:
    """
    Applies a height-based Label Binary Pattern (LBP) to a single voxel's points.

    Parameters:
    - points_in_voxel: Nx3 array of points in 3D space within a single voxel.
    - k_neighbors: Number of neighbors to consider for LBP calculation. Default is 6.

    Returns:
    - LBP pattern as an integer for the points in this voxel.
    """
    if len(points) < 2:
        return 0  # Not enough points to calculate neighbors

    # Initialize the nearest neighbors model
    knn = NearestNeighbors(n_neighbors=min(k_neighbors, len(points)))
    knn.fit(points)

    # Calculate the centroid of the voxel points
    voxel_centroid = np.mean(points, axis=0).reshape(1, -1)

    # Find nearest neighbors to the centroid
    distances, indices = knn.kneighbors(voxel_centroid)

    # Calculate the LBP pattern based on height (z-coordinate) differences
    lbp_value = 0
    for i, neighbor_idx in enumerate(indices[0]):
        neighbor_point = points[neighbor_idx]
        if voxel_centroid[0][2] > neighbor_point[2]:  # Check if neighbor is lower in height
            lbp_value |= (1 << i)  # Set the corresponding bit if height difference is positive

    return lbp_value

# Function to perform PCA and compute variance ratios, flatness, and elongation
def compute_pca_metrics(points: np.ndarray) -> Tuple[np.ndarray, float, float]:
    pca = PCA(n_components=3)
    pca.fit(points[:, :3])  # Use only XYZ coordinates
    variance_ratios = pca.explained_variance_ratio_
    flatness = variance_ratios[1] / variance_ratios[2] if variance_ratios[2] > 0 else 0
    elongation = variance_ratios[0] / variance_ratios[1] if variance_ratios[1] > 0 else 0
    return variance_ratios, flatness, elongation

# Function to compute height variability and vertical skewness
def compute_height_variability(points: np.ndarray) -> Tuple[float, float]:
    z_values = points[:, 2]
    height_variability = np.std(z_values)
    vertical_skewness = skew(z_values)
    return height_variability, vertical_skewness


# Function to compute curvature
def compute_curvature(points: np.ndarray, k: int = 10) -> float:
    if len(points) < k:
        logging.warning("Not enough points to compute curvature. Returning 0.")
        return 0.0
    neighbors = NearestNeighbors(n_neighbors=k).fit(points[:, :3])
    _, indices = neighbors.kneighbors(points[:, :3])
    curvatures = []

    for idx in indices:
        local_points = points[idx, :3]
        pca = PCA(n_components=3, random_state=1)
        pca.fit(local_points)
        curvatures.append(pca.explained_variance_ratio_[2])

    return np.mean(curvatures)

# Function to compute the mean nearest neighbor distance
def compute_mean_nearest_neighbor_distance(points: np.ndarray, k: int = 1) -> float:
    if len(points) <= k:
        logging.warning("Not enough points to compute mean nearest neighbor distance. Returning 0.")
        return 0.0
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(points[:, :3])
    distances, _ = neighbors.kneighbors(points[:, :3])
    return np.mean(distances[:, 1:])  # Skip the first column (distance to self)

# Function to compute surface roughness
def compute_surface_roughness(points: np.ndarray, k: int = 10) -> float:
    if len(points) < k:
        logging.warning("Not enough points to compute surface roughness. Returning 0.")
        return 0.0
    neighbors = NearestNeighbors(n_neighbors=k).fit(points[:, :3])
    _, indices = neighbors.kneighbors(points[:, :3])
    roughness = []

    for idx in indices:
        local_points = points[idx, :3]
        local_pca = PCA(n_components=1, random_state=1)
        local_pca.fit(local_points)
        roughness.append(local_pca.explained_variance_ratio_[0])

    return np.mean(roughness)