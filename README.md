# LiDAR Point Cloud Classification Using Lightweight ML Algorithms on the GOOSE Dataset

This project was developed by Mathis Doutre, Antoine Revel, and Yehonatan Mileguir. To use the scripts in this repository, please ensure the GOOSE goose_3d dataset is downloaded. The dataset should be located at `data/goose_3d_val/goose_label_mapping.csv` (only the validation file is used). You can modify this path in `import_helper.py`.  

For more information, visit [GOOSE Dataset](https://goose-dataset.de/).

## Installation

Install dependencies with:

```bash
  pip install -r requirements.txt
```

## Scripts

- **`Voxel3d.py`**: Helper functions used by other scripts in the repository which extract the point cloud, load the labels, and voxelize the point cloud.

- **`plane_Fitting.py`**: Script that holds the helper functions needed for plane fitting of the voxels. Used by other scripts for feature engineering. 

- **`rmseComparison.py`**: Produces violin plots that compare the majority, minority, and total RMSE for different voxels. 

- **`t_test.py`**: Runs an independent t-test on two target labels' RMSE, randomly chosen from voxels across point clouds. Can be run for multiple iterations. 

- **`VoxelizationAnalysis.py`**: Produces a heatmap showing the number of usable voxels based on minimum point count and majority label thresholds.

- **`LBP.py`**: Helper function to create the LBP feature, aiding the model in distinguishing between low grass and high grass.

- **`ClassificationTest.py`**: Processes the data for multiple images and datasets, and runs a Random Forest classifier. Can be used to test the model on specific images and datasets. 

- **`modelComparison.py`**: Unfinished script designed for testing the inference time of 3D DL segmentation models and this lightweight model.

### Full Dataset Analysis

This section contains the pipeline for working with the entire dataset, culminating in the final model presented.  

- **`import_helper.py`**: Standardizes the data import process.  
- **`voxel_utils.py`**: Handles point cloud operations, analysis, and statistics.  
- **`VoxelGrid.py`**: Introduces a class to streamline the voxel creation process.  
- **Visualization Scripts**: Includes tools like `PointCloudVisualiser.py` and `generate_label_plots.py` for data visualization.  

The pipeline is divided into two main steps:  
1. **Data Preprocessing**: Converts raw point cloud data into voxel data and computes statistics. The processed data is saved as a pickle file.  
   - Performed by `process_voxel_data.py`, which generates data for training and saves it as `processed_voxel_data_reduced.local`.  
2. **Feature and Model Analysis**:  
   - Feature analysis is conducted using `feature_importance_all_features.py`.  
   - Model analysis uses `feature_importance_reduced.py`, leveraging the preprocessed pickle file.  
