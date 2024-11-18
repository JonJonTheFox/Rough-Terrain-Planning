
# LiDAR Point Cloud Classificacation Using Light Weight ML Algorithms on GOOSE dataset

This project was built by Mathis Doutre, Antoine Revel, and
Yehonatan Mileguir. Please have the GOOSE goose_3d dataset downloaded to use the scripts in this repository. 







## Installation

Install dependencies with 

```bash
  pip install -r requirements.txt
```
    
## Scripts

Voxel3d.py: Helper functions used by other scripts in the repository which extract the point cloud, load the labels, and voxelize the point cloud.

plane_Fitting.py: Script that holds the helper functions needed for plane fititng of the voxels. Used by other scripts for feature engineering. 

rmseComparison.py: Produces violon plots that compare the majority, minority, and total rmse for different voxels. 

t_test.py: Runs an independent t_test on two target labels RMSE which are randomly chosen out of voxels from a number of point cloud. Can be run for multiple iterations. 

VoxelizationAnalysis.py: Script which produces a heatmap of what number of voxels are usable based on minimum number of points and majority label threshold.

LBP.py: LBP Helper function to create LBP feature which can help the model differentiate between Low Grass and High Grass

ClassificationTest.py: Script which runs the preprocessing of the data for multiple images for multiple datasets and runs the RandomForrest Classifier. Can be used for testing the model on specific images and datasets. 

modelComparison.py: Unfinished script meant for the testing the inference time of 3D DL semgentation models and our model. 
