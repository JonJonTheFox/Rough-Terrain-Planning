
# LiDAR Point Cloud Classificacation Using Light Weight ML Algorithms on GOOSE dataset

This project was built by Mathis Gaultier Doutre, Antoine Revel, and
Yehonatan Mileguir. Please have the GOOSE goose_3d dataset downloaded to use the scripts in this repository. 







## Installation

Install dependencies with 

```bash
  pip install -r requirements.txt
```
    
## Scripts

Voxel3d.py: Helper functions used by other scripts in the repository which extract the point cloud, load the labels, and voxelize the point cloud.

plane_Fitting.py: Script that holds the helper functions needed for plane fititng for the voxels. Used by other scripts for feature engineering. 

rmseComparison.py: Produces violen plots that compare the majority, minority, and total rmse for different voxels. 

t_test.py: 