# Task
* Input: point cloud of indoor building. 
* Five objects: wall, window, door, ceiling, floor
* Output: 3D bounding boxes of objects

# SYNBIM
We have constructed the first large-scale as-built BIM dataset.
* 5239 for training + 1311 for test
* Average foot area of each building is 471.936 m^2. Total area reaches 3.093 km^2.

# Detection Examles    
* mAP: 77%
* mIoU: 75%

|Synthetic mesh  | Point Cloud   |
| ------------- | ------------- |
| mesh | ![Pcl1](./detect_res/1/pcl1.png) |
| Ground truth | Detection |
| ![Gt1](./detect_res/1/gt1.png) | ![Det1](./detect_res/1/det1.png) | 

# Scene 1
|Synthetic mesh  | Point Cloud   |
| ------------- | ------------- |
| mesh | ![Pcl1](./detect_res/1/pcl1.png) |
| Ground truth | Detection |
| ![Gt1](./detect_res/1/gt1.png) | ![Det1](./detect_res/1/det1.png) | 

# Scene 2
|Synthetic mesh  | Point Cloud   |
| ------------- | ------------- |
| mesh | ![Pcl2](./detect_res/2/pcl2.png)  |
| Ground truth | Detection |
|![Gt2](./detect_res/2/gt2.png) | ![Det2](./detect_res/2/det2.png)   |

# Scene 3
|Synthetic mesh  | Point Cloud   |
| ------------- | ------------- |
| mesh |![Pcl3](./detect_res/3/pcl3.png) |
| Ground truth | Detection |
| ![Gt3](./detect_res/3/gt3.png)  | ![Det3](./detect_res/3/det3.png) |

# Scene 4
|Synthetic mesh  | Point Cloud   |
| ------------- | ------------- |
| mesh | ![Pcl4](./detect_res/4/pcl4.png) |
| Ground truth | Detection |
|![Gt4](./detect_res/4/gt4.png)  | ![Det4](./detect_res/4/det4.png)  |

