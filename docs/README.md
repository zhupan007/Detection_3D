# Task
* Input: point cloud of indoor building. 
* Five objects: wall, window, door, ceiling, floor
* Output: 3D bounding boxes of objects

# SYNBIM
We have constructed the first large-scale as-built BIM dataset.
* 5239 for training + 1311 for test
* Average foot area of each building is 471.936 m^2. Total area reaches 3.093 km^2.

# Detection Examles    
mAP: 81%
mIoU: 86%

| First Header  | Second Header |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |

# Scene 1
|||
:-------------------------:|:-------------------------:
 | ![Pcl1](./detect_res/1/pcl1.png) Point cloud  
 ![Gt1](./detect_res/1/gt1.png) Ground truth  | ![Det1](./detect_res/1/det1.png) Detection   

# Scene 2
| | ![Pcl2](./detect_res/2/pcl2.png) Point cloud   |
|---|---|
|![Gt2](./detect_res/2/gt2.png) Ground truth  | ![Det2](./detect_res/2/det2.png) Detection    |

# Scene 3
| |![Pcl3](./detect_res/3/pcl3.png) Point cloud  |
|---|---|
| ![Gt3](./detect_res/3/gt3.png) Ground truth  | ![Det3](./detect_res/3/det3.png) Detection   |

# Scene 4
| | ![Pcl4](./detect_res/4/pcl4.png) Point cloud  |
|---|---|
|![Gt4](./detect_res/4/gt4.png) Ground truth  | ![Det4](./detect_res/4/det4.png) Detection    |

