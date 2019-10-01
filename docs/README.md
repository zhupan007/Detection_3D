This is the first 3D object detection network optimized for building primitives. 
Potential applications: as-built BIM, indoor map reconstruction.

# Task
* Input: point cloud of indoor building. 
* Five objects: wall, window, door, ceiling, floor
* Output: 3D bounding boxes of objectis

# Data
We have constructed the first large-scale synthetic as-built BIM dataset (SYNBIM).
* 6550 buildings: 5239 for training + 1311 for test
* Average foot area of each building is 471.936 m^2. Total area reaches 3.093 km^2.

# Assumption
* Shortest wall instance: Long wall pieces are croped by intersected ones to generate short instances.

# Detection Performance
* time per building: 4.75 s

| | Wall | Window | Door | Floor| Ceiling | Classes Mean |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| AP(%) | 74.75 | 65.15 | 86.80 | 77.60| 77.46 | **76.35** |
| AIoU(%) | 78.33 | 65.88 | 86.22 | 78.84| 78.21 | **77.50** |



# Scene 1
Each intance is colorized by a random color, except blue denotes incorrect detection or missed ground truth.

|Synthetic mesh  | Point Cloud   |
| :-------------: | :-------------: |
| ![Mesh1](./detect_res/1/mesh1.png)  | ![Pcl1](./detect_res/1/pcl1.png) |
| **Ground truth** | **Detection** |
| ![Gt1](./detect_res/1/gt1.png) | ![Det1](./detect_res/1/det1.png) | 

# Scene 2
  
|Synthetic mesh  | Point Cloud   |
| :-------------: | :-------------: |
| ![Mesh2](./detect_res/2/mesh2.png)  | ![Pcl2](./detect_res/2/pcl2.png)  |
| **Ground truth** | **Detection** |
|![Gt2](./detect_res/2/gt2.png) | ![Det2](./detect_res/2/det2.png)   |

# Scene 3
  
|Synthetic mesh  | Point Cloud   |
| :-------------: | :-------------: |
|  ![Mesh3](./detect_res/3/mesh3.png)   |![Pcl3](./detect_res/3/pcl3.png) |
| **Ground truth** | **Detection** |
| ![Gt3](./detect_res/3/gt3.png)  | ![Det3](./detect_res/3/det3.png) |

# Scene 4
  
|Synthetic mesh  | Point Cloud   |
| :-------------: | :-------------: |
|  ![Mesh4](./detect_res/4/mesh4.png)  | ![Pcl4](./detect_res/4/pcl4.png) |
| **Ground truth** | **Detection** |
|![Gt4](./detect_res/4/gt4.png)  | ![Det4](./detect_res/4/det4.png)  |

