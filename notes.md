1. self.conv in RPNHead in rpn_sparse3d.py 
    Originally, kernel is 3, but is set as 1 now. Check if submanifold with kernel_size=3 is required.
2. clip_to_pcl in bounding_box_3d.py not implemented yet
3. remove_small_boxes3d is not enabled in inference_3d.py
4. About boxlist.clip_to_image 
        Originally, it is performed in forward_for_single_feature_map in RPNPostProcessor in inference.py
        This is a force fix of proposals. 
        What about directly clip the anchor size by scene size?
        Currently, do no clip yet.
5. some boxes with yaw !=0 and !=90 are not cropped properly
6. Add dirction loss later
