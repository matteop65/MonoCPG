import numpy as np

def camera_info(map, camera):
    """
        Returns:
            - intrinsic matrix, K, 
            - extrinsic matrix R_t, 
            - camera location
        
        depending on map and camera
    """
    
    if map == "autoplex":
        if camera == 8:
            K = [ [ 1.14408345e3, 0, 960],
                  [ 0, 1.14408345e3, 540],
                  [ 0, 0, 1]]

            R_t = [ [ 0.4330196, 0.74999988, -0.5, -117.1752771],
                    [ -0.86602527, 0.5000003, -0, -28.44947815],
                    [ 0.25000015, 0.43301263, 0.86602539, -79.19818115]]
            location = np.array([45.90, 136.40, 10]).reshape(3,1)
            
    elif map == "town03":
        raise(Exception("need to add functionalise for map town03"))
        # location = np.array([72.20, -213.20, 8.10]).reshape(3,1)
        # if img_num == 1 or img_num == 2:
        #     K2 = [ [ 400, 0, 400],
        #         [ 0, 400, 300],
        #         [ 0, 0, 1]]
        # elif img_num == 3:
        #     K3 = [ [960, 0, 960],
        #         [0,  960, 540],
        #         [0,   0,  1]]
    return K, R_t, location