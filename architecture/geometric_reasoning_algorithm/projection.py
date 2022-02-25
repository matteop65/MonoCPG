import numpy as np
import statistics

def forward_projection(P,X):
    """
        Will return the 2D (u,v) pixel coordinates of the 3D point X in (x,y,z) meters.
        P is the camera matrix. In our case it's K * M * [R|t]
    """
    X = np.append(X,1).reshape(4,1)
    x = P @ X
    depth = x[2,0]
    uv = x[:2,0] / depth
    return uv, depth

def back_projection(P, x, depth):
    """
        Will return a 3D location (x,y,z) in meters given a 2D point (u,v) in pixels 
        and the depth between 3D point and camera
    """
    # make P and x homogenous
    x = np.concatenate( (x, np.array([1])), axis=0)
    P = np.concatenate([P, np.array((0, 0, 0, 1)).reshape(1,-1)], axis=0)
    Pinv = np.linalg.inv(P)
    xprime = np.array((depth*x[0], depth*x[1], depth*1, 1)).reshape(-1,1)
    X3D = Pinv @ xprime
    return X3D
