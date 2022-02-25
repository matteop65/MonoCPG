import numpy
import matplotlib.pyplot as plt
import numpy as np
import math


def direction(X, C):
    """
        Returns the direction between 2 vectors
    """
    X = X[:-1]
    d = np.subtract(C, X)
    return d


def dot_product(a, b):
    """
        Calculated the dot product between a and b
        if a and b aren't vectors, performs normal multiplication
    """
    try:
        if len(a) != len(b):
            raise Exception(f'Dot product between: {a} and {b} not possible')
        else:
            result = 0
            for element in range(len(a)):
                result = result + a[element]*b[element]
    except:
        result = a * b
    return result


def draw_vector_w_point_through_plane(origin, point_on_plane):
    vector = [0,0,0]
    return vector


def find_point_at_distance_from_initial_point(initial_point, direction, distance):
    """
        This function will find a point given a distance from a starting point. 
        In our application, the starting point will be the camera, and the distance will represent the focal distance, 
            from which to project the image plane.
        In our application, the camera always faces the ground, so the distance is effectively negative by mathematical convention. 
        Both must be 3 dimensional arrays 
    """
    a = np.array(initial_point)
    d = np.array(direction)
    d_unit = d*1./( math.sqrt(d[0]^2+d[1]^2+d[2]^2))
    b = a - distance*d_unit
    return b


def intersection_plane_and_vector(plane, vector):
    """
        Finds intersection point of plane and vector. 
        plane = [a b c d]
        vector = [ [u1 u2 u3], [d1 d2 d3]] where a is initial point and d is direction
    """
    # plane = plane.tolist()
    # print(f'type: {type(vector)}')
    # vector = vector.tolist()

    a = plane[0]
    b = plane[1]
    c = plane[2]
    d = plane[3]

    u1 = vector[0][0]
    u2 = vector[0][1]
    u3 = vector[0][2]

    d1 = vector[1][0]
    d2 = vector[1][1]
    d3 = vector[1][2]

    lambda_1 = (d - a*u1 - b*u2 - c*u3) / (a*d1 + b*d2 + c*d3)

    intersection = []
    for i in range(len(vector[0])):
        intersection.append(vector[0][i] + lambda_1*vector[1][i])
    return intersection


def draw_vector_from_2_points(point1, point2):
    """
        Will find vector that passes through 2 points. 
        Point1 is starting point. 
        return format: [ [direction x,y,z], [starting point, x,y,z]
    """
    direction = np.subtract(np.array(point1), np.array(point2))
    vector = [point1, np.array(direction).tolist()]
    return vector


def find_plane_from_2_points(point1, point2):
    """
        Will find a plane that passes through 2 points. 
        return format: [ a b c d]
        Only deals with vertical planes atm. 
        Need to add capability to draw vectors parallel to axis as well. 
    """
    direction = np.subtract(np.array(point1), np.array(point2))
    normal_direction = [direction[1], -direction[0], direction[2]] # vector for 2D shape essentially. 

    # add method to verify that the normal is actually the normal through scalar product. 
    return find_plane_from_normal_and_point(normal_direction, point1)


def find_plane_from_normal_and_point(normal, point):
    """
        Will return plane given the normal and a point. 
        return format: [a b c d]
    """
    image_plane = []
    for element in normal: image_plane.append(element)    
    image_plane.append(dot_product(normal, point))
    return image_plane


def find_plane_from_2_lines(line1, line2, point):
    """
        Finds plane from 2 lines and points
    """
    normal = np.cross(line1[1], line2[1])
    return find_plane_from_normal_and_point(normal, point)


def find_intersection_2_lines(line1, line2):
    """
        Finds intersection of 2 lines: 
            r1 = [a1, a2, a3] + lambda[d1, d2, d3]
            r2 = [b1, b2, b3] + gamma[h1, h2, h3]
    """
    a1 = line1[0][0]
    a2 = line1[0][1]
    a3 = line1[0][2]
    d1 = line1[1][0]
    d2 = line1[1][1]
    d3 = line1[1][2]

    b1 = line2[0][0]
    b2 = line2[0][1]
    b3 = line2[0][1]
    h1 = line2[1][0]
    h2 = line2[1][1]
    h3 = line2[1][1]

    A = [ [d1, -h1], [d2, -h2]]
    Y = [ (b1 - a1), (b2 - a2)]
    res = np.linalg.inv(A).dot(Y)

    return res


def def_ground_plane():
    """
        Will return 3D Ground Plane. 
        Due to assumptions of model, there will be no height variations on the x and y axes
            hence x and y slopes = 0, and there will only be a z =const. 
        Due to way
    """
    # z = camera_xyz[2][0]
    pi_g = np.array([0, 0, 1, 0])
    return pi_g
