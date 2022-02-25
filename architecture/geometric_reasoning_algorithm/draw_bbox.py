"""
    This script is used to draw bbox onto said image
"""
import enum
import cv2
import numpy as np
from architecture.geometric_reasoning_algorithm.projection import forward_projection
from architecture.geometric_reasoning_algorithm.geometry import def_ground_plane, direction, find_intersection_2_lines, intersection_plane_and_vector
from numpy import arctan
from math import cos, sin, radians




def anchor_points_4(annotated_image_path, original_image_path, P, vertices, dimensions):
    """
        Draws predicted 3D BBox using anchor points
    """
    img = cv2.imread(original_image_path, 1)

    a1 = vertices["a1"]
    a2 = vertices["a2"]
    a3 = vertices["a3"]
    a4 = vertices["a4"]

    # find length and width directions. 
    dlength = np.subtract(a1, a2)
    dwidth = np.subtract(a2, a3)

    # naming the vertices so they have same format as in CARLA. 
    # from bottom back left, top back left, bottom back right, top back right, front bottom left, front top left...
    v1 = a2
    v2 = a4
    v3 = a3
    v4 = np.add(a3, [0,0,dimensions[2]])
    v5 = a1
    v6 = np.add(a1, [0,0,dimensions[2]])
    int = find_intersection_2_lines( [a1, dwidth], [a3, dlength])
    v7 = a1 + int[0]*dwidth
    v8 = np.add(v7, [0,0,dimensions[2]])

    v = [v1, v2, v3, v4, v5, v6, v7, v8]
    # draw vertices
    img = cv2.imread(original_image_path, 1)
    img, _ = draw_vertices(img, P, vertices, [0,0,0])
    img,v2d = draw_vertices(img, P, v, [0,0,255])
    img = draw_lines(img, v2d, [144,238,144])
    # while True:
    #     cv2.imshow("2D forward projection", img)
    #     key = cv2.waitKey(20) & 0xFF
    #     if key == ord('q'):
    #         break

    cv2.imwrite(annotated_image_path, img)


def anchor_points_5(annotated_image_path, original_image_path, P, vertices, dimensions):
    """
        Draws predicted 3D BBox using anchor points
    """
    img = cv2.imread(original_image_path, 1)

    a1 = vertices["a1"]
    a2 = vertices["a2"]
    a3 = vertices["a3"]
    a4 = vertices["a4"]
    ag5 = vertices["ag5"]

    # find length and width directions. 
    dlength = np.subtract(a1, a2)
    dwidth = np.subtract(a2, a3)
    dheight = np.subtract(a4,a2)

    # naming the vertices so they have same format as in CARLA. 
    # from bottom back left, top back left, bottom back right, top back right, front bottom left, front top left...
    # v1 = [a2[0], a2[1], 0]
    v1 = intersection_plane_and_vector( def_ground_plane(), [a2, [0,0,1]])
    v2 = a4
    v3 = intersection_plane_and_vector( def_ground_plane(), [a3, [0,0,1]])
    v4 = np.add(v3, [0,0,dimensions[2]])
    v5 = [a1[0], a1[1], 0]
    v6 = np.add(v5, [0,0,dimensions[2]])
    int = find_intersection_2_lines( [a1, dwidth], [a3, dlength])
    v7 = v5 + int[0]*dwidth
    v8 = np.add(v7, [0,0,dimensions[2]])

    v = [v1, v2, v3, v4, v5, v6, v7, v8]
    # print(f'vertices: {v}')
    del vertices["ag5"]
    # draw vertices
    img = cv2.imread(original_image_path, 1)
    img, _ = draw_vertices(img, P, vertices, [0,0,0])
    img,v2d = draw_vertices(img, P, v, [0,0,255])
    img = draw_lines(img, v2d, [144,238,144])
    # while True:
    #     cv2.imshow("2D forward projection", img)
    #     key = cv2.waitKey(20) & 0xFF
    #     if key == ord('q'):
    #         break

    cv2.imwrite(annotated_image_path, img)


def anchor_points_4_center(annotated_image_path, original_image_path, P, anchor_pt_vertices, dimensions):
    """
        Draws predicted 3D BBox, using vehicles center point
    """
    img = cv2.imread(original_image_path, 1)

    a1 = anchor_pt_vertices["a1"]
    a2 = anchor_pt_vertices["a2"]
    a3 = anchor_pt_vertices["a3"]

    # find center point of 3D BBox 
    center_point = find_center(a1, a2, a3)

    # find orientation of vehicle
    yaw = arctan( (a1[1] - a2[1]) / (a1[0] - a2[0]))
    pitch = arctan( (a1[2] - a2[2]) / (a1[0] - a2[0]))
    roll = arctan( (a3[2] - a2[2]) / (a3[1] - a2[1]))

    # if pitch or roll not 0, raise warining
    if round(pitch,1) != 0 or round(roll,1) != 0:
        print(f'[WARNING] pitch or roll angles not 0! this model does not yet support functionality for this')
        print(f'pitch: {pitch}')
        print(f'roll: {roll}')

    # we don't need pitch and roll as they are assumed to be zero
    half_len = dimensions[0] / 2 # half length    
    half_wid = dimensions[1] / 2
    half_hei = dimensions[2] / 2

    # determine vertices in local coordinate frame (aligned with x,y, z)
    # treat as 3D problem, and then just add height, as v1, v3, v5, v7 all lie on ground plane. 
    v1_l = [center_point[0] - half_len, center_point[1] - half_wid]
    v2_l = v1_l
    v3_l = [center_point[0]-half_len, center_point[1] + half_wid]
    v4_l = v3_l
    v5_l = [center_point[0]+half_len, center_point[1] - half_wid]
    v6_l = v5_l
    v7_l = [center_point[0] + half_len, center_point[1] + half_wid]    
    v8_l = v7_l

    v2d = [v1_l, v2_l, v3_l, v4_l, v5_l, v6_l, v7_l, v8_l]

    # rotated all the values about the center of the 3d bbox
    v2d_r = []
    for idx, num in enumerate(v2d):
        v2d_r.append(rotate_point(num, yaw, center_point[:2]))
    
    # convert points to 3d
    v3d = []
    for idx, num in enumerate(v2d):
        # if even index, then leave on ground (note even idx is odd numbered vertice above)
        if (idx % 2) == 0:
            v3d.append([num[0], num[1], 0])
        else: # if odd index then vertex is above ground
            v3d.append([num[0], num[1], dimensions[2]])

    # draw vertices
    img = cv2.imread(original_image_path, 1)
    img = draw_vertices(img, P, anchor_pt_vertices, [0,0,0])
    img = draw_vertices(img, P, v3d, [0,0,255])

    while True:
        cv2.imshow("2D forward projection", img)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break

    cv2.imwrite(annotated_image_path, img)


def draw_lines(img, vertices, color):
    """
        Draws lines between the 2D vertices on the image
    """

    for idx, num in enumerate(vertices):
        vertices[idx] = [int(j) for j in num]

    cv2.line(img, (vertices[0][0], vertices[0][1]), (vertices[1][0], vertices[1][1]), color, 2)
    cv2.line(img, (vertices[0][0], vertices[0][1]), (vertices[2][0], vertices[2][1]), color, 2)
    cv2.line(img, (vertices[0][0], vertices[0][1]), (vertices[4][0], vertices[4][1]), color, 2)
    cv2.line(img, (vertices[3][0], vertices[3][1]), (vertices[1][0], vertices[1][1]), color, 2)
    cv2.line(img, (vertices[3][0], vertices[3][1]), (vertices[2][0], vertices[2][1]), color, 2)
    cv2.line(img, (vertices[3][0], vertices[3][1]), (vertices[7][0], vertices[7][1]), color, 2)
    cv2.line(img, (vertices[5][0], vertices[5][1]), (vertices[1][0], vertices[1][1]), color, 2)
    cv2.line(img, (vertices[5][0], vertices[5][1]), (vertices[4][0], vertices[4][1]), color, 2)
    cv2.line(img, (vertices[5][0], vertices[5][1]), (vertices[7][0], vertices[7][1]), color, 2)
    cv2.line(img, (vertices[6][0], vertices[6][1]), (vertices[2][0], vertices[2][1]), color, 2)
    cv2.line(img, (vertices[6][0], vertices[6][1]), (vertices[4][0], vertices[4][1]), color, 2)
    cv2.line(img, (vertices[6][0], vertices[6][1]), (vertices[7][0], vertices[7][1]), color, 2)
    # cv2.line(img, (vertices[0][0], vertices[0][1]), (vertices[1][0], vertices[1][1]), color, 2)

    # for idx, num in enumerate(vertices):
    #     if idx <= 6:
    #         cv2.line(img, (int(vertices[idx][0]), int(vertices[idx][1])), (int(vertices[idx+1][0]), int(vertices[idx+1][1])), color, 2)
    #     else:
    #         break
    
    return img

def draw_vertices(img, P, vertices, color):
    """
        Draws the predicted anchor points
    """
    v2d = [] # 2D vertices
    for idx, num in enumerate(vertices):
        if isinstance(num, str):
            x, depth = forward_projection(P, vertices[num])
        else:
            x, depth = forward_projection(P, num)
        # draw circle
        cv2.circle(img, (int(x[0]), int(x[1])), 4, color, -1)

        v2d.append(x)

    return img, v2d


def find_center(*args):
    """
        Finds center  given a set of points
    """
    if len(args) == 3:
        a1 = args[0]
        a2 = args[1]
        a3 = args[2]

        # center defined by ( (x1 + x3) / 2, (y1 + y3) / 2 ) as no height element
        center_x = (a1[0] + a3[0]) / 2
        center_y = (a1[1] + a3[1]) / 2
        center_z = 0

        return [center_x, center_y, center_z]


def rotate_point(point, angle, center_point):
    """Rotates a point around center_point(origin by default)
    Angle is in degrees.
    Rotation is counter-clockwise
    """
    angle_rad = radians(angle % 360)
    # Shift the point so that center_point becomes the origin
    new_point = (point[0] - center_point[0], point[1] - center_point[1])
    new_point = (new_point[0] * cos(angle_rad) - new_point[1] * sin(angle_rad),
                 new_point[0] * sin(angle_rad) + new_point[1] * cos(angle_rad))
    # Reverse the shifting we have done
    new_point = (new_point[0] + center_point[0], new_point[1] + center_point[1])
    return new_point