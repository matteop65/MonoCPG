import statistics
import numpy as np
import architecture.geometric_reasoning_algorithm.geometry
from architecture.geometric_reasoning_algorithm.geometry import direction
from architecture.geometric_reasoning_algorithm.projection import back_projection, forward_projection
from architecture.load_img import load_img_place_anchor_pts


def five_keypoints_suggested(all_keypoint_info, camera_location, ground_plane):
    """
        Takes 5 anchor point inputs (in x,y,z) and uses coordinate geometry to solve relevant dimensions. 
        Uses suggested method outlined in progress document
    """
    a_gi = []
    
    # find intersection points between ground plane and each direction vector per anchor point
    for i in range(len(all_keypoint_info)):
        keypoint_info = all_keypoint_info[f'keypoint_{i+1}']
        direction = keypoint_info['direction']

        vector = [camera_location.T[0], direction.T[0].tolist()]
        intersection_pt = architecture.geometric_reasoning_algorithm.geometry.intersection_plane_and_vector(ground_plane, vector)
        a_gi.append(intersection_pt)

        # store 3D anchor point position, used to forward projection 3D BBox onto image. 
        keypoint_info['3D'] = intersection_pt
        all_keypoint_info[f'keypoint_{i+1}'] = keypoint_info

    # create plane pi_1
    d12 = np.array(a_gi[0])- np.array(a_gi[1])
    d12 = d12 / np.linalg.norm(d12) # normalise d12
    normal1 = [-d12[1], d12[0],0]
    pi_1 = architecture.geometric_reasoning_algorithm.geometry.find_plane_from_normal_and_point(normal1, a_gi[4])
    
    # find length
    u1 = all_keypoint_info[f'keypoint_1']
    u1_d = u1['direction']
    v1 = [camera_location.T[0], u1_d.T[0].tolist()]
    a_1 = architecture.geometric_reasoning_algorithm.geometry.intersection_plane_and_vector(pi_1, v1)

    u2 = all_keypoint_info[f'keypoint_2']
    u2_d = u2['direction']
    v2 = [camera_location.T[0], u2_d.T[0].tolist()]
    a_2 = architecture.geometric_reasoning_algorithm.geometry.intersection_plane_and_vector(pi_1, v2)

    length = abs( ( (a_1[0]-a_2[0])**2 + (a_1[1]-a_2[1])**2 ) ** 0.5 )

    # find width
    d23 = np.array(a_gi[1] - np.array(a_gi[2]))
    normal2 = [d12[0], d12[1], 0]
    pi_2 = architecture.geometric_reasoning_algorithm.geometry.find_plane_from_normal_and_point(normal2, a_2)

    u3 = all_keypoint_info[f'keypoint_3']
    u3_d = u3['direction']
    v3 = [camera_location.T[0], u3_d.T[0].tolist()]
    a_3 = architecture.geometric_reasoning_algorithm.geometry.intersection_plane_and_vector(pi_2, v3)

    # width = abs(a_2[1] - a_3[1])
    width = abs( ( (a_2[0]-a_3[0])**2 + (a_2[1]-a_3[1]) **2 ) ** 0.5 )

    # find height
    u4 = all_keypoint_info['keypoint_4']
    u4_d = u4['direction']
    v4 = [camera_location.T[0], u4_d.T[0].tolist()]
    a_4 = architecture.geometric_reasoning_algorithm.geometry.intersection_plane_and_vector(pi_1, v4)
    a_4_test = architecture.geometric_reasoning_algorithm.geometry.intersection_plane_and_vector(pi_2, v4)

    height = a_4[2]

    dimensions = [length, width, height]

    # store the information of the predicted keypoint_vertices, a_1, a_2, a_3, a_4
    keypoint_vertices = {
        "a1":a_1, 
        "a2":a_2,
        "a3":a_3,
        "a4":a_4,
        "ag5":a_gi[4]
    }

    # all_keypoint_info[""]

    all_keypoint_info["pi_1"] = pi_1
    all_keypoint_info["pi_2"] = pi_2

    return dimensions, all_keypoint_info, keypoint_vertices


def four_keypoints(all_keypoint_info, camera_location, ground_plane):
    """
        Takes 4 anchor point inputs (in x,y,z) and uses coordinate geometry to solve for relevant dimensions
        Steps to solve for width and height are the same as with three anchor points. 
    """
    a_gi = [] # intersection point between v_i and ground plane 

    # delete anchor point 4 from dictionary passed to three anchor point solver. 
    keypoint_4 = all_keypoint_info['keypoint_4']
 
    if 'keypoint_4' in all_keypoint_info:
        del all_keypoint_info['keypoint_4']

    # Use 3 anchor point solver for length and width
    dimensions, all_keypoint_info, keypoint_vertices = three_keypoints(all_keypoint_info, camera_location, ground_plane)

    # solve for height, similar to width and length
    direction = keypoint_4['direction']
    keypoint_1 = all_keypoint_info['keypoint_1']
    keypoint_2 = all_keypoint_info['keypoint_2']
    a_1 = keypoint_1['3D']
    a_2 = keypoint_2['3D']
    pi_1 = architecture.geometric_reasoning_algorithm.geometry.find_plane_from_2_points(a_1, a_2)
    v_4 = [camera_location.T[0].tolist(), direction.T[0].tolist()]

    # intersection point is between plane, pi_1 and vector
    a_4 = architecture.geometric_reasoning_algorithm.geometry.intersection_plane_and_vector(pi_1, v_4)

    # store 3D anchor point 4 info
    keypoint_4['3D'] = a_4
    all_keypoint_info['keypoint_4'] = keypoint_4

    # height defined as height difference between anchor point 2 and 4
    height = abs(a_4[2] - a_2[2])

    dimensions = [dimensions[0], dimensions[1], height]

    # store information in predicted keypoint_vertices, a_1, a_2, a_3, a_4
    keypoint_vertices["a4"] = a_4

    all_keypoint_info["pi_1"] = pi_1


    return dimensions, all_keypoint_info, keypoint_vertices


def three_keypoints(all_keypoint_info, camera_location, ground_plane):
    """
        Takes 3 anchor point inputs (in x,y,z) and uses coordinate geometry to solve for relevant dimensions
    """
    a_gi = [] # intersection point between v_i and ground plane 

    for i in range(len(all_keypoint_info)):
        # load relevant information
        keypoint_info = all_keypoint_info[f'keypoint_{i+1}']
        direction = keypoint_info['direction']

        # find intersection between vector and ground plane
        vector = [camera_location.T[0].tolist(), direction.T[0].tolist()]
        intersection_pt = architecture.geometric_reasoning_algorithm.geometry.intersection_plane_and_vector(ground_plane, vector)
        a_gi.append(intersection_pt)

        # store 3D anchor point position, used to forward projection 3D BBox onto image. 
        keypoint_info['3D'] = intersection_pt
        all_keypoint_info[f'keypoint_{i+1}'] = keypoint_info
    
    # determine length and width

    # length is horizontal distance between anchor points 1 and 2
    length = abs( ( (a_gi[0][0]-a_gi[1][0])**2 + (a_gi[0][1]-a_gi[1][1])**2 ) ** 0.5 )

    # distance between anchor points 2 and 3
    width = abs(a_gi[1][1] - a_gi[2][1])
    height = 0 # to be determined as a constant value per vehicle class
    dimensions = [length, width, height]

    keypoint_vertices = {
        "a1":a_gi[0],
        "a2":a_gi[1],
        "a3":a_gi[2],
        "a4":[0,0,0]
    }

    return dimensions, all_keypoint_info, keypoint_vertices
    
    

def solve_main(input):
    """
        Step1: Load image and anchor points (u,v)
    """
    if not input['image_folder']:
        img, anchor_pts = load_img_place_anchor_pts(input['img_path'], input['number_of_keypoints'])
        input['number_of_keypoints'] = len(anchor_pts)
        print(f'anchor pts: {anchor_pts}')
    else:
        anchor_pts = input['anchor_pts']
    point_info = {} # store all the anchor point related information in this dictionary

    """
        Step2: Back-projection from 2D (u,v) to 3D (x,y,z)
            As the depth of the vehicle is not known (we are trying to solve this)
            the 3D ray of the point is found.     
    """
    # define camera matrix P 
    K = input['intrinsics']
    R_t = input['extrinsics']
    M = input['convention'] # used to go between UE4 and CARLA coordinate systems
    P = K @ M @ R_t
    R = np.array([R_t[0][:3], R_t[1][:3], R_t[2][:3]])
    t = np.array([R_t[0][3], R_t[1][3], R_t[2][3]]).reshape(3,1)

    # to get accurate direction, need to compute for a variety of depths
    depth_range = np.linspace(1,100,10000)

    for i in range(len(anchor_pts)):
        x = anchor_pts[i]
        info = {
            'index':i+1,
            'uv':x,
            'direction': [],
            '3D':[],
        }

        d_x = []
        d_y = []
        d_z = []
        for depth in depth_range:
            X = back_projection(P, x, depth)
            # d = direction(X, input['camera_location'])
            d = direction(X, -R.T@t)    # R.T@t is the same as the camera location
            d = d / np.linalg.norm(d) # normalise direction vector
            d_x.append(d[0,0].tolist())
            d_y.append(d[1].tolist()[0])
            d_z.append(d[2].tolist()[0])

        d_x_avg = statistics.mean(d_x)
        d_y_avg = statistics.mean(d_y)
        d_z_avg = statistics.mean(d_z)

        d_avg = np.array([d_x_avg, d_y_avg, d_z_avg]).reshape(3,1)
        info['direction'] = d_avg
        # print(f'info: {info}')
        point_info[f'keypoint_{i+1}'] = info

    """
        Step3: Solve for the dimensions, given the number of anchor points
    """
    dimensions = []
    ground_plane = architecture.geometric_reasoning_algorithm.geometry.def_ground_plane()

    if input['number_of_keypoints'] == 3:
        dimensions, keypoint_info, keypoint_vertices = three_keypoints(point_info,input['camera_location'], ground_plane)
    elif input['number_of_keypoints'] == 4:
        dimensions, keypoint_info, keypoint_vertices = four_keypoints(point_info, input['camera_location'], ground_plane)
    elif input['number_of_keypoints'] == 5:
        dimensions, keypoint_info, keypoint_vertices = five_keypoints_suggested(point_info, input['camera_location'], ground_plane)

    dimensions = np.round(dimensions, 4)
 



    return dimensions, keypoint_info, keypoint_vertices
