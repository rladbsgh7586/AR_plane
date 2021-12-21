import numpy as np


def step_printer(step):
    print("------------step------------")
    print(step)
    print("----------------------------")


def parsing_transformation_matrix(array_string):
    array_string = array_string.replace("[","")
    array_string = array_string.replace("]", "")
    array_string = array_string.replace(" ", "")
    array = np.array(array_string.split(","))
    transformation_matrix = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            transformation_matrix[i][j] = float(array[i*4+j])
    return transformation_matrix


def get_depth_value(x, y, parameter, camera):
    u = (x - camera[2]) / camera[0]
    v = - (y - camera[3]) / camera[1]

    # parameter [normal[0], normal[1], normal[2], offset]
    t = -(parameter[3] - (u * parameter[0]) - (v * parameter[1])) / parameter[2]
    return t


def get_3d_point(xy, plane_normal, offset):
    t = offset / (plane_normal[0] * xy[0] + plane_normal[1] * xy[1] + plane_normal[2])
    return np.array([xy[0] * t, xy[1] * t, t])


def convert_to_normal_coordinate(x, y, camera):
    u = (x - camera[2]) / camera[0]
    v = - (y - camera[3]) / camera[1]

    return u, v


def calc_plane_equation(vtp_matrix):
    A = np.dot([0, 0, 0, 1], vtp_matrix)[:3]
    B = np.dot([0.1, 0, 0, 1], vtp_matrix)[:3]
    C = np.dot([0, 0, 0.1, 1], vtp_matrix)[:3]

    v1 = C - A
    v2 = B - A

    cp = np.cross(v1, v2)
    a, b, c = cp

    plane_normal = np.array([a, b, c])
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    return np.array([-plane_normal[0], -plane_normal[1], plane_normal[2], -np.dot(plane_normal, A)])
