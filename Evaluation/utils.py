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