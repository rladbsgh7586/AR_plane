import numpy as np
import pyrealsense2 as rs
import os

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

    t = parameter[3] / (parameter[0] * u + parameter[1] * v + parameter[2])
    # parameter [normal[0], normal[1], normal[2], offset]
    t = (parameter[3] - (u * parameter[0]) - (v * parameter[1])) / parameter[2]
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


def bag2depth_npy_align(file_name, save_path, todo_frames):
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, file_name)
    config.enable_stream(rs.stream.depth, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 15)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    while todo_frames:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        frame_num = frames.get_frame_number()
        if frame_num not in todo_frames:
            continue

        depth_frame = frames.get_depth_frame()

        depth_npy = np.zeros((480, 640))
        for y in range(480):
            for x in range(640):
                dist = depth_frame.get_distance(x, y)
                depth_npy[y][x] = dist

        np.save(os.path.join(save_path, "frame%06i.npy" % frame_num), depth_npy)
        todo_frames.remove(frame_num)


def param_diff(a, b):
    a_offset = a[3]
    b_offset = b[3]
    new_a, new_b = [], []
    for i in range(len(a)-1):
        new_a.append(a[i] * a_offset)
        new_b.append(b[i] * b_offset)
    diff = np.abs(np.subtract(new_a,new_b))
    diff_mean = np.mean(diff)
    return diff_mean