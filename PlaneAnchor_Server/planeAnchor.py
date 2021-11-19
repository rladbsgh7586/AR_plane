import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import struct
import cv2
import matplotlib.pyplot as plt
import PIL.Image as pilimg
from skimage.measure import find_contours
from utils import *
from scipy.spatial.transform import Rotation as R
import math as m

def host_plane_anchor(room_number, total_image_number):
    cred = credentials.Certificate('key_file.json')
    # firebase_admin.initialize_app(cred, {
    #     'databaseURL': 'https://planeanchor-default-rtdb.firebaseio.com/'
    # })

    dir = db.reference().child('hotspot_list').child(str(room_number))
    plane_number = int(dir.child('plane_number').get())

    data_directory = "./smartphone_indoor/" + str(room_number) + "/"
    result_directory = "./test/inference/" + str(room_number) + "/"
    points = []
    xs = []
    ys = []
    zs = []
    for num in range(total_image_number):
        print("----------------" + str(num) + "----------------")
        plane_mask = np.load(result_directory + str(num) + "_plane_masks_0.npy")
        plane_depth = np.load(result_directory + str(num) + "_plane_depth_0.npy")
        plane_parameters = np.load(result_directory + str(num) + "_plane_parameters_0.npy")
        model_view_matrix = np.load(data_directory + "inverse_model_view_matrix.npy")
        view_matrix = np.load(data_directory + "view_matrix.npy")
        # print(np.shape(model_view_matrix))
        # print(np.shape(view_matrix))
        threshold = 1 / 10 * np.size(plane_mask[0])
        camera_intrinsics = np.loadtxt(data_directory + "camera.txt")
        point_cloud = get_point_cloud(data_directory + str(num + 1) + "_v.txt", 0)
        # point_cloud = get_point_cloud(data_directory + str(num + 1) + "_vp.txt", 0)
        projected_point_cloud = get_projected_point_cloud(point_cloud)
        im = pilimg.open(result_directory + str(num) + "_image_0.png")
        image_path = result_directory + str(num) + "_image_mask"
        image_pixel = np.array(im)
        #
        # translations = [[0,0,-1],
        #                 [-0.5, 0, -1],
        #                 [0.5, 0, -1],
        #                 [0, -0.5, -1],
        #                 [0, 0.5, -1]]

        # for i in translations:
        for i in range(plane_mask.shape[0]):
        # for i in range(1):
        #     i = 2
            if count_mask(plane_mask[i]) < threshold:
                continue
            temp_image_path = image_path + str(i) + ".png"
            transformation_matrix, width, height = get_plane_matrix(plane_mask[i], plane_parameters[i],
                                                                    camera_intrinsics, point_cloud,
                                                                    projected_point_cloud, image_pixel, temp_image_path)

            # camera_normal = [0, 1, 0]
            # plane_normal = [0, 0, 1]
            # rotation_matrix = rotation_matrix_from_vectors(camera_normal, plane_normal)
            # center_translation = np.array([[1, 0, 0, i[0]],
            #                                [0, 1, 0, i[1]],
            #                                [0, 0, 1, i[2]],
            #                                [0, 0, 0, 1]])
            # width = 50
            # height = 50
            # transformation_matrix = get_transformation_matrix(rotation_matrix, center_translation)

            if width != 0:
                plane_number += 1
                print("plane" + str(plane_number))
                transformation_matrix = np.transpose(transformation_matrix)
                transformation_matrix = np.dot(transformation_matrix, model_view_matrix[num])
                dir.child('plane_anchors').child('plane' + str(plane_number)).update(
                    {'transformation_matrix': list(transformation_matrix.reshape(-1))})
                dir.child('plane_anchors').child('plane' + str(plane_number)).update({'width': width})
                dir.child('plane_anchors').child('plane' + str(plane_number)).update({'height': height})
                dir.update({'plane_number': plane_number})
                print("\n")


def get_point_cloud(file_name, confidence_value=-1):
    f = open(file_name, 'rb')
    data = f.read()
    point_cloud = []
    coordinate_length = 4
    for ii in range(int(len(data) / (4 * coordinate_length))):
        i = ii * 4
        confidence = bytearray(data[(i + coordinate_length - 1) * 4:(i + coordinate_length) * 4])
        confidence.reverse()
        confidence = struct.unpack('f', confidence)[0]

        depth = bytearray(data[(i + coordinate_length - 2) * 4:(i + coordinate_length -1) * 4])
        depth.reverse()
        depth = abs(struct.unpack('f', depth)[0])
        # if confidence >= confidence_value:
        #     for j in range(coordinate_length - 1):
        #         swap_data = bytearray(data[(i + j) * 4:(i + j + 1) * 4])
        #         swap_data.reverse()
        #         point_cloud.append(struct.unpack('f', swap_data)[0])
        #     point_cloud.append(confidence)

        if depth <= 10:
            for j in range(coordinate_length):
                swap_data = bytearray(data[(i + j) * 4:(i + j + 1) * 4])
                swap_data.reverse()
                point_cloud.append(struct.unpack('f', swap_data)[0])
    np.set_printoptions(precision=4, suppress=True)
    point_cloud_numpy = np.reshape(point_cloud, (-1, 4))

    f.close()

    return point_cloud_numpy


def get_projected_point_cloud(point_cloud):
    projected_point_cloud = []
    # camera_point_cloud = []
    # print(np.array(point_cloud))
    # inv_view_matrix = np.linalg.inv(view_matrix)
    # print(np.dot(view_matrix, inv_view_matrix))

    # print(view_matrix[3])
    for coordinate in point_cloud:
        # print(camera_coordinate)
        projected_point_cloud.append(
            [coordinate[0] / abs(coordinate[2]), coordinate[1] / abs(coordinate[2])])
        # print(camera_coordinate[0] / abs(camera_coordinate[2]), camera_coordinate[1] / abs(camera_coordinate[2]))

    np.set_printoptions(precision=6, suppress=True)
    # print(np.array(projected_point_cloud))
    return projected_point_cloud


def count_mask(a):
    return np.count_nonzero(a)


def get_plane_matrix(mask, parameter, camera, point_cloud, projected_point_cloud, image, image_path):
    total = 0
    x = 0
    y = 0

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] != 0:
                total += 1
                y += i
                x += j

    center_x = round(x / total)
    center_y = round(y / total)
    center_2d = convert_to_normal_coordinate(center_x * 3 / 4, (center_y * 3 / 4) + 140, camera)

    # plane range in normal coordinate [left, right, bottom, top]
    plane_range = get_plane_range(center_x, center_y, mask, camera, image, image_path)
    print(plane_range)
    plane_point_index = []
    for i in range(len(projected_point_cloud)):
        xy = projected_point_cloud[i]
        if plane_range[0][0] < xy[0] < plane_range[1][0] and plane_range[2][1] < xy[1] < plane_range[3][1]:
            plane_point_index.append(i)

    threshold = 10
    # print(len(plane_point_index))
    if len(plane_point_index) > threshold:
        # plot_points(point_cloud, plane_point_index)
        plane_normal, offset, centroid = plane_points_svd(point_cloud, plane_point_index)
        # plane_normal[2] = abs(plane_normal[2])
        center_3d, width, height = get_plane_info(plane_range, plane_normal, offset)
        print("center", center_3d)
        if abs(plane_normal[2]) > 0.5:
            center_depth = get_center_depth(center_2d, plane_normal, offset)
            center_3d = [-center_2d[0] * center_depth, -center_2d[1] * center_depth, center_depth]
            width = abs(width * center_depth)[0]
            height = abs(height * center_depth)[0]
        else:
            width = abs(width * center_3d[2])[0]
            height = abs(height * center_3d[2])[0]
        print("center", center_3d)
        center_translation = np.array([[1, 0, 0, center_3d[0]],
                                       [0, 1, 0, center_3d[1]],
                                       [0, 0, 1, center_3d[2]],
                                       [0, 0, 0, 1]])

        print(center_3d, width, height)
        camera_normal = [0, 1, 0]
        sub_normal = [1, 0, 0]

        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        # print(plane)
        if abs(plane_normal[1]) > abs(plane_normal[0]):
            plane_normal[0] = 0
            plane_normal = plane_normal / np.linalg.norm(plane_normal)
            rotation_matrix = rotation_matrix_from_vectors(camera_normal, plane_normal)
        else:
            plane_normal[1] = 0
            plane_normal = plane_normal / np.linalg.norm(plane_normal)
            angle = np.arccos(np.dot([0,0,1], plane_normal))
            rotation = R.from_euler('xyz', [90, -math.degrees(angle), 0], degrees=True)
            rotation_matrix = []
            for i in rotation.as_matrix():
                rotation_matrix.append([i[0], i[1], i[2], 0])
            rotation_matrix.append([0, 0, 0, 1])

        print("plane_normal", plane_normal)
        print("normal", np.dot(rotation_matrix, [camera_normal[0],camera_normal[1],camera_normal[2],0]))
        print("sub_normal", np.dot(rotation_matrix, [sub_normal[0], sub_normal[1], sub_normal[2], 0]))

        transformation_matrix = get_transformation_matrix(rotation_matrix, center_translation)
        if width > 3000 or height > 3000:
            return 0, 0, 0
    else:
        return 0, 0, 0

    return transformation_matrix, width, height


def get_plane_range(x, y, mask, camera, image, image_path):
    top_y = -1
    print(np.shape(mask))
    for i in range(mask.shape[0]):
        if mask[i][x] != 0:
            if top_y == -1:
                top_y = i
            else:
                bottom_y = i

    left_x = -1
    for j in range(mask.shape[1]):
        if mask[y][j] != 0:
            if left_x == -1:
                left_x = j
            else:
                right_x = j

    plane_mask = np.zeros(np.shape(mask))
    print(left_x, right_x, bottom_y, top_y)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if left_x <= j <= right_x and bottom_y >= i >= top_y:
                plane_mask[i][j] = 1

    draw_instances(image, plane_mask, image_path)

    left_point = convert_to_normal_coordinate(left_x * 3 / 4, (y * 3 / 4) + 140, camera)
    right_point = convert_to_normal_coordinate(right_x * 3 / 4, (y * 3 / 4) + 140, camera)
    top_point = convert_to_normal_coordinate(x * 3 / 4, (top_y * 3 / 4) + 140, camera)
    bottom_point = convert_to_normal_coordinate(x * 3 / 4, (bottom_y * 3 / 4) + 140, camera)

    return [left_point, right_point, bottom_point, top_point]


def plane_points_svd(point_cloud, index):
    x = []
    y = []
    z = []
    print("number of plane points: ", len(index))
    for i in index:
        z.append(point_cloud[i][2])

    z = []

    for i in index:
        x.append(point_cloud[i][0])
        y.append(point_cloud[i][1])
        z.append(point_cloud[i][2])

    points = np.array([x, y, z])
    # print(points)

    svd = np.linalg.svd(points - np.mean(points, axis=1, keepdims=True))
    left = svd[0]
    centroid = np.mean(points, axis=1, keepdims=True)
    plane_normal = left[:, -1]
    offset = plane_normal[0] * centroid[0] + plane_normal[1] * centroid[1] + plane_normal[2] * centroid[2]

    return plane_normal, offset, centroid


def get_plane_info(plane_range, plane_normal, offset):
    left_2d = plane_range[0]
    right_2d = plane_range[1]
    bottom_2d = plane_range[2]
    top_2d = plane_range[3]

    left_3d = get_3d_point(left_2d, plane_normal, offset)
    right_3d = get_3d_point(right_2d, plane_normal, offset)
    bottom_3d = get_3d_point(bottom_2d, plane_normal, offset)
    top_3d = get_3d_point(top_2d, plane_normal, offset)

    # width = calc_3d_distance(left_3d, right_3d) * 1000
    # height = calc_3d_distance(bottom_3d, top_3d) * 1000

    width = (right_2d[0] - left_2d[0]) * 1000
    height = (bottom_2d[1] - top_2d[1]) * 1000
    center = np.mean([left_3d, right_3d, bottom_3d, top_3d], axis=0)
    # print(left_2d, right_2d, bottom_2d, top_2d)
    # print(left_3d, right_3d, bottom_3d, top_3d)
    # print(width, height)
    # print("center", center)

    return center, width, height


def get_center_depth(center, plane_normal, offset):
    center_depth = offset / (plane_normal[0]*center[0] + plane_normal[1]* center[1] + plane_normal[2])
    return center_depth


def get_3d_point(xy, plane_normal, offset):
    t = offset / np.dot(plane_normal, [xy[0], xy[1], -1])

    return np.array([xy[0] * t, xy[1] * t, -t])


def convert_to_normal_coordinate(x, y, camera):
    u = (x - camera[2]) / camera[0]
    v = - (y - camera[3]) / camera[1]

    return u, v


def get_depth(u, v, parameter):
    offset = np.linalg.norm(parameter)
    para = parameter / offset

    return (offset - (para[0] * u) - (para[1] * v)) / para[2]


def calc_3d_distance(p1, p2):
    squared_dist = np.sum((p1 - p2) ** 2, axis=0)
    return np.sqrt(squared_dist)


def rotation_matrix_from_vectors(vec1, vec2):
    # vec1: source
    # vec2: destination
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    kmat = np.array([[0, -v[2], v[1], 0],
                     [v[2], 0, -v[0], 0],
                     [-v[1], v[0], 0, 0],
                     [0, 0, 0, 0]])
    rotation_matrix = np.eye(4) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    rotation_matrix[3][3] = 1
    return rotation_matrix


def get_transformation_matrix(rotation_matrix, translation_matrix):
    return np.dot(translation_matrix, rotation_matrix)


def plot_points(point_cloud, index):
    x_in = []
    y_in = []
    z_in = []
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i in index:
        x_in.append(point_cloud[i][0])
        y_in.append(point_cloud[i][1])
        z_in.append(point_cloud[i][2])
        print(point_cloud[i])

    x_out = []
    y_out = []
    z_out = []

    for i in range(len(point_cloud)):
        if i not in index:
            x_out.append(point_cloud[i][0])
            y_out.append(point_cloud[i][1])
            z_out.append(point_cloud[i][2])

    ax.scatter(x_in, y_in, z_in, color='orange')
    ax.scatter(x_out, y_out, z_out)
    plt.show()


def draw_instances(image, mask, image_path):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    figsize: (optional) the size of the image.
    """
    ## Number of instances

    ## Generate random colors
    instance_colors = ColorPalette(11).getColorMap(returnTuples=True)

    class_colors = ColorPalette(11).getColorMap(returnTuples=True)
    class_colors[0] = (128, 128, 128)

    masked_image = image.astype(np.uint8).copy()
    masked_image = apply_mask(masked_image.astype(np.float32), mask, instance_colors[1]).astype(np.uint8)

    ## Mask Polygon
    ## Pad to ensure proper polygons for masks that touch image edges.

    padded_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    for verts in contours:
        verts = np.fliplr(verts) - 1
        cv2.polylines(masked_image, np.expand_dims(verts.astype(np.int32), 0), True,
                      color=class_colors[1])

    print(image_path)
    cv2.imwrite(image_path, masked_image)


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  np.minimum(image[:, :, c] *
                                             (1 - alpha) + alpha * color[c], 255),
                                  image[:, :, c])
    return image


def euler_to_quaternion(phi, theta, psi):
    qw = m.cos(phi / 2) * m.cos(theta / 2) * m.cos(psi / 2) + m.sin(phi / 2) * m.sin(theta / 2) * m.sin(psi / 2)
    qx = m.sin(phi / 2) * m.cos(theta / 2) * m.cos(psi / 2) - m.cos(phi / 2) * m.sin(theta / 2) * m.sin(psi / 2)
    qy = m.cos(phi / 2) * m.sin(theta / 2) * m.cos(psi / 2) + m.sin(phi / 2) * m.cos(theta / 2) * m.sin(psi / 2)
    qz = m.cos(phi / 2) * m.cos(theta / 2) * m.sin(psi / 2) - m.sin(phi / 2) * m.sin(theta / 2) * m.cos(psi / 2)

    return [qw, qx, qy, qz]

# def get_plane_info(mask, parameter, camera, plane_depth):
#     total = 0
#     x = 0
#     y = 0
#
#     for i in range(mask.shape[0]):
#         for j in range(mask.shape[1]):
#             if mask[i][j] != 0:
#                 total += 1
#                 y += i
#                 x += j
#
#     center_x = round(x/total)
#     center_y = round(y/total)
#
#     width, height = get_plane_size(center_x, center_y, mask, camera, plane_depth)
#     center_depth = plane_depth[center_y][center_x]
#     u, v = convert_to_normal_coordinate(center_x, center_y, camera)
#     center_translation = np.array([[1, 0, 0, -u*center_depth],
#                                    [0, 1, 0, -v*center_depth],
#                                    [0, 0, 1, -center_depth],
#                                    [0, 0, 0, 1]])
#     print(u*center_depth, v*center_depth, center_depth)
#     print(width, height)
#     center_normal = parameter / np.linalg.norm(parameter)
#     camera_normal = [0, 0, 1]
#     rotation_matrix = rotation_matrix_from_vectors(center_normal, camera_normal)
#     transformation_matrix = get_transformation_matrix(rotation_matrix, center_translation)
#
#     return transformation_matrix, width, height


# def get_plane_size(x, y, mask, camera, plane_depth):
#     top_y = -1
#     for i in range(mask.shape[0]):
#         if mask[i][x] != 0:
#             if top_y == -1:
#                 top_y = i
#             else:
#                 bottom_y = i
#
#     left_x = -1
#     for j in range(mask.shape[1]):
#         if mask[y][j] != 0:
#             if left_x == -1:
#                 left_x = j
#             else:
#                 right_x = j
#
#     left_point = convert_to_normal_coordinate(left_x, y, camera)
#     right_point = convert_to_normal_coordinate(right_x, y, camera)
#     top_point = convert_to_normal_coordinate(x, top_y, camera)
#     bottom_point = convert_to_normal_coordinate(x, bottom_y, camera)
#     left_depth = plane_depth[y][left_x]
#     right_depth = plane_depth[y][right_x]
#     top_depth = plane_depth[top_y][x]
#     bottom_depth = plane_depth[bottom_y][x]
#     print(left_depth, right_depth, top_depth, bottom_depth)
#
#     # print("["+str(x)+", "+str(y)+"]")
#     # print(left_point[0]*left_depth, left_point[1]*left_depth, left_depth)
#     # print(right_point[0]*right_depth, right_point[1]*right_depth, right_depth)
#     # print(top_point[0]*top_depth, top_point[1]*top_depth, top_depth)
#     # print(bottom_point[0]*bottom_depth, bottom_point[1]*bottom_depth, bottom_depth)
#
#     width = calc_3d_distance(left_point[0]*left_depth, left_point[1]*left_depth, left_depth, right_point[0]*right_depth, right_point[1]*right_depth, right_depth)
#     height = calc_3d_distance(top_point[0]*top_depth, top_point[1]*top_depth, top_depth, bottom_point[0]*bottom_depth, bottom_point[1]*bottom_depth, bottom_depth)
#     return width*1000, height*1000

