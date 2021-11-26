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


class Rect:
    def __init__(self, l, r, t, b):
        self.left = l
        self.right = r
        self.top = t
        self.bottom = b

    def is_in_rect(self, x, y):
        if self.left[0] < x < self.right[0] and self.bottom[1] < y < self.top[1]:
            return True
        else:
            return False

    def get_width(self):
        return (self.right[0] - self.left[0]) * 1000

    def get_height(self):
        return (self.top[1] - self.bottom[1]) * 1000


def host_plane(room_number, total_image_number):
    plane_size_threshold = 0.1
    sampled_point_threshold = 5
    zero_padding_pixel = 80
    cred = credentials.Certificate('firebase_key.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://planeanchor-default-rtdb.firebaseio.com/'
    })

    dir = db.reference().child('hotspot_list').child(str(room_number))
    dir.child('plane_anchors').delete()
    dir.update({'plane_number': 0})
    plane_number = int(dir.child('plane_number').get())

    data_directory = "./smartphone_indoor/" + str(room_number) + "/"
    result_directory = "./inference/" + str(room_number) + "/"

    for num in range(total_image_number):
        print("----------------" + str(num) + "----------------")
        plane_mask = np.load(result_directory + str(num) + "_plane_masks_0.npy")
        plane_parameters = np.load(result_directory + str(num) + "_plane_parameters_0.npy")
        inv_model_view_matrix = np.load(data_directory + "inverse_model_view_matrix.npy")
        threshold = plane_size_threshold * np.size(plane_mask[0])
        camera_intrinsics = np.loadtxt(data_directory + "camera.txt")
        point_cloud = get_point_cloud(data_directory + str(num + 1) + "_v.txt", 0)
        projected_point_cloud = get_projected_point_cloud(point_cloud)
        im = pilimg.open(result_directory + str(num) + "_image_0.png")
        image_path = result_directory + str(num) + "_image_mask"
        image_pixel = np.array(im)

        for i in range(plane_mask.shape[0]):
            mask = result_zero_padding(plane_mask[i], zero_padding_pixel)
            if count_mask(mask) < threshold:
                continue
            temp_image_path = image_path + str(i) + ".png"
            transformation_matrix, width, height = get_plane_matrix(mask, camera_intrinsics, point_cloud,
                                                                    projected_point_cloud, image_pixel, temp_image_path,
                                                                    sampled_point_threshold)

            if width != 0:
                plane_number += 1
                transformation_matrix = np.transpose(transformation_matrix)
                transformation_matrix = np.dot(transformation_matrix, inv_model_view_matrix[num])
                dir.child('plane_anchors').child('plane' + str(plane_number)).update(
                    {'transformation_matrix': list(transformation_matrix.reshape(-1))})
                dir.child('plane_anchors').child('plane' + str(plane_number)).update({'width': width})
                dir.child('plane_anchors').child('plane' + str(plane_number)).update({'height': height})
                dir.update({'plane_number': plane_number})


def result_zero_padding(plane_mask, zero_padding_pixel):
    zero_pad = np.zeros([plane_mask.shape[0], zero_padding_pixel])
    plane_mask[:, :zero_padding_pixel] = zero_pad
    plane_mask[:, -zero_padding_pixel:] = zero_pad
    return plane_mask


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

    for coordinate in point_cloud:
        projected_point_cloud.append(
            [coordinate[0] / abs(coordinate[2]), coordinate[1] / abs(coordinate[2])])

    np.set_printoptions(precision=6, suppress=True)
    return projected_point_cloud


def count_mask(a):
    return np.count_nonzero(a)


def get_plane_matrix(mask, camera, point_cloud, projected_point_cloud, image, image_path, sampled_point_threshold):
    # get 2d center coordinate
    total, x, y = 0, 0, 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] != 0:
                total += 1
                y += i
                x += j
    center_x = round(x / total)
    center_y = round(y / total)

    # plane range in normal coordinate [l, r, t, b]
    rect = extract_rect(center_x, center_y, mask, camera, image, image_path)

    sampled_point_index = []
    for i in range(len(projected_point_cloud)):
        xy = projected_point_cloud[i]
        if rect.is_in_rect(xy[0], xy[1]):
            sampled_point_index.append(i)

    if len(sampled_point_index) > sampled_point_threshold:
        plane_normal, offset, centroid = plane_points_svd(point_cloud, sampled_point_index)
        center_3d = get_3d_plane_center(rect, plane_normal, offset)
        width = abs(rect.get_width() * center_3d[2])
        height = abs(rect.get_height() * center_3d[2])

        camera_normal = [0, 1, 0]
        sub_normal = [1, 0, 0]

        # plane_normal = plane_normal / np.linalg.norm(plane_normal)
        # rotation_matrix = rotation_matrix_from_vectors(camera_normal, plane_normal)
        if abs(plane_normal[1]) > abs(plane_normal[0]):
            plane_normal[0] = 0
            plane_normal = plane_normal / np.linalg.norm(plane_normal)
            rotation_matrix = rotation_matrix_from_vectors(camera_normal, plane_normal)
        else:
            plane_normal[1] = 0
            plane_normal = plane_normal / np.linalg.norm(plane_normal)
            angle = np.arccos(np.dot([0, 0, 1], plane_normal))
            rotation = R.from_euler('xyz', [90, -math.degrees(angle), 0], degrees=True)
            rotation_matrix = []
            for i in rotation.as_matrix():
                rotation_matrix.append([i[0], i[1], i[2], 0])
            rotation_matrix.append([0, 0, 0, 1])

        print("plane_normal", plane_normal)
        print("normal", np.dot(rotation_matrix, [camera_normal[0],camera_normal[1],camera_normal[2],0]))
        print("sub_normal", np.dot(rotation_matrix, [sub_normal[0], sub_normal[1], sub_normal[2], 0]))

        print(center_3d)
        center_translation = np.array([[1, 0, 0, center_3d[0]],
                                       [0, 1, 0, center_3d[1]],
                                       [0, 0, 1, center_3d[2]],
                                       [0, 0, 0, 1]])
        print(center_translation)

        transformation_matrix = get_transformation_matrix(rotation_matrix, center_translation)

        # camera_normal = [0, 1, 0]
        # plane_normal = [0, 0, 1]
        # rotation_matrix = rotation_matrix_from_vectors(camera_normal, plane_normal)
        # center_translation = np.array([[1, 0, 0, i[0]],
        #                                [0, 1, 0, i[1]],
        #                                [0, 0, 1, i[2]],
        #                                [0, 0, 0, 1]])
        # transformation_matrix = get_transformation_matrix(rotation_matrix, center_translation)

        return transformation_matrix, width, height
    return 0, 0, 0

def extract_rect(x, y, mask, camera, image, image_path):
    top_y = -1
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

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if left_x <= j <= right_x and bottom_y >= i >= top_y:
                plane_mask[i][j] = 1

    draw_rect(image, plane_mask, image_path)

    l = convert_to_normal_coordinate(left_x, y, camera)
    r = convert_to_normal_coordinate(right_x, y, camera)
    t = convert_to_normal_coordinate(x, top_y, camera)
    b = convert_to_normal_coordinate(x, bottom_y, camera)

    rect = Rect(l, r, t, b)

    return rect


def plane_points_svd(point_cloud, index):
    x, y, z = [], [], []

    for i in index:
        x.append(point_cloud[i][0])
        y.append(point_cloud[i][1])
        z.append(point_cloud[i][2])

    points = np.array([x, y, z])

    centroid = np.mean(points, axis=1, keepdims=True)
    svd = np.linalg.svd(points - centroid)
    left = svd[0]
    plane_normal = left[:, -1]
    offset = plane_normal[0] * centroid[0] + plane_normal[1] * centroid[1] + plane_normal[2] * centroid[2]

    return plane_normal, offset, centroid


def get_3d_plane_center(rect, plane_normal, offset):
    left_3d = get_3d_point(rect.left, plane_normal, offset)
    right_3d = get_3d_point(rect.right, plane_normal, offset)
    bottom_3d = get_3d_point(rect.bottom, plane_normal, offset)
    top_3d = get_3d_point(rect.top, plane_normal, offset)

    center = np.mean([left_3d, right_3d, bottom_3d, top_3d], axis=0)
    return center.reshape(-1)


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


def draw_rect(image, mask, image_path):
    instance_colors = ColorPalette(11).getColorMap(returnTuples=True)

    class_colors = ColorPalette(11).getColorMap(returnTuples=True)
    class_colors[0] = (128, 128, 128)

    masked_image = image.astype(np.uint8).copy()
    masked_image = apply_mask(masked_image.astype(np.float32), mask, instance_colors[1]).astype(np.uint8)

    padded_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    for verts in contours:
        verts = np.fliplr(verts) - 1
        cv2.polylines(masked_image, np.expand_dims(verts.astype(np.int32), 0), True,
                      color=class_colors[1])

    print(image_path)
    cv2.imwrite(image_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))


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