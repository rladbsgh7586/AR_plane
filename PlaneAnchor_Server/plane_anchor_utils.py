import matplotlib.pyplot as plt
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


def get_2d_center_coordinate(mask):
    total, x, y = 0, 0, 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] != 0:
                total += 1
                y += i
                x += j
    center_x = round(x / total)
    center_y = round(y / total)
    return center_x, center_y


def normal_to_rotation_matrix(plane_normal):
    camera_normal = [0, 1, 0]

    # rotation_matrix = rotation_matrix_from_vectors(camera_normal, plane_normal)
    # return rotation_matrix
    if abs(plane_normal[1]) > abs(plane_normal[0]):
        plane_normal[0] = 0
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        rotation_matrix = rotation_matrix_from_vectors(camera_normal, plane_normal)
    else:
        plane_normal[1] = 0
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        # rotation_matrix = rotation_matrix_from_vectors(camera_normal, plane_normal)
        angle = np.arccos(np.dot([0, 0, 1], plane_normal))
        if plane_normal[0] < 0:
            rotation = R.from_euler('xyz', [90, -math.degrees(angle), 0], degrees=True)
        else:
            rotation = R.from_euler('xyz', [90, math.degrees(angle), 0], degrees=True)
        rotation_matrix = []
        for i in rotation.as_matrix():
            rotation_matrix.append([i[0], i[1], i[2], 0])
        rotation_matrix.append([0, 0, 0, 1])

    return rotation_matrix


def extract_rect(x, y, mask, camera, image, image_path):
    min_x, min_y, max_x, max_y = 10000, 10000, -1, -1
    segment_size = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] != 0:
                segment_size += 1
                if i > max_y:
                    max_y = i
                if i < min_y:
                    min_y = i
                if j > max_x:
                    max_x = j
                if j < min_x:
                    min_x = j

    left_x, right_x, top_y, bottom_y = min_x, max_x, min_y, max_y
    plane_mask = np.zeros(np.shape(mask))
    rect_size = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if left_x <= j <= right_x and bottom_y >= i >= top_y:
                rect_size += 1
                plane_mask[i][j] = 1

    print(segment_size, rect_size)
    print(segment_size / rect_size)

    draw_rect(image, plane_mask, image_path)

    l = convert_to_normal_coordinate(left_x, y, camera)
    r = convert_to_normal_coordinate(right_x, y, camera)
    t = convert_to_normal_coordinate(x, top_y, camera)
    b = convert_to_normal_coordinate(x, bottom_y, camera)
    print(l, r, b, t)
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

    return plane_normal, offset[0], centroid


def get_3d_plane_center(rect, plane_normal, offset):
    left_3d = get_3d_point(rect.left, plane_normal, offset)
    right_3d = get_3d_point(rect.right, plane_normal, offset)
    bottom_3d = get_3d_point(rect.bottom, plane_normal, offset)
    top_3d = get_3d_point(rect.top, plane_normal, offset)
    print(left_3d, right_3d, bottom_3d, top_3d)
    center = np.mean([left_3d, right_3d, bottom_3d, top_3d], axis=0)
    return center.reshape(-1)


def get_center_depth(center, plane_normal, offset):
    center_depth = offset / (plane_normal[0]*center[0] + plane_normal[1]* center[1] + plane_normal[2])
    return center_depth


def get_3d_point(xy, plane_normal, offset):
    # t = offset / np.dot(plane_normal, [xy[0], xy[1], -1])
    t = -offset / (plane_normal[0] * xy[0] + plane_normal[1] * xy[1] + plane_normal[2])
    # t = -(offset - (xy[0] * plane_normal[0]) - (xy[1] * plane_normal[1])) / plane_normal[2]
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

def parsing_plane_parameter(plane_parameter):
    offset = np.linalg.norm(plane_parameter)
    plane_normal = plane_parameter / offset
    temp = plane_normal[1]
    plane_normal[0] = -plane_normal[0]
    plane_normal[1] = -plane_normal[2]
    plane_normal[2] = temp

    print("plane_normal", plane_normal)
    print("offset", offset)

    return plane_normal, -offset