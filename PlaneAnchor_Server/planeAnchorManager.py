import socket
from planeDetector import run_model
from google.cloud import storage as s
from planeAnchor import *
import os
import time
import glob
import numpy as np
import re
from PIL import Image
import sys


def download_device_data(room, image_path):
    storage_client = s.Client()

    file_prefix = "Image/" + str(room) + "/"
    blobs = storage_client.list_blobs('planeanchor.appspot.com', prefix=file_prefix, delimiter="/")
    inverse_model_view_matrix = []
    view_matrix = []

    image_num = 0
    try:
        os.system("rm -r " + image_path)
        os.makedirs(image_path)
    except OSError:
        print('Error: Creating directory. ' + image_path)

    for blob in blobs:
        file_path = image_path + blob.name.split("/")[-1]
        blob.download_to_filename(file_path)
        print(blob.metadata)
        if 'depth' in file_path:
            parsing_depth_image(file_path, image_path, blob.metadata['width'], blob.metadata['height'])
        if 'jpg' in file_path:
            image_num += 1
            try:
                inverse_model_view_matrix.append(parsing_transformation_matrix(blob.metadata['inverseModelViewMatrix']))
            except KeyError:
                print("missed inverse model view matrix ", file_path)
                os.remove(file_path)

    np.save(image_path + "inverse_model_view_matrix", inverse_model_view_matrix)

    preprocess_images(image_path)


def count_device_data(image_path):
    file_list = os.listdir(image_path)
    data_num = 0
    for item in file_list:
        file_name = item.split(".")[0]
        if file_name.isdigit():
            data_num += 1
    return data_num


def parsing_depth_image(file_name, image_path, width, height):
    f = open(file_name, 'rb')
    data = f.read()
    x = np.fromstring(data, dtype=np.uint8)
    depth_bit = np.unpackbits(x)
    depth_image = []
    coordinate_length = 4
    for i in range(int(width)* int(height)):
        depth_confidence = depth_bit[i*16:i*16+3]
        pixel_depth = depth_bit[i*16+3:(i+1)*16]
        depth_image.append(bit2int(pixel_depth))
    new_file_name = str.split(file_name, "/")[-1]
    new_file_name = str.split(new_file_name, ".txt")[0] + '.png'
    add_path = image_path + 'add/'
    if not os.path.exists(add_path):
        os.system("mkdir -p %s" % add_path)
        pass

    depth_np =np.array(depth_image, dtype='u4').reshape(int(height), int(width))
    im = Image.fromarray(depth_np)
    im.save(add_path + new_file_name)


def bit2int(bitarray):
    res = 0
    for ele in bitarray:
        res = (res << 1) | ele

    return res


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


def save_camera_intrinsic(image_path):
    camera_intrinsic = [492, 490, 308, 229, 640, 480]
    f = open(image_path + "camera.txt", "w")
    for i in camera_intrinsic:
        f.write(str(i) + " ")
    f.close()


def preprocess_images(image_path):
    images = glob.glob(image_path + "*.jpg")
    add_path = image_path + "add/"
    for path in images:
        original_img = cv2.imread(path)
        img = original_img[80:560, :, :]
        new_img = np.zeros([480, 640, 3], dtype=np.float32)
        new_img[:, 80:560, :] = img
        file_name = str.split(path, "/")[-1]
        file_name = str.split(file_name, ".jpg")[0]

        cv2.imwrite(image_path + file_name + ".jpg", new_img)
        cv2.imwrite(add_path + file_name + "_ori.jpg", original_img)
        cv2.imwrite(add_path + file_name + "_crop.jpg", img)


def listen_device(host, port):
    while (1):
        server_sock = socket.socket(socket.AF_INET)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((host, port))
        server_sock.listen(1)

        client_sock, addr = server_sock.accept()

        print('Connected by', addr)
        data = client_sock.recv(1024)
        deviceData = data.decode("utf-8")

        parsingData = deviceData.split("/")
        room_num = parsingData[1]
        time.sleep(10)

        download_device_data(room_num)
        total_image_number = count_device_data(room_num)
        save_camera_intrinsic(room_num)
        run_model(room_num)
        host_plane(room_num, total_image_number)

        client_sock.close()
        server_sock.close()


def test_plane_anchor(room_num, skip_download=False, skip_inference=False, method="Ours"):
    image_path = "./smartphone_indoor/%d_%s/" % (room_num, method)
    if skip_download == False:
        download_device_data(room_num, image_path)
    total_image_number = count_device_data(image_path)
    save_camera_intrinsic(image_path)
    if skip_inference == False:
        run_model(room_num, method)
    host_plane(room_num, total_image_number, method)


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def atoi(text):
    return int(text) if text.isdigit() else text

if __name__ == "__main__":
    host = "192.168.1.16"
    port = 7586
    # scenarios = [1]
    scenarios = [1,2,4,6,7, 8, 10, 11, 16, 17, 19, 20, 23, 25]
    for i in scenarios:
        test_plane_anchor(room_num=i, skip_download=False, skip_inference=False, method="ours")
    # update_room_number()
    # listen_device(host, port)
    # test_plane_anchor(room_num=4, skip_download=False, skip_inference=False, method="planercnn")
    # test_plane_anchor(room_num=4, skip_download=False, skip_inference=False, method="planenet")
    # test_plane_anchor(room_num=4, skip_download=False, skip_inference=False, method="mws")
    # test_plane_anchor(room_num=4, skip_download=True, skip_inference=True, method="planenet")
    # test_plane_anchor(room_num=4, skip_download=True, skip_inference=True, method="mws")
    # test_plane_anchor(room_num=4, skip_download=True, skip_inference=False, method="mws")
    # test_plane_anchor(room_num=4, skip_download=True, skip_inference=True, method="planercnn")

    # room_list = sorted(glob.glob("../Evaluation/data/*"), key=lambda x: (len(x), x))
    # for room in room_list:
    #     number = room.split("HOST")[-1]
    #     test_plane_anchor(room_num=number, skip_download=True, skip_inference=True, method="gt")

    # test_plane_anchor(room_num=2, skip_download=False, skip_inference=False, method="ours")
    # test_plane_anchor(room_num=2, skip_download=True, skip_inference=True, method="ours")
