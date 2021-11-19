import socket
import firebase_admin
from firebase_admin import credentials, storage
from planeDectector import run_model
from firebase_admin import db
from google.cloud import storage as s
from planeAnchor import *
from options import parse_args_custom
import os
import time
from evaluate import evaluate
import numpy as np
from PIL import Image

host = "192.168.1.16"
port = 7586


def get_firebase_image(room):
    # storage = firebase_admin.storage()

    storage_client = s.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    my_prefix = "Image/" + str(room) + "/"
    blobs = storage_client.list_blobs('planeanchor.appspot.com', prefix=my_prefix, delimiter="/")

    inverse_model_view_matrix = []
    view_matrix = []

    image_number = 0
    directory = "./smartphone_indoor/"+str(room)+"/"
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

    for blob in blobs:
        file_path = directory+blob.name.split("/")[-1]
        blob.download_to_filename(file_path)
        print(blob.metadata)
        if 'depth' in file_path:
            parsing_depth_image(file_path, blob.metadata['width'], blob.metadata['height'])
        if 'jpg' in file_path:
            image_number += 1
            inverse_model_view_matrix.append(parsing_modelMatrix(blob.metadata['inverseModelViewMatrix']))
            view_matrix.append(parsing_modelMatrix(blob.metadata['viewMatrix']))
    np.save(directory+"inverse_model_view_matrix", inverse_model_view_matrix)
    np.save(directory + "view_matrix", view_matrix)

    return image_number


def parsing_depth_image(file_name, width, height):
    print(file_name)
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
    new_file_name = file_name[:-3] + 'png'
    depth_np =np.array(depth_image, dtype='u4').reshape(int(height), int(width))
    im = Image.fromarray(depth_np)
    im.save(new_file_name)


def bit2int(bitarray):
    res = 0
    for ele in bitarray:
        res = (res << 1) | ele

    return res


def parsing_modelMatrix(array_string):
    array_string = array_string.replace("[","")
    array_string = array_string.replace("]", "")
    array_string = array_string.replace(" ", "")
    array = np.array(array_string.split(","))
    model_matrix = np.zeros((4,4))
    print(array)
    for i in range(4):
        for j in range(4):
            model_matrix[i][j] = float(array[i*4+j])
    print(model_matrix)
    return model_matrix


if __name__ == "__main__":
    cred = credentials.Certificate('key_file.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://planeanchor-default-rtdb.firebaseio.com/',
        'storageBucket': 'gs://planeanchor.appspot.com'
    })
    # roomCode = 2
    # parsing_depth_image('./smartphone_indoor/7/1_depth.txt', 160, 90)
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
        roomCode = parsingData[1]
        time.sleep(10)
        # break
        # roomCode = 9
        # print(roomCode)

        total_image_number = get_firebase_image(roomCode)
        run_model(roomCode)
        # total_image_number = 4
        host_plane_anchor(roomCode, total_image_number)
        break
        #
        # client_sock.close()
        # server_sock.close()
    #
    # for roomCode in [154, 155, 156, 158, 159]:
    #     total_image_number = get_firebase_image(roomCode)
    #     run_model(roomCode)
    #     # total_image_number = 4
    #     host_plane_anchor(roomCode, total_image_number)

