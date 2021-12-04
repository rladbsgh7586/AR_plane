import os
import argparse
import pyrealsense2 as rs

import time
import datetime
import rosbag
import numpy as np
import glob
import re
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2

def make_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print("Error: Failed to create the directory.")


def video2image(video_name, save_path):
    make_dir(save_path)
    vidcap = cv2.VideoCapture(video_name)
    count = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
        fname = "{}.jpg".format("{0:05d}".format(count))
        img90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        img = img90[80:560, :, :]
        new_img = np.zeros([480, 640, 3], dtype=np.float32)
        new_img[:, 80:560, :] = img
        cv2.imwrite(save_path + fname, new_img)  # save frame as JPEG file
        count += 1
    print("{} images are extracted in {}.".format(count, save_path))


# def bag2image(bag_name, save_path):
#     make_dir(save_path)
#     bag = rosbag.Bag(bag_name, "r")
#     bridge = CvBridge()
#     count = 0
#     for topic, msg, t in bag.read_messages():
#         if "/data" in topic:
#             if "Color" in topic:
#                 img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
#                 cv2.imwrite(os.path.join(save_path, "frame%06i.png" % count), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#                 count += 1
#     bag.close()


def bag2timestamp(file_name, save_path):
    make_dir(save_path)

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, file_name)
    config.enable_stream(rs.stream.depth, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 15)
    pipeline.start(config)

    start_time = -1
    last_time = -1
    timestamps = []

    while True:
        frames = pipeline.wait_for_frames()
        timestamp = frames.get_timestamp()
        frame_num = frames.get_frame_number()
        if start_time == -1:
            if frame_num == 0:
                start_time = timestamp
            else:
                continue

        time = (timestamp - start_time) / 1000
        if time < last_time:
            break
        last_time = time
        timestamps.append(np.array([frame_num, time]))
    np.save(os.path.join(save_path, "timestamp.npy"), np.array(timestamps))


def bag2img(file_name, save_path):
    img_path = save_path + "image/"
    make_dir(img_path)

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, file_name)
    config.enable_stream(rs.stream.depth, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 15)
    pipeline.start(config)
    colorizer = rs.colorizer()

    count = 0

    timestamp_npy = np.load(save_path + "timestamp.npy")
    todo_frames = get_todo_frames(timestamp_npy)

    while todo_frames:
        frames = pipeline.wait_for_frames()
        frame_num = frames.get_frame_number()
        if frame_num not in todo_frames:
            continue
        color_frame = np.asanyarray(frames.get_color_frame().get_data())
        cv2.imwrite(os.path.join(img_path, "frame%06i.png" % frame_num), cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR))

        todo_frames.remove(frame_num)


def bag2depth_img(file_name, save_path):
    depth_img_path = save_path + "depth_img/"
    make_dir(depth_img_path)

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, file_name)
    config.enable_stream(rs.stream.depth, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 15)
    pipeline.start(config)
    colorizer = rs.colorizer()

    count = 0

    timestamp_npy = np.load(save_path + "timestamp.npy")
    todo_frames = get_todo_frames(timestamp_npy)

    while todo_frames:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        frame_num = frames.get_frame_number()
        if frame_num not in todo_frames:
            continue

        depth_color_frame = colorizer.colorize(depth_frame)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        cv2.imwrite(os.path.join(depth_img_path, "frame%06i.png" % frame_num), depth_color_image)

        todo_frames.remove(frame_num)


def bag2depth_npy(file_name, save_path):
    depth_npy_path = save_path + "depth_npy/"
    make_dir(depth_npy_path)

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, file_name)
    config.enable_stream(rs.stream.depth, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 15)
    pipeline.start(config)
    colorizer = rs.colorizer()

    timestamp_npy = np.load(save_path + "timestamp.npy")
    todo_frames = get_todo_frames(timestamp_npy)

    while todo_frames:
        frames = pipeline.wait_for_frames()
        frame_num = frames.get_frame_number()
        if frame_num not in todo_frames:
            continue

        depth_frame = frames.get_depth_frame()

        depth_npy = np.zeros((480, 640))
        for y in range(480):
            for x in range(640):
                dist = depth_frame.get_distance(x, y)
                depth_npy[y][x] = dist

        np.save(os.path.join(depth_npy_path, "frame%06i.npy" % frame_num), depth_npy)
        todo_frames.remove(frame_num)


def bag2depth_ply(file_name, save_path):
    depth_ply_path = save_path + "depth_ply/"
    make_dir(depth_ply_path)

    pc = rs.pointcloud()
    points = rs.points()

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, file_name)
    config.enable_stream(rs.stream.depth, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 15)
    pipeline.start(config)
    colorizer = rs.colorizer()

    timestamp_npy = np.load(save_path + "timestamp.npy")
    todo_frames = get_todo_frames(timestamp_npy)

    while todo_frames:
        frames = pipeline.wait_for_frames()
        frame_num = frames.get_frame_number()
        if frame_num not in todo_frames:
            continue

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_color_frame = colorizer.colorize(depth_frame)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        cv2.imwrite(os.path.join(depth_ply_path, "frame%06i.png" % frame_num), depth_color_image)

        if color_frame:
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)

            points.export_to_ply(os.path.join(depth_ply_path, "frame%06i.ply" % frame_num), color_frame)
        todo_frames.remove(frame_num)


def get_todo_frames(timestamp_npy):
    todo = []
    for timestamp in timestamp_npy:
        todo.append(timestamp[0])
    return todo


if __name__ == "__main__":
    type = "RESOLVE"
    lists = glob.glob("input_data/*.bag")
    print(lists)
    for file in lists:
        if "HOST" in file:
            type = "HOST"
        if "RESOLVE" in file:
            type = "RESOLVE"
        scenario = re.sub(r'[^0-9]', '', file)

        file_name = "input_data/" + type + str(scenario)
        save_path = "data/" + type + str(scenario) + "/"

        if os.path.exists(save_path):
            continue

        video2image(file_name+".mp4", save_path + "phone_image/")
        bag2timestamp(file_name+".bag", save_path)
        bag2img(file_name + ".bag", save_path)
        bag2depth_img(file_name+".bag", save_path)
        bag2depth_npy(file_name + ".bag", save_path)
        bag2depth_ply(file_name + ".bag", save_path)