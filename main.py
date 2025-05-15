

import struct
from typing import Any, Dict, List, Set

import cv2
from sympy import continued_fraction_convergents
import torch
import pyvirtualcam
import numpy as np
import mediapipe as mp
from PIL import Image

import tha2.poser.modes.mode_20_wx
from models import TalkingAnimeLight, TalkingAnime3
from pose import get_pose
from utils import preprocessing_image, postprocessing_image

import errno
import json
import os
import queue
import socket
import time
import math
from pynput import mouse
import re
from collections import OrderedDict
from multiprocessing import Value, Process, Queue

from pyanime4k import ac

from tha2.mocap.ifacialmocap_constants import *

from args import args

from tha3.util import torch_linear_to_srgb, resize_PIL_image, extract_PIL_image_from_filelike, \
    extract_pytorch_image_from_PIL_image

import collections

from AnimationsTha.animations import AnimationStatesTha
from AnimationsTha.parameters import model_input_split

from python_utils_aisu import utils
import inspect
import logging
from pathlib import Path
import concurrent.futures
from collections import defaultdict
import matplotlib.pyplot as plt
import keyboard

logger = utils.loggingGetLogger(__name__)
logger.setLevel('INFO')

def rescale_01(value, min_new, max_new = 1.0):
    return max(0.0, min(
        1.0, (value - min_new) / (max_new - min_new)
    ))

def clamp(v, min_value = 0.0, max_value = 1.0):
    return max(min_value, min(max_value, v))

def print_number(x):
    return f'{x:>5.2f}'

def print_number_array(xa):
    return (f"{print_number(x)}, " for x in xa)

to_rad_m = 1 / 57.2958

def convert_linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    rgb_image = torch_linear_to_srgb(image[0:3, :, :])
    return torch.cat([rgb_image, image[3:4, :, :]], dim=0)

class FPS:
    def __init__(self, avarageof=50):
        self.frametimestamps = collections.deque(maxlen=avarageof)

    def __call__(self):
        self.frametimestamps.append(time.time())
        if len(self.frametimestamps) > 1:
            return len(self.frametimestamps) / (self.frametimestamps[-1] - self.frametimestamps[0])
        else:
            return 0.0


device = torch.device('cuda') if torch.cuda.is_available() and not args.skip_model else torch.device('cpu')


def create_default_blender_data():
    data = {}

    for blendshape_name in BLENDSHAPE_NAMES:
        data[blendshape_name] = 0.0

    data[HEAD_BONE_X] = 0.0
    data[HEAD_BONE_Y] = 0.0
    data[HEAD_BONE_Z] = 0.0
    data[HEAD_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]

    data[LEFT_EYE_BONE_X] = 0.0
    data[LEFT_EYE_BONE_Y] = 0.0
    data[LEFT_EYE_BONE_Z] = 0.0
    data[LEFT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]

    data[RIGHT_EYE_BONE_X] = 0.0
    data[RIGHT_EYE_BONE_Y] = 0.0
    data[RIGHT_EYE_BONE_Z] = 0.0
    data[RIGHT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]

    return data


class OSFClientProcess(Process):
    def __init__(self):
        super().__init__()
        self.queue = Queue()
        self.should_terminate = Value('b', False)
        self.address = args.osf.split(':')[0]
        self.port = int(args.osf.split(':')[1])
        self.ifm_fps_number = Value('f', 0.0)
        self.perf_time = 0

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("", self.port))

        self.socket.setblocking(True)
        self.socket.settimeout(0.1)

        ifm_fps = FPS()
        while True:
            if self.should_terminate.value:
                break
            try:
                socket_bytes = self.socket.recv(8192)
            except socket.error as e:
                err = e.args[0]
                if err == errno.EAGAIN or err == errno.EWOULDBLOCK or err == 'timed out':
                    continue
                else:
                    raise e

            # socket_string = socket_bytes.decode("utf-8")
            osf_raw = (struct.unpack('=di2f2fB1f4f3f3f68f136f210f14f', socket_bytes))
            # print(osf_raw[432:])
            data = {}
            OpenSeeDataIndex = [
                'time',
                'id',
                'cameraResolutionW',
                'cameraResolutionH',
                'rightEyeOpen',
                'leftEyeOpen',
                'got3DPoints',
                'fit3DError',
                'rawQuaternionX',
                'rawQuaternionY',
                'rawQuaternionZ',
                'rawQuaternionW',
                'rawEulerX',
                'rawEulerY',
                'rawEulerZ',
                'translationY',
                'translationX',
                'translationZ',
            ]
            for i in range(len(OpenSeeDataIndex)):
                data[OpenSeeDataIndex[i]] = osf_raw[i]
            data['translationY'] *= -1
            data['translationZ'] *= -1
            data['rotationY'] = data['rawEulerY']-10
            data['rotationX'] = (-data['rawEulerX'] + 360)%360-180
            data['rotationZ'] = (data['rawEulerZ'] - 90)
            OpenSeeFeatureIndex = [
                'EyeLeft',
                'EyeRight',
                'EyebrowSteepnessLeft',
                'EyebrowUpDownLeft',
                'EyebrowQuirkLeft',
                'EyebrowSteepnessRight',
                'EyebrowUpDownRight',
                'EyebrowQuirkRight',
                'MouthCornerUpDownLeft',
                'MouthCornerInOutLeft',
                'MouthCornerUpDownRight',
                'MouthCornerInOutRight',
                'MouthOpen',
                'MouthWide'
            ]

            for i in range(68):
                data['confidence' + str(i)] = osf_raw[i + 18]
            for i in range(68):
                data['pointsX' + str(i)] = osf_raw[i * 2 + 18 + 68]
                data['pointsY' + str(i)] = osf_raw[i * 2 + 18 + 68 + 1]
            for i in range(70):
                data['points3DX' + str(i)] = osf_raw[i * 3 + 18 + 68 + 68 * 2]
                data['points3DY' + str(i)] = osf_raw[i * 3 + 18 + 68 + 68 * 2 + 1]
                data['points3DZ' + str(i)] = osf_raw[i * 3 + 18 + 68 + 68 * 2 + 2]

            for i in range(len(OpenSeeFeatureIndex)):
                data[OpenSeeFeatureIndex[i]] = osf_raw[i + 432]
            # print(data['rotationX'],data['rotationY'],data['rotationZ'])

            a = np.array([
                data['points3DX66'] - data['points3DX68'] + data['points3DX67'] - data['points3DX69'],
                data['points3DY66'] - data['points3DY68'] + data['points3DY67'] - data['points3DY69'],
                data['points3DZ66'] - data['points3DZ68'] + data['points3DZ67'] - data['points3DZ69']
            ])
            a = (a / np.linalg.norm(a))
            data['eyeRotationX'] = a[0]
            data['eyeRotationY'] = a[1]
            try:
                self.queue.put_nowait(data)
            except queue.Full:
                pass
        self.queue.close()
        self.socket.close()


ifm_converter = tha2.poser.modes.mode_20_wx.IFacialMocapPoseConverter20()


class IFMClientProcess(Process):
    def __init__(self):
        super().__init__()
        self.queue = Queue()
        self.should_terminate = Value('b', False)
        self.address = args.ifm.split(':')[0]
        self.port = int(args.ifm.split(':')[1])
        self.ifm_fps_number = Value('f', 0.0)
        self.perf_time = 0

    def run(self):

        udpClntSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        data = "iFacialMocap_sahuasouryya9218sauhuiayeta91555dy3719"

        data = data.encode('utf-8')

        udpClntSock.sendto(data, (self.address, self.port))

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.socket.bind(("", self.port))
        self.socket.settimeout(0.1)
        ifm_fps = FPS()
        pre_socket_string = ''
        while True:
            if self.should_terminate.value:
                break
            try:
                socket_bytes = self.socket.recv(8192)
            except socket.error as e:
                err = e.args[0]
                if err == errno.EAGAIN or err == errno.EWOULDBLOCK or err == 'timed out':
                    continue
                else:
                    raise e
            socket_string = socket_bytes.decode("utf-8")
            if args.debug and pre_socket_string != socket_string:
                self.ifm_fps_number.value = ifm_fps()
                pre_socket_string = socket_string
            # print(socket_string)
            # blender_data = json.loads(socket_string)
            data = self.convert_from_blender_data(socket_string)

            try:
                self.queue.put_nowait(data)
            except queue.Full:
                pass
        self.queue.close()
        self.socket.close()

    @staticmethod
    def convert_from_blender_data(blender_data):
        data = {}

        for item in blender_data.split('|'):
            if item.find('#') != -1:
                k, arr = item.split('#')
                arr = [float(n) for n in arr.split(',')]
                data[k.replace("_L", "Left").replace("_R", "Right")] = arr
            elif item.find('-') != -1:
                k, v = item.split("-")
                data[k.replace("_L", "Left").replace("_R", "Right")] = float(v) / 100

        to_rad = 57.2958
        data[HEAD_BONE_X] = data["=head"][0] / to_rad
        data[HEAD_BONE_Y] = data["=head"][1] / to_rad
        data[HEAD_BONE_Z] = data["=head"][2] / to_rad
        data[HEAD_BONE_QUAT] = [data["=head"][3], data["=head"][4], data["=head"][5], 1]
        # print(data[HEAD_BONE_QUAT][2],min(data[EYE_BLINK_LEFT],data[EYE_BLINK_RIGHT]))
        data[RIGHT_EYE_BONE_X] = data["rightEye"][0] / to_rad
        data[RIGHT_EYE_BONE_Y] = data["rightEye"][1] / to_rad
        data[RIGHT_EYE_BONE_Z] = data["rightEye"][2] / to_rad
        data[LEFT_EYE_BONE_X] = data["leftEye"][0] / to_rad
        data[LEFT_EYE_BONE_Y] = data["leftEye"][1] / to_rad
        data[LEFT_EYE_BONE_Z] = data["leftEye"][2] / to_rad

        return data


class MouseClientProcess(Process):
    def __init__(self):
        super().__init__()
        self.queue = Queue()

    def run(self):
        self.mouse_controller = mouse.Controller()
        posLimit = [int(x) for x in args.mouse_input.split(',')]
        prev = {
            'eye_l_h_temp': 0,
            'eye_r_h_temp': 0,
            'mouth_ratio': 0,
            'eye_y_ratio': 0,
            'eye_x_ratio': 0,
            'x_angle': 0,
            'y_angle': 0,
            'z_angle': 0,
        }
        while True:
            pos = self.mouse_controller.position
            # print(pos)
            eye_limit = [0.8, 0.5]
            head_eye_reduce = 0.6
            head_slowness = 0.2
            mouse_data = {
                'eye_l_h_temp': 0,
                'eye_r_h_temp': 0,
                'mouth_ratio': 0,
                'eye_y_ratio': np.interp(pos[1], [posLimit[1], posLimit[3]], [1, -1]) * eye_limit[1],
                'eye_x_ratio': np.interp(pos[0], [posLimit[0], posLimit[2]], [1, -1]) * eye_limit[0],
                'x_angle': np.interp(pos[1], [posLimit[1], posLimit[3]], [1, -1]),
                'y_angle': np.interp(pos[0], [posLimit[0], posLimit[2]], [1, -1]),
                'z_angle': 0,
            }
            mouse_data['x_angle'] = np.interp(head_slowness, [0, 1], [prev['x_angle'], mouse_data['x_angle']])
            mouse_data['y_angle'] = np.interp(head_slowness, [0, 1], [prev['y_angle'], mouse_data['y_angle']])
            mouse_data['eye_y_ratio'] -= mouse_data['x_angle'] * eye_limit[1] * head_eye_reduce
            mouse_data['eye_x_ratio'] -= mouse_data['y_angle'] * eye_limit[0] * head_eye_reduce
            if args.bongo:
                mouse_data['y_angle'] += 0.05
                mouse_data['x_angle'] += 0.05
            prev = mouse_data
            self.queue.put_nowait(mouse_data)
            time.sleep(1 / 60)


class ModelClientProcess(Process):
    INPUT_RESOLUTION = 512

    def __init__(
            self,
            # Numpy images
            input_images_list,
        ):
        super().__init__()
        self.should_terminate = Value('b', False)
        self.updated = Value('b', False)
        self.data = None
        self.input_images_list = input_images_list

        self.input_image_current_idx = Value('i', 0)
        self.input_image_current_idx_prev = -999
        
        self.overlay_mask_current_idx = Value('i', -1)
        self.overlay_mask_current_idx_prev = -999
        
        self.overlay_mask_extra_current_idx = Value('i', -1)
        self.overlay_mask_extra_current_idx_prev = -999
        
        self.overlay_mask_incremental_current_idx = Value('i', -1)
        self.overlay_mask_incremental_current_idx_prev = -999
        
        self.overlay_mask_condoms_incremental_current_idx = Value('i', -1)
        self.overlay_mask_condoms_incremental_current_idx_prev = -999

        self.output_queue = Queue()
        self.input_queue = Queue()
        self.model_fps_number = Value('f', 0.0)
        self.gpu_fps_number = Value('f', 0.0)
        self.cache_hit_ratio = Value('f', 0.0)
        self.gpu_cache_hit_ratio = Value('f', 0.0)


    def run(self):
        model = None
        if not args.skip_model:
            model = TalkingAnime3().to(device)
            model = model.eval()
            model = model
            print("Pretrained Model Loaded")

        dtype = torch.half if args.model.endswith('half') else torch.float
        eyebrow_vector = torch.empty(1, 12, dtype=dtype)
        mouth_eye_vector = torch.empty(1, 27, dtype=dtype)
        pose_vector = torch.empty(1, 6, dtype=dtype)

        eyebrow_vector = eyebrow_vector.to(device)
        mouth_eye_vector = mouth_eye_vector.to(device)
        pose_vector = pose_vector.to(device)

        model_cache = OrderedDict()
        input_image_cache = OrderedDict()
        tot = 0
        hit = 0
        hit_in_a_row = 0
        model_fps = FPS()
        gpu_fps = FPS()
        tic = time.perf_counter()
        while True:
            nowaits = 0
            model_input = None
            try:
                model_input = self.input_queue.get()
                while not self.input_queue.empty():
                    nowaits += 1
                    model_input = self.input_queue.get_nowait()
            except queue.Empty:
                pass
            if model_input is None: continue

            if 'model' in args.perf:
                inspect_frame = inspect.getframeinfo(inspect.currentframe())
                print(f"\t\t model: self.input_queue.get_nowait   \n\t\t elapsed = {(time.perf_counter() - tic) * 1000:>6.2g}   {Path(inspect_frame.filename).as_posix()}:{inspect_frame.lineno}", f"nowaits={nowaits}")
                tic = time.perf_counter()

            simplify_arr = [1000] * ifm_converter.pose_size
            if args.simplify >= 1:
                simplify_arr = [200] * ifm_converter.pose_size
                simplify_arr[ifm_converter.eye_wink_left_index] = 50
                simplify_arr[ifm_converter.eye_wink_right_index] = 50
                simplify_arr[ifm_converter.eye_happy_wink_left_index] = 50
                simplify_arr[ifm_converter.eye_happy_wink_right_index] = 50
                simplify_arr[ifm_converter.eye_surprised_left_index] = 30
                simplify_arr[ifm_converter.eye_surprised_right_index] = 30
                simplify_arr[ifm_converter.iris_rotation_x_index] = 25
                simplify_arr[ifm_converter.iris_rotation_y_index] = 25
                simplify_arr[ifm_converter.eye_raised_lower_eyelid_left_index] = 10
                simplify_arr[ifm_converter.eye_raised_lower_eyelid_right_index] = 10
                simplify_arr[ifm_converter.mouth_lowered_corner_left_index] = 5
                simplify_arr[ifm_converter.mouth_lowered_corner_right_index] = 5
                simplify_arr[ifm_converter.mouth_raised_corner_left_index] = 5
                simplify_arr[ifm_converter.mouth_raised_corner_right_index] = 5
            if args.simplify >= 2:
                simplify_arr[ifm_converter.head_x_index] = 100
                simplify_arr[ifm_converter.head_y_index] = 100
                simplify_arr[ifm_converter.eye_surprised_left_index] = 10
                simplify_arr[ifm_converter.eye_surprised_right_index] = 10
                model_input[ifm_converter.eye_wink_left_index] += model_input[
                    ifm_converter.eye_happy_wink_left_index]
                model_input[ifm_converter.eye_happy_wink_left_index] = model_input[
                                                                           ifm_converter.eye_wink_left_index] / 2
                model_input[ifm_converter.eye_wink_left_index] = model_input[
                                                                     ifm_converter.eye_wink_left_index] / 2
                model_input[ifm_converter.eye_wink_right_index] += model_input[
                    ifm_converter.eye_happy_wink_right_index]
                model_input[ifm_converter.eye_happy_wink_right_index] = model_input[
                                                                            ifm_converter.eye_wink_right_index] / 2
                model_input[ifm_converter.eye_wink_right_index] = model_input[
                                                                      ifm_converter.eye_wink_right_index] / 2

                uosum = model_input[ifm_converter.mouth_uuu_index] + \
                        model_input[ifm_converter.mouth_ooo_index]
                model_input[ifm_converter.mouth_ooo_index] = uosum
                model_input[ifm_converter.mouth_uuu_index] = 0
                is_open = (model_input[ifm_converter.mouth_aaa_index] + model_input[
                    ifm_converter.mouth_iii_index] + uosum) > 0
                model_input[ifm_converter.mouth_lowered_corner_left_index] = 0
                model_input[ifm_converter.mouth_lowered_corner_right_index] = 0
                model_input[ifm_converter.mouth_raised_corner_left_index] = 0.5 if is_open else 0
                model_input[ifm_converter.mouth_raised_corner_right_index] = 0.5 if is_open else 0
                simplify_arr[ifm_converter.mouth_lowered_corner_left_index] = 0
                simplify_arr[ifm_converter.mouth_lowered_corner_right_index] = 0
                simplify_arr[ifm_converter.mouth_raised_corner_left_index] = 0
                simplify_arr[ifm_converter.mouth_raised_corner_right_index] = 0
            if args.simplify >= 3:
                simplify_arr[ifm_converter.iris_rotation_x_index] = 20
                simplify_arr[ifm_converter.iris_rotation_y_index] = 20
                simplify_arr[ifm_converter.eye_wink_left_index] = 32
                simplify_arr[ifm_converter.eye_wink_right_index] = 32
                simplify_arr[ifm_converter.eye_happy_wink_left_index] = 32
                simplify_arr[ifm_converter.eye_happy_wink_right_index] = 32
            if args.simplify >= 4:
                simplify_arr[ifm_converter.head_x_index] = 50
                simplify_arr[ifm_converter.head_y_index] = 50
                simplify_arr[ifm_converter.neck_z_index] = 100
                model_input[ifm_converter.eye_raised_lower_eyelid_left_index] = 0
                model_input[ifm_converter.eye_raised_lower_eyelid_right_index] = 0
                simplify_arr[ifm_converter.iris_rotation_x_index] = 10
                simplify_arr[ifm_converter.iris_rotation_y_index] = 10
                simplify_arr[ifm_converter.eye_wink_left_index] = 24
                simplify_arr[ifm_converter.eye_wink_right_index] = 24
                simplify_arr[ifm_converter.eye_happy_wink_left_index] = 24
                simplify_arr[ifm_converter.eye_happy_wink_right_index] = 24
                simplify_arr[ifm_converter.eye_surprised_left_index] = 8
                simplify_arr[ifm_converter.eye_surprised_right_index] = 8
                model_input[ifm_converter.eye_wink_left_index] += model_input[
                    ifm_converter.eye_wink_right_index]
                model_input[ifm_converter.eye_wink_right_index] = model_input[
                                                                      ifm_converter.eye_wink_left_index] / 2
                model_input[ifm_converter.eye_wink_left_index] = model_input[
                                                                     ifm_converter.eye_wink_left_index] / 2

                model_input[ifm_converter.eye_surprised_left_index] += model_input[
                    ifm_converter.eye_surprised_right_index]
                model_input[ifm_converter.eye_surprised_right_index] = model_input[
                                                                           ifm_converter.eye_surprised_left_index] / 2
                model_input[ifm_converter.eye_surprised_left_index] = model_input[
                                                                          ifm_converter.eye_surprised_left_index] / 2

                model_input[ifm_converter.eye_happy_wink_left_index] += model_input[
                    ifm_converter.eye_happy_wink_right_index]
                model_input[ifm_converter.eye_happy_wink_right_index] = model_input[
                                                                            ifm_converter.eye_happy_wink_left_index] / 2
                model_input[ifm_converter.eye_happy_wink_left_index] = model_input[
                                                                           ifm_converter.eye_happy_wink_left_index] / 2
                model_input[ifm_converter.mouth_aaa_index] = min(
                    model_input[ifm_converter.mouth_aaa_index] +
                    model_input[ifm_converter.mouth_ooo_index] / 2 +
                    model_input[ifm_converter.mouth_iii_index] / 2 +
                    model_input[ifm_converter.mouth_uuu_index] / 2, 1
                )
                model_input[ifm_converter.mouth_ooo_index] = 0
                model_input[ifm_converter.mouth_iii_index] = 0
                model_input[ifm_converter.mouth_uuu_index] = 0
            for i in range(4, args.simplify):
                simplify_arr = [max(math.ceil(x * 0.8), 5) for x in simplify_arr]
            for i in range(0, len(simplify_arr)):
                if simplify_arr[i] > 0:
                    model_input[i] = round(model_input[i] * simplify_arr[i]) / simplify_arr[i]
            
            input_image_current_idx = self.input_image_current_idx.value
            overlay_mask_current_idx = self.overlay_mask_current_idx.value
            overlay_mask_extra_current_idx = self.overlay_mask_extra_current_idx.value
            overlay_mask_incremental_current_idx = self.overlay_mask_incremental_current_idx.value
            overlay_mask_condoms_incremental_current_idx = self.overlay_mask_condoms_incremental_current_idx.value

            img_tuple = tuple([
                input_image_current_idx,
                overlay_mask_current_idx,
                overlay_mask_incremental_current_idx,
                overlay_mask_condoms_incremental_current_idx,
                overlay_mask_extra_current_idx,
            ])
            input_hash = hash(tuple(model_input) + img_tuple)
            cached = model_cache.get(input_hash)
            tot += 1
            eyebrow_vector_c = [0.0] * 12
            mouth_eye_vector_c = [0.0] * 27
            if cached is not None and hit_in_a_row < self.model_fps_number.value:
                self.output_queue.put_nowait(cached)
                model_cache.move_to_end(input_hash)
                hit += 1
                hit_in_a_row += 1

                if 'model' in args.perf:
                    inspect_frame = inspect.getframeinfo(inspect.currentframe())
                    print(f"\t\t model: cached   \n\t\t elapsed = {(time.perf_counter() - tic) * 1000:>6.2g}   {Path(inspect_frame.filename).as_posix()}:{inspect_frame.lineno}")
                    tic = time.perf_counter()
            else:
                hit_in_a_row = 0
                if args.eyebrow:
                    for i in range(12):
                        eyebrow_vector[0, i] = model_input[i]
                        eyebrow_vector_c[i] = model_input[i]
                for i in range(27):
                    mouth_eye_vector[0, i] = model_input[i + 12]
                    mouth_eye_vector_c[i] = model_input[i + 12]
                for i in range(6):
                    pose_vector[0, i] = model_input[i + 27 + 12]

                def apply_mask(base_image, overlay_mask_image):
                    return Image.alpha_composite(base_image, overlay_mask_image)
                
                def apply_mask_idx(base_image, overlay_mask_image_idx):
                    if overlay_mask_image_idx > 0:
                        return apply_mask(base_image, self.input_images_list[overlay_mask_image_idx])
                    return base_image
                    
                if (
                    (input_image_current_idx != self.input_image_current_idx_prev) or
                    (overlay_mask_incremental_current_idx != self.overlay_mask_incremental_current_idx_prev) or
                    (overlay_mask_condoms_incremental_current_idx != self.overlay_mask_condoms_incremental_current_idx_prev) or
                    (overlay_mask_current_idx != self.overlay_mask_current_idx_prev) or
                    (overlay_mask_extra_current_idx != self.overlay_mask_extra_current_idx_prev) or
                    False
                ):
                    input_image_hash = hash(img_tuple)
                    input_image = input_image_cache.get(input_image_hash)
                    if input_image is None:
                        input_image_pil = self.input_images_list[input_image_current_idx]
                        input_image_pil = apply_mask_idx(
                            input_image_pil,
                            overlay_mask_extra_current_idx,
                        )
                        input_image_pil = apply_mask_idx(
                            input_image_pil,
                            overlay_mask_current_idx,
                        )
                        input_image_pil = apply_mask_idx(
                            input_image_pil,
                            overlay_mask_incremental_current_idx,
                        )
                        input_image_pil = apply_mask_idx(
                            input_image_pil,
                            overlay_mask_condoms_incremental_current_idx,
                        )
                        input_image = self.img_preprocess(input_image_pil)
                        input_image_cache[input_image_hash] = input_image

                self.input_image_current_idx_prev = input_image_current_idx
                self.overlay_mask_current_idx_prev = overlay_mask_current_idx
                self.overlay_mask_extra_current_idx_prev = overlay_mask_extra_current_idx
                self.overlay_mask_incremental_current_idx_prev = overlay_mask_incremental_current_idx
                self.overlay_mask_condoms_incremental_current_idx_prev = overlay_mask_condoms_incremental_current_idx

                if model is None:
                    output_image = input_image
                else:
                    if 'model' in args.perf:
                        inspect_frame = inspect.getframeinfo(inspect.currentframe())
                        print(f"\t\t model: inference()   \n\t\t elapsed = {(time.perf_counter() - tic) * 1000:>6.2g}   {Path(inspect_frame.filename).as_posix()}:{inspect_frame.lineno}")
                        tic = time.perf_counter()
                    output_image = model(input_image, mouth_eye_vector, pose_vector, eyebrow_vector, mouth_eye_vector_c,
                                         eyebrow_vector_c,
                                         self.gpu_cache_hit_ratio)
                if 'model' in args.perf:
                    torch.cuda.synchronize()
                    inspect_frame = inspect.getframeinfo(inspect.currentframe())
                    print(f"\t\t model: inference   \n\t\t elapsed = {(time.perf_counter() - tic) * 1000:>6.2g}   {Path(inspect_frame.filename).as_posix()}:{inspect_frame.lineno}")
                    tic = time.perf_counter()
                postprocessed_image = output_image[0].float()
                postprocessed_image = convert_linear_to_srgb((postprocessed_image + 1.0) / 2.0)
                c, h, w = postprocessed_image.shape
                postprocessed_image = 255.0 * torch.transpose(postprocessed_image.reshape(c, h * w), 0, 1).reshape(h, w,
                                                                                                                   c)
                postprocessed_image = postprocessed_image.byte().detach().cpu().numpy()
                if 'model' in args.perf:
                    inspect_frame = inspect.getframeinfo(inspect.currentframe())
                    print(f"\t\t model: postprocess   \n\t\t elapsed = {(time.perf_counter() - tic) * 1000:>6.2g}   {Path(inspect_frame.filename).as_posix()}:{inspect_frame.lineno}")
                    tic = time.perf_counter()

                self.output_queue.put_nowait(postprocessed_image)
                if args.debug:
                    self.gpu_fps_number.value = gpu_fps()
                if args.max_cache_len > 0:
                    model_cache[input_hash] = postprocessed_image
                    if len(model_cache) > args.max_cache_len:
                        model_cache.popitem(last=False)
            if args.debug:
                self.model_fps_number.value = model_fps()
                self.cache_hit_ratio.value = hit / tot

    def img_preprocess(self, img):
        # global extra_image
        img = img.convert('RGBA')
        wRatio = img.size[0] / self.INPUT_RESOLUTION
        img = img.resize((self.INPUT_RESOLUTION, int(img.size[1] / wRatio)), resample=Image.Resampling.LANCZOS)
        for i, px in enumerate(img.getdata()):
            if px[3] <= 0:
                y = i // self.INPUT_RESOLUTION
                x = i % self.INPUT_RESOLUTION
                img.putpixel((x, y), (0, 0, 0, 0))
        input_image = preprocessing_image(img.crop((0, 0, self.INPUT_RESOLUTION, self.INPUT_RESOLUTION)))

        # width, height = img.size
        # if height > self.INPUT_RESOLUTION:
        #     extra_image = np.array(img.crop((0, self.INPUT_RESOLUTION, img.size[0], img.size[1])))
        return image_to_device(input_image)


class PlotterProcess(Process):
    def __init__(self):
        super().__init__()
        self.var_history = defaultdict(list)
        self.input_queue = Queue()
        self.var_history_max = 10**4

        self.plt_h_scale = 1
        self.interval_seconds = 3
        self.history_overlap = 3
        self.fps = 60 # assumed

        self.plt_config = {
            'interval': self.fps * self.interval_seconds * self.plt_h_scale,
            'hist_size': self.fps * self.interval_seconds * self.plt_h_scale * self.history_overlap,
        }

    def queue_push(self, var, time_counter, loop_counter):
        self.input_queue.put_nowait({
            'var': var,
            'time_counter': time_counter,
            'loop_counter': loop_counter,
        })

    def run(self):
        tic = time.perf_counter()
        while True:
            plot_data = None
            try:
                plot_data = self.input_queue.get()
            except queue.Empty:
                continue
            if plot_data is None: continue

            tic = time.perf_counter()

            loop_counter = plot_data['loop_counter']
            for k in plot_data['var'].keys():
                self.var_history[k].append(plot_data['var'][k])
                self.var_history[k] = self.var_history[k][-self.var_history_max:]

            if loop_counter > 1 and (loop_counter % self.plt_config['interval']) == 0:
                def plotter(var_history, keys, filename):
                    # Clear axes and re-plot
                    plt.clf()

                    time_slice = var_history["time_counter_blender"][-self.plt_config['hist_size']:]
                    blender_data_slice = var_history['blender_data'][-self.plt_config['hist_size']:]
                    def plot_keys(keys, scalex=True, scaley=True, **kwargs):
                        for key in keys:
                            linestyle = None
                            if "Exp" in key:
                                linestyle='dashed'

                            plt.plot(time_slice,
                                [
                                    v[key]
                                    for v in blender_data_slice
                                ],
                                label=key,
                                scalex=scalex,
                                scaley=scaley,
                                linestyle=linestyle,
                                linewidth=0.75,
                                **kwargs,
                            )

                            # avg_rank = 10
                            # avg = np.convolve([
                            #         v[key]
                            #         for v in blender_data_slice
                            #     ], np.ones(avg_rank)/avg_rank, mode='same')

                            # plt.plot(time_slice,
                            #     avg,
                            #     label=key + "-avg",
                            #     linestyle='dashed',
                            #     scalex=scalex,
                            #     scaley=scaley,
                            #     **kwargs,
                            # )

                    plot_keys(keys)
                    plt.grid(True)
                    plt.tight_layout()
                    plt.legend(loc=2, prop={'size': 3})
                    # plt.ylim(-1.0, 1.0) # Add y-axis limits
                    # print(f"Saving file: {filename}", flush=True)
                    plt.savefig(filename)

                def giga_plotter(var_history):
                    try:
                        plotter(
                            var_history,
                            [
                                'MouthCornerUpDownLeft',
                                'MouthCornerUpDownRight',
                                'MouthCornerInOutLeft',
                                'MouthCornerInOutRight',
                                'MouthOpen',
                                'MouthWide',
                            ],
                            "plt_movement_parameters_osf_mouth.pdf",
                        )

                        plotter(
                            var_history,
                            [
                                'EyebrowUpDownLeft',
                                'EyebrowUpDownRight',
                                'EyebrowSteepnessLeft',
                                'EyebrowSteepnessRight',
                                'EyebrowQuirkLeft',
                                'EyebrowQuirkRight',
                            ],
                            "plt_movement_parameters_osf_eyebrows.pdf",
                        )

                        plotter(
                            var_history,
                            [
                                'EyebrowUpDownLeft',
                                'EyebrowUpDownRight',
                                'EyebrowUpDownClamped',
                                'EyebrowUpDownLeftScaled',
                                'EyebrowUpDownRightScaled',
                                # 'EyebrowUpDownExpS',
                                # 'EyebrowUpDownExp0.5',
                            ],
                            "plt_movement_parameters_eyebrow_scaled.pdf",
                        )

                        plotter(
                            var_history,
                            [
                                'rotationX',
                                'rotationY',
                                'rotationZ',
                            ],
                            "plt_movement_parameters_osf_head.pdf",
                        )

                        plotter(
                            var_history,
                            [
                                'eyeRotationY',
                                'eyeRotationX',
                            ],
                            "plt_movement_parameters_eye_rotation.pdf",
                        )

                        plotter(
                            var_history,
                            [
                                'leftEyeOpen',
                                'rightEyeOpen',
                                'leftEyeOpenExpS',
                                'rightEyeOpenExpS',
                                # 'EyebrowUpDownExp0.5',
                                'eye_open',
                            ],
                            "plt_movement_parameters_eye.pdf",
                        )
                    except Exception as e:
                        logger.exception(e)

                giga_plotter(self.var_history)


                if 'plotter' in args.perf:
                    inspect_frame = inspect.getframeinfo(inspect.currentframe())
                    print(f"\t\t plotter: fin   \n\t\t elapsed = {(time.perf_counter() - tic) * 1000:>6.2g}   {Path(inspect_frame.filename).as_posix()}:{inspect_frame.lineno}")
                    tic = time.perf_counter()


model_input_arr_names = [
    'eyebrow_troubled_left', 'eyebrow_troubled_right', 'eyebrow_angry_left',
    'eyebrow_angry_right', 'eyebrow_lowered_left', 'eyebrow_lowered_right',
    'eyebrow_raised_left', 'eyebrow_raised_right', 'eyebrow_happy_left',
    'eyebrow_happy_right', 'eyebrow_serious_left', 'eyebrow_serious_right',
    # eyebrow_vector_c
    'eye_wink_left', 'eye_wink_right', 'eye_happy_wink_left', # 0
    'eye_happy_wink_right', 'eye_surprised_left', 'eye_surprised_right', # 3
    'eye_relaxed_left', 'eye_relaxed_right', 'eye_unimpressed_left', # 6
    'eye_unimpressed_right', 'eye_raised_lower_eyelid_left', 'eye_raised_lower_eyelid_right', # 9
    'iris_small_left', 'iris_small_right', 'mouth_aaa', # 12
    'mouth_iii', 'mouth_uuu', 'mouth_eee', # 15
    'mouth_ooo', 'mouth_delta', 'mouth_lowered_corner_left', # 18
    'mouth_lowered_corner_right', 'mouth_raised_corner_left', 'mouth_raised_corner_right', # 21
    'mouth_smirk', 'iris_rotation_x', 'iris_rotation_y', # 24
    # mouth_eye_vector_c
    'head_x', 'head_y', 'neck_z', 'body_y', 'body_z', 'breathing',
    # pose_vector_c
]
# pose_vector_c[neck_z] < 0 = tilt left
# pose_vector_c[neck_z] > 0 = tilt right
# pose_vector_c[head_y] < 0 = looking left
# pose_vector_c[head_y] > 0 = looking right
# pose_vector_c[head_x] < 0 = looking down
# pose_vector_c[head_x] > 0 = looking up
eyebrow_vector_c_names = model_input_arr_names[0:12]
mouth_eye_vector_c_names = model_input_arr_names[12:12+27]
pose_vector_c_names = model_input_arr_names[12+27:12+27+6]
def names_to_idxs_dict(l):
    return {
        l[idx]: idx
        for idx in range(len(l))
    }
model_input_arr_name_to_idx = names_to_idxs_dict(model_input_arr_names)
eyebrow_vector_c_name_to_idx = names_to_idxs_dict(eyebrow_vector_c_names)
mouth_eye_vector_c_name_to_idx = names_to_idxs_dict(mouth_eye_vector_c_names)
pose_vector_c_name_to_idx = names_to_idxs_dict(pose_vector_c_names)


eyebrow_troubled_left = eyebrow_vector_c_name_to_idx['eyebrow_troubled_left']
eyebrow_troubled_right = eyebrow_vector_c_name_to_idx['eyebrow_troubled_right']
eyebrow_happy_left = eyebrow_vector_c_name_to_idx['eyebrow_happy_left']
eyebrow_happy_right = eyebrow_vector_c_name_to_idx['eyebrow_happy_right']
eyebrow_angry_left = eyebrow_vector_c_name_to_idx['eyebrow_angry_left']
eyebrow_angry_right = eyebrow_vector_c_name_to_idx['eyebrow_angry_right']
eyebrow_serious_left = eyebrow_vector_c_name_to_idx['eyebrow_serious_left']
eyebrow_serious_right = eyebrow_vector_c_name_to_idx['eyebrow_serious_right']

eyebrow_lowered_left = eyebrow_vector_c_name_to_idx['eyebrow_lowered_left']
eyebrow_lowered_right = eyebrow_vector_c_name_to_idx['eyebrow_lowered_right']

eyebrow_raised_left = eyebrow_vector_c_name_to_idx['eyebrow_raised_left']
eyebrow_raised_right = eyebrow_vector_c_name_to_idx['eyebrow_raised_right']

eye_wink_left = mouth_eye_vector_c_name_to_idx['eye_wink_left']
eye_wink_right = mouth_eye_vector_c_name_to_idx['eye_wink_right']
eye_happy_wink_left = mouth_eye_vector_c_name_to_idx['eye_happy_wink_left']
eye_happy_wink_right = mouth_eye_vector_c_name_to_idx['eye_happy_wink_right']
eye_unimpressed_left = mouth_eye_vector_c_name_to_idx['eye_unimpressed_left']
eye_unimpressed_right = mouth_eye_vector_c_name_to_idx['eye_unimpressed_right']
eye_raised_lower_eyelid_left = mouth_eye_vector_c_name_to_idx['eye_raised_lower_eyelid_left']
eye_raised_lower_eyelid_right = mouth_eye_vector_c_name_to_idx['eye_raised_lower_eyelid_right']
eye_relaxed_left = mouth_eye_vector_c_name_to_idx['eye_relaxed_left']
eye_relaxed_right = mouth_eye_vector_c_name_to_idx['eye_relaxed_right']
eye_surprised_left = mouth_eye_vector_c_name_to_idx['eye_surprised_left']
eye_surprised_right = mouth_eye_vector_c_name_to_idx['eye_surprised_right']

mouth_smirk = mouth_eye_vector_c_name_to_idx['mouth_smirk']
mouth_aaa = mouth_eye_vector_c_name_to_idx['mouth_aaa']
mouth_uuu = mouth_eye_vector_c_name_to_idx['mouth_uuu']
mouth_ooo = mouth_eye_vector_c_name_to_idx['mouth_ooo']
mouth_iii = mouth_eye_vector_c_name_to_idx['mouth_iii']
mouth_eee = mouth_eye_vector_c_name_to_idx['mouth_eee']
mouth_delta = mouth_eye_vector_c_name_to_idx['mouth_delta']
mouth_lowered_corner_left = mouth_eye_vector_c_name_to_idx['mouth_lowered_corner_left']
mouth_lowered_corner_right = mouth_eye_vector_c_name_to_idx['mouth_lowered_corner_right']
mouth_raised_corner_left = mouth_eye_vector_c_name_to_idx['mouth_raised_corner_left']
mouth_raised_corner_right = mouth_eye_vector_c_name_to_idx['mouth_raised_corner_right']
iris_rotation_x = mouth_eye_vector_c_name_to_idx['iris_rotation_x']
iris_rotation_y = mouth_eye_vector_c_name_to_idx['iris_rotation_y']
iris_small_left = mouth_eye_vector_c_name_to_idx['iris_small_left']
iris_small_right = mouth_eye_vector_c_name_to_idx['iris_small_right']

head_x = pose_vector_c_name_to_idx['head_x']
head_y = pose_vector_c_name_to_idx['head_y']
neck_z = pose_vector_c_name_to_idx['neck_z']
body_y = pose_vector_c_name_to_idx['body_y']
body_z = pose_vector_c_name_to_idx['body_z']
breathing = pose_vector_c_name_to_idx['breathing']



detection_results_face_blendshapes = {
    "face_blendshapes": {
        "_neutral": 1.1090934322055546e-06,
        "browDownLeft": 0.015915164723992348,
        "browDownRight": 0.029138997197151184,
        "browInnerUp": 0.0980389192700386,
        "browOuterUpLeft": 0.31564557552337646,
        "browOuterUpRight": 0.09290259331464767,
        "cheekPuff": 1.2152116141805891e-05,
        "cheekSquintLeft": 2.627861306336854e-07,
        "cheekSquintRight": 3.403693256132101e-07,
        "eyeBlinkLeft": 0.10593031346797943,
        "eyeBlinkRight": 0.2897402048110962,
        "eyeLookDownLeft": 0.36658012866973877,
        "eyeLookDownRight": 0.35729309916496277,
        "eyeLookInLeft": 0.00022994456230662763,
        "eyeLookInRight": 0.8785527348518372,
        "eyeLookOutLeft": 0.8738786578178406,
        "eyeLookOutRight": 0.0008621902670711279,
        "eyeLookUpLeft": 0.026638375595211983,
        "eyeLookUpRight": 0.01417731773108244,
        "eyeSquintLeft": 0.29653552174568176,
        "eyeSquintRight": 0.20308230817317963,
        "eyeWideLeft": 0.014349953271448612,
        "eyeWideRight": 0.002306187991052866,
        "jawForward": 4.930951399728656e-05,
        "jawLeft": 0.0002052613563137129,
        "jawOpen": 0.01124357059597969,
        "jawRight": 0.0014180116122588515,
        "mouthClose": 0.001968075055629015,
        "mouthDimpleLeft": 0.001240593963302672,
        "mouthDimpleRight": 0.0008351363940164447,
        "mouthFrownLeft": 0.0017723670462146401,
        "mouthFrownRight": 0.0024346879217773676,
        "mouthFunnel": 0.0010977962519973516,
        "mouthLeft": 0.00041581387631595135,
        "mouthLowerDownLeft": 0.0008013963815756142,
        "mouthLowerDownRight": 0.0009908571373671293,
        "mouthPressLeft": 0.00819825567305088,
        "mouthPressRight": 0.005804543849080801,
        "mouthPucker": 0.3323761522769928,
        "mouthRight": 0.004035480320453644,
        "mouthRollLower": 0.04836229234933853,
        "mouthRollUpper": 0.005762582644820213,
        "mouthShrugLower": 0.029601749032735825,
        "mouthShrugUpper": 0.0051505169831216335,
        "mouthSmileLeft": 4.8480555960850324e-06,
        "mouthSmileRight": 4.550876383291325e-06,
        "mouthStretchLeft": 0.00014152548101264983,
        "mouthStretchRight": 0.0025767209008336067,
        "mouthUpperUpLeft": 4.0023329347604886e-05,
        "mouthUpperUpRight": 3.564128928701393e-05,
        "noseSneerLeft": 4.536689175438369e-06,
        "noseSneerRight": 9.60697661867016e-07
    },
    "rotation": {
        "yaw": 14.716486577209983,
        "pitch": -13.76613020819128,
        "roll": -11.770539263238515
    },
    "timestamp": "2025-02-07T20:48:24.423539"
}
# face_blendshapes are between 0 to 1
# # Head rotation:
# ['rotation']['yaw'] < 0 => looking left
# ['rotation']['yaw'] > 0 => looking right
# ['rotation']['pitch'] < 0 => looking up
# ['rotation']['pitch'] > 0 => looking down
# ['rotation']['roll'] < 0 => tilt right
# ['rotation']['roll'] > 0 => tilt left

# roll

def map_detection_restults_face_blendshapes(
    detection_results_face_blendshapes,
    calibration = {
        "roll": 0.0,
        "yaw": 8.6,
        "pitch": -10,
    },
    settings = {
        "head_pitch_limit_lower": -20,
        "head_pitch_limit_upper": 20,
        "head_yaw_limit_lower": -20,
        "head_yaw_limit_upper": 20,
        "head_roll_limit_lower": -20,
        "head_roll_limit_upper": 20,
    },
):
    # Map `detection_results_face_blendshapes` into `model_input_arr`
    eyebrow_vector_c = [0.0] * 12
    mouth_eye_vector_c = [0.0] * 27
    pose_vector_c = [0.0] * 6

    pose_vector_c[head_x] = rescale_01(
        -(detection_results_face_blendshapes['rotation']['pitch'] - calibration['pitch']),
        settings['head_pitch_limit_lower'],
        settings['head_pitch_limit_upper'],
    )
    pose_vector_c[head_x] = pose_vector_c[head_x] * 2 - 1

    pose_vector_c[head_y] = rescale_01(
        -(detection_results_face_blendshapes['rotation']['yaw'] - calibration['yaw']),
        settings['head_yaw_limit_lower'],
        settings['head_yaw_limit_upper'],
    )
    pose_vector_c[head_y] = pose_vector_c[head_y] * 2 - 1

    pose_vector_c[neck_z] = rescale_01(
        detection_results_face_blendshapes['rotation']['roll'] - calibration['roll'],
        settings['head_roll_limit_lower'],
        settings['head_roll_limit_upper'],
    )
    pose_vector_c[neck_z] = pose_vector_c[neck_z] * 2 - 1

    pose_vector_c[head_x] = clamp(
        pose_vector_c[head_x], -1.0, 1.0)
    pose_vector_c[head_y] = clamp(
        pose_vector_c[head_y], -1.0, 1.0)
    pose_vector_c[neck_z] = clamp(
        pose_vector_c[neck_z], -1.0, 1.0)

    face_blendshapes = detection_results_face_blendshapes['face_blendshapes']


    eyebrow_vector_c[eyebrow_raised_left] = face_blendshapes['browInnerUp'] + face_blendshapes['browOuterUpLeft']
    eyebrow_vector_c[eyebrow_raised_right] = face_blendshapes['browInnerUp'] + face_blendshapes['browOuterUpRight']
    eyebrow_vector_c[eyebrow_lowered_left] = face_blendshapes['browDownLeft']
    eyebrow_vector_c[eyebrow_lowered_right] = face_blendshapes['browDownRight']
    eyebrow_vector_c[eyebrow_angry_left] = face_blendshapes['browDownLeft']
    eyebrow_vector_c[eyebrow_angry_right] = face_blendshapes['browDownRight']
    eyebrow_vector_c[eyebrow_troubled_left] = face_blendshapes['browInnerUp']
    eyebrow_vector_c[eyebrow_troubled_right] = face_blendshapes['browInnerUp']
    eyebrow_vector_c[eyebrow_serious_left] = face_blendshapes['browDownLeft']
    eyebrow_vector_c[eyebrow_serious_right] = face_blendshapes['browDownRight']
    eyebrow_vector_c[eyebrow_happy_left] = face_blendshapes['browOuterUpLeft']
    eyebrow_vector_c[eyebrow_happy_right] = face_blendshapes['browOuterUpRight']

    smile_avg = (face_blendshapes['mouthSmileLeft'] + face_blendshapes['mouthSmileRight']) / 2
    smile_avg = clamp(math.sqrt(smile_avg) * 4)


    mouth_eye_vector_c[eye_wink_left] = face_blendshapes['eyeBlinkLeft'] * (1.0 - smile_avg)
    mouth_eye_vector_c[eye_wink_right] = face_blendshapes['eyeBlinkRight'] * (1.0 - smile_avg)
    # mouth_eye_vector_c[eye_happy_wink_left] = face_blendshapes['eyeBlinkLeft'] * smile_avg
    # mouth_eye_vector_c[eye_happy_wink_right] = face_blendshapes['eyeBlinkRight'] * smile_avg
    mouth_eye_vector_c[eye_relaxed_left] = face_blendshapes['eyeBlinkLeft'] * smile_avg
    mouth_eye_vector_c[eye_relaxed_right] = face_blendshapes['eyeBlinkRight'] * smile_avg
    eyeWide_factor = 5
    eyeWide_avg = (face_blendshapes['eyeWideLeft'] + face_blendshapes['eyeWideRight']) / 2
    mouth_eye_vector_c[eye_surprised_left] = eyeWide_avg * eyeWide_factor
    mouth_eye_vector_c[eye_surprised_right] = eyeWide_avg * eyeWide_factor
    # mouth_eye_vector_c[eye_unimpressed_left] = face_blendshapes['eyeSquintLeft'] + face_blendshapes['eyeBlinkLeft']
    # mouth_eye_vector_c[eye_unimpressed_right] = face_blendshapes['eyeSquintRight'] + face_blendshapes['eyeBlinkRight']
    mouth_eye_vector_c[eye_raised_lower_eyelid_left] = face_blendshapes['eyeSquintLeft'] # rough mapping
    mouth_eye_vector_c[eye_raised_lower_eyelid_right] = face_blendshapes['eyeSquintRight'] # rough mapping
    surprised_factor = ((face_blendshapes['eyeWideLeft'] * eyeWide_factor * 0.1 +
                        face_blendshapes['eyeWideRight'] * eyeWide_factor * 0.1) +
                        face_blendshapes['browInnerUp'] * 0.4 +
                        face_blendshapes['browOuterUpLeft'] * 0.3 +
                        face_blendshapes['browOuterUpRight'] * 0.3)
    surprised = clamp(surprised_factor)
    mouth_eye_vector_c[iris_small_left] = clamp(surprised * 0.5, 0.0, 0.6)
    mouth_eye_vector_c[iris_small_right] = clamp(surprised * 0.5, 0.0, 0.6)

    breathing_rate = 3.0 # + (3 * surprised)
    pose_vector_c[breathing] = (math.sin(time.perf_counter() * breathing_rate) + 1) / 2

    iris_x = ((face_blendshapes['eyeLookUpLeft'] + face_blendshapes['eyeLookUpRight']) - (face_blendshapes['eyeLookDownLeft'] + face_blendshapes['eyeLookDownRight'])) / 2
    iris_y = ((face_blendshapes['eyeLookInLeft'] + face_blendshapes['eyeLookOutRight']) - (face_blendshapes['eyeLookOutLeft'] + face_blendshapes['eyeLookInRight'])) / 2

    mouth_eye_vector_c[iris_rotation_x] = iris_x
    mouth_eye_vector_c[iris_rotation_y] = iris_y


    def map_mouth_expressions(face_blendshapes, mouth_eye_vector_c):
        # exaggeration_factor = 2.0 # tweakable

        # mouth_eye_vector_c[mouth_aaa] = face_blendshapes['jawOpen'] * exaggeration_factor + face_blendshapes['mouthFunnel'] * 0.5 # Added mouthFunnel to aaa, and exaggerate jawOpen
        # mouth_eye_vector_c[mouth_iii] = (face_blendshapes['mouthSmileLeft'] + face_blendshapes['mouthSmileRight']) * exaggeration_factor + face_blendshapes['jawOpen'] * 0.5 # Exaggerate smile and jawOpen a bit
        # mouth_eye_vector_c[mouth_uuu] = face_blendshapes['mouthPucker'] * exaggeration_factor # Exaggerate pucker
        # mouth_eye_vector_c[mouth_eee] = (face_blendshapes['mouthSmileLeft'] + face_blendshapes['mouthSmileRight']) * exaggeration_factor + face_blendshapes['eyeSquintLeft'] * 0.3 + face_blendshapes['eyeSquintRight'] * 0.3 + face_blendshapes['jawOpen'] * 0.8 # Exaggerate smile + squint + jawOpen for eee
        # mouth_eye_vector_c[mouth_ooo] = face_blendshapes['mouthFunnel'] * exaggeration_factor + face_blendshapes['mouthPucker'] * exaggeration_factor # Exaggerate funnel and pucker for ooo
        # mouth_eye_vector_c[mouth_delta] = (face_blendshapes['mouthLowerDownLeft'] + face_blendshapes['mouthLowerDownRight']) * exaggeration_factor # Exaggerate mouthLowerDown

        exaggeration_factor = 1.2 # tweakable

        # Applying exaggerating factors based on examples and heuristic mapping
        mouth_eye_vector_c[mouth_aaa] = (face_blendshapes['jawOpen'] * 1.5 + face_blendshapes['mouthFunnel'] * 0.3 + face_blendshapes['mouthPucker'] * 0.2) * exaggeration_factor
        mouth_eye_vector_c[mouth_aaa] = clamp(mouth_eye_vector_c[mouth_aaa])
        # mouth_eye_vector_c[mouth_iii] = ((face_blendshapes['mouthSmileLeft'] + face_blendshapes['mouthSmileRight']) * 2 + (face_blendshapes['mouthLowerDownLeft'] + face_blendshapes['mouthLowerDownRight']) * 0.5 + face_blendshapes['mouthPressLeft'] * 0.8) * exaggeration_factor
        # mouth_eye_vector_c[mouth_iii] = clamp(mouth_eye_vector_c[mouth_iii])
        # mouth_eye_vector_c[mouth_uuu] = (face_blendshapes['mouthPucker'] * 2) * exaggeration_factor
        # mouth_eye_vector_c[mouth_uuu] = clamp(mouth_eye_vector_c[mouth_uuu])
        # mouth_eye_vector_c[mouth_eee] = (face_blendshapes['jawOpen'] * 0.8 + (face_blendshapes['mouthFunnel'] + face_blendshapes['mouthPucker']) * 0.5 + (face_blendshapes['mouthLowerDownLeft'] + face_blendshapes['mouthLowerDownRight']) * 0.3) * exaggeration_factor
        # mouth_eye_vector_c[mouth_eee] = clamp(mouth_eye_vector_c[mouth_eee])
        mouth_eye_vector_c[mouth_ooo] = (face_blendshapes['jawOpen'] * 6) * (face_blendshapes['mouthFunnel'] * 1.8 + (face_blendshapes['mouthPucker'] - 0.6) * 3) * exaggeration_factor
        mouth_eye_vector_c[mouth_ooo] = clamp(mouth_eye_vector_c[mouth_ooo])

        max_open_mouth = max(
            mouth_eye_vector_c[mouth_aaa],
            mouth_eye_vector_c[mouth_iii],
            mouth_eye_vector_c[mouth_uuu],
            mouth_eye_vector_c[mouth_eee],
            mouth_eye_vector_c[mouth_ooo],
        )
        max_closed_mouth = 1 - max_open_mouth

        mouth_eye_vector_c[mouth_lowered_corner_left] = max_closed_mouth * (face_blendshapes['mouthFrownLeft'] * 1.5) * exaggeration_factor
        mouth_eye_vector_c[mouth_lowered_corner_right] = max_closed_mouth * (face_blendshapes['mouthFrownRight'] * 1.5) * exaggeration_factor
        mouth_eye_vector_c[mouth_raised_corner_left] = max_closed_mouth * (face_blendshapes['mouthSmileLeft'] * 1.7) * exaggeration_factor
        mouth_eye_vector_c[mouth_raised_corner_right] = max_closed_mouth * (face_blendshapes['mouthSmileRight'] * 1.7) * exaggeration_factor

        exaggeration_factor = 3.0 # tweakable
        # mouth_eye_vector_c[mouth_delta] = (face_blendshapes['jawOpen'] * 0.6 + face_blendshapes['mouthFunnel'] * 1.2 + face_blendshapes['mouthPucker'] * 0.8) * exaggeration_factor

        # mouth_eye_vector_c[mouth_smirk] = (face_blendshapes['mouthSmileLeft'] + face_blendshapes['mouthFrownRight']) * (example_mouth_iii['mouthSmileLeft'] + example_mouth_eee['mouthFrownRight']) * exaggeration_factor
        # mouth_eye_vector_c[mouth_smirk] = (face_blendshapes['mouthSmileLeft'] * 1.2 + face_blendshapes['mouthFrownRight'] * 1.2) * exaggeration_factor
        smile_max_diff = max(
            face_blendshapes['mouthSmileLeft'] - face_blendshapes['mouthSmileRight'],
            face_blendshapes['mouthSmileRight'] - face_blendshapes['mouthSmileLeft']
        )
        mouth_eye_vector_c[mouth_smirk] = max_closed_mouth * (smile_max_diff * 8) * exaggeration_factor

        mouth_eye_vector_c[mouth_lowered_corner_left] = clamp(mouth_eye_vector_c[mouth_lowered_corner_left])
        mouth_eye_vector_c[mouth_lowered_corner_right] = clamp(mouth_eye_vector_c[mouth_lowered_corner_right])
        mouth_eye_vector_c[mouth_raised_corner_left] = clamp(mouth_eye_vector_c[mouth_raised_corner_left])
        mouth_eye_vector_c[mouth_raised_corner_right] = clamp(mouth_eye_vector_c[mouth_raised_corner_right])
        mouth_eye_vector_c[mouth_delta] = clamp(mouth_eye_vector_c[mouth_delta])
        mouth_eye_vector_c[mouth_smirk] = clamp(mouth_eye_vector_c[mouth_smirk])

    map_mouth_expressions(face_blendshapes, mouth_eye_vector_c)

    print(
        "smile_avg" ,
        print_number(smile_avg),
        "mouth_smirk" ,
        print_number(mouth_eye_vector_c[mouth_smirk]),
        "aaa" ,
        print_number(mouth_eye_vector_c[mouth_aaa]),
        "jawOpen" ,
        print_number(face_blendshapes['jawOpen']),
        "mouthFunnel" ,
        print_number(face_blendshapes['mouthFunnel']),
        "mouthPucker" ,
        print_number(face_blendshapes['mouthPucker']),

        "lowered_corner_left",
        print_number(mouth_eye_vector_c[mouth_lowered_corner_left]),
        "lowered_corner_right",
        print_number(mouth_eye_vector_c[mouth_lowered_corner_right]),
        "raised_corner_left",
        print_number(mouth_eye_vector_c[mouth_raised_corner_left]),
        "raised_corner_right",
        print_number(mouth_eye_vector_c[mouth_raised_corner_right]),

        "yaw",
        print_number(detection_results_face_blendshapes['rotation']['yaw']),
        "head_y",
        print_number(pose_vector_c[head_y]),
    )

    return eyebrow_vector_c, mouth_eye_vector_c, pose_vector_c



input_images = {}

input_image_current_name = ''
overlay_mask_current_name = ''
overlay_mask_extra_current_name = ''

overlay_mask_incremental_current = {
    # 'tally': {
    #     'name': '',
    #     'value': 0,
    #     'max': 20,
    # },
}

expression_current_name = ''

numpad_scan_code_conversion = {
    0: 115,
    1: 95,
    2: 94,
    3: 93,
    4: 92,
    5: 91,
    6: 90,
    7: 9,
    8: 113,
    9: 73,
    "+": 98,
    "enter": 89,
    ".": 126,
    "/": 6,
    "*": 99,
    "-": 118,
}
scan_code_to_numpad = {v: k for k, v in numpad_scan_code_conversion.items()}

# Chord handling globals
chord_base = None  # Tracks '0' or '1' chord start
chord_timer = None        # Timer for chord sequence timeout
chord_bases = [
    numpad_scan_code_conversion[0],
    numpad_scan_code_conversion[1],
    numpad_scan_code_conversion[2],
    numpad_scan_code_conversion[3],
    numpad_scan_code_conversion[4],
    numpad_scan_code_conversion[7],
]

def torch_from_numpy(image, dtype_str):
    image = torch.from_numpy(image) * 2.0 - 1
    if dtype_str.endswith('half'):
        image = image.half()
    else:
        image = image.float()
    image = image.unsqueeze(0)
    return image

def image_to_device(img):
    return torch_from_numpy(img, args.model).to(device)



@torch.no_grad()
def main():

    if args.mediapipe is not None:
        # HOST = "localhost"
        # PORT = 7879
        HOST, PORT = args.mediapipe.split(":")
        PORT = int(PORT)

        from server_broadcasting_py import Client, logger

        logger.setLevel(logging.DEBUG)

        socket_address = (HOST, PORT)

        def detection_results_face_blendshapes_update(obj):
            global detection_results_face_blendshapes
            detection_results_face_blendshapes = obj
            # print("detection_results_face_blendshapes")
            # print(detection_results_face_blendshapes)
        client = Client(socket_address, detection_results_face_blendshapes_update)
        client.connect()


    global input_images

    imgs_dir = "data/images/"
    imgs = {}
    imgs_paths = {
        # # Aisu
        # ## base images (outfits)
        'gyaru_school': f"custom/533/533af185-[illustriousXLPersonalMerge_v30Noob10based_e12e6ce657]-0.25-20250210233744-1224915828-vt-gyaru_school.png",
        'ribbon': f"custom/533/533af185-[illustriousXLPersonalMerge_v30Noob10based_e12e6ce657]-0.25-20250210233744-1224915828-vt-ribbon.png",
        'armor_bikini': f"custom/533/533af185-[illustriousXLPersonalMerge_v30Noob10based_e12e6ce657]-0.25-20250210233744-1224915828-vt-armor_bikini.png",
        # ## overlay masks
        'mask-face': f"custom/533/533af185-[illustriousXLPersonalMerge_v30Noob10based_e12e6ce657]-0.25-20250210233744-1224915828-vt-mask-face.png",
        'mask-face-blush': f"custom/533/533af185-[illustriousXLPersonalMerge_v30Noob10based_e12e6ce657]-0.25-20250210233744-1224915828-vt-mask-face-blush.png",
        'hair-purple': f"custom/533/533af185-[illustriousXLPersonalMerge_v30Noob10based_e12e6ce657]-0.25-20250210233744-1224915828-vt-mask-hair-purple.png",
    }
    imgs_directory_paths = {
        # the images should be named "{int number}.png" inside the directories.
        'tally': f"custom/533/tally/",
    }
    hotkeys_numpad_image_name = {
        0: '',
        1: 'gyaru_school',
        2: 'ribbon',
    }
    hotkeys_numpad_overlay_masks = {
        0: 'mask-face',
        1: 'mask-face-blush',
    }
    hotkeys_numpad_overlay_mask_extra = {
        0: 'none',
        1: 'hair-purple',
    }
    hotkeys_numpad_expressions = {
        0: '',
        1: 'happy',
        2: 'relaxed',
        4: 'angry',
        5: 'serious',
        6: 'eyebrow_raise',
        7: 'troubled',
    }
    hotkeys_numpad_incremental = {
        1: 'reset',
        2: 'decrement',
        3: 'increment',
    }

    def img_open(path):
        img = None
        errs = []
        for ext in ["", ".png", ".jpg", ".webp", ]:
            try:
                img = Image.open(f"{path}{ext}")
                return img
            except Exception as e:
                errs.append(e)
        raise Exception(errs)


    img = None
    base_img_filepath = f"{imgs_dir}{args.character}"
    print("base_img_filepath='", base_img_filepath,"'")
    img = img_open(base_img_filepath)
    imgs[''] = img

    for k, v in imgs_paths.items():
        imgs[k] = img_open(imgs_dir + v)

    for k, v in imgs_directory_paths.items():
        overlay_mask_incremental_current[k] = {
            'name': '',
            'value': 0,
            'max': 0,
        }
        stat = overlay_mask_incremental_current[k]
        # Each directory has multiple numbered images ("{int number}.png"), load them all into 'k + str(number)' keys
        for img_path in Path(imgs_dir + v).rglob("*.png"):
            img = img_open(img_path)
            value = int(img_path.stem)
            stat['max'] = max(stat['max'], value)
            imgs[k + str(value)] = img
        print('\n', k)
        print(stat)

    extra_image = None

    input_images[''] = img
    print("Character Image Loaded:", args.character)
    for k, v in imgs.items():
        input_images[k] = v
        print("Character Image Loaded:", k)
    
    # hotkeys_all_images = {}
    # hotkeys_all_images.update(hotkeys_numpad_image_name)
    # hotkeys_all_images.update(hotkeys_numpad_overlay_masks)

    input_image_name_to_idx = {}
    input_images_list = []
    
    input_image_name_to_idx[''] = len(input_images_list)
    input_images_list.append(img)

    for img_name, _ in imgs.items():
        input_image_name_to_idx[img_name] = len(input_images_list)
        input_images_list.append(input_images[img_name])
        print("img_name", img_name)

    # Chord keybindings to change outfits (base images) and expressions (overlay masks)
    def reset_chord():
        global chord_base, chord_timer
        if chord_timer:
            chord_timer.cancel()
            chord_timer = None
        chord_base = None

    def on_key_event(event):
        global chord_base, chord_timer

        if event.event_type != keyboard.KEY_DOWN:
            return

        # Check for chord initiation (main keyboard 0/1)
        if event.scan_code in chord_bases and chord_base is None:
            if chord_timer:
                chord_timer.cancel()
            chord_base = scan_code_to_numpad[event.scan_code]
            print(f"Chord started with {chord_base}, press numpad key...")
            chord_timer = threading.Timer(1.0, reset_chord)
            chord_timer.start()
            return

        # Handle numpad key follow-up
        if chord_base is not None and event.scan_code in scan_code_to_numpad:
            numpad_num = scan_code_to_numpad[event.scan_code]

            print(f"Chord {chord_base} + {numpad_num} pressed")

            if chord_base == 0:
                expression = hotkeys_numpad_expressions.get(numpad_num, '')
                if expression:
                    change_expression(expression)
            elif chord_base == 1:
                image = hotkeys_numpad_image_name.get(numpad_num, None)
                if image is not None:
                    change_img(image)
            elif chord_base == 2:
                mask = hotkeys_numpad_overlay_masks.get(numpad_num, None)
                if mask is not None:
                    change_overlay_mask_face(mask)
            elif chord_base == 7:
                mask = hotkeys_numpad_overlay_mask_extra.get(numpad_num, None)
                if mask is not None:
                    change_overlay_mask_extra(mask)
            elif chord_base == 3:
                command = hotkeys_numpad_incremental.get(numpad_num, None)
                change_overlay_mask_incremental(command, 'tally')
            elif chord_base == 4:
                command = hotkeys_numpad_incremental.get(numpad_num, None)
                change_overlay_mask_incremental(command, 'condoms')

            reset_chord()
            return


    # Setup keyboard listener
    keyboard.hook(on_key_event)
    print("Chord hotkeys enabled:")
    print("- Outfits: 1 + numpad key")
    print("- Expressions: 0 + numpad key")
    print("- Expressions: 2 + numpad key")


    def change_img(name: str):
        global input_image_current_name, input_image_current_idx
        input_image_current_name = name
        img_idx = input_image_name_to_idx[input_image_current_name]
        model_process.input_image_current_idx.value = img_idx
        print("Changed BASE IMAGE to", input_image_current_name, img_idx)

    def change_overlay_mask_face(name: str):
        global overlay_mask_current_name
        # if overlay_mask_current_name == name:
        #     overlay_mask_current_name = 'none'
        # else:
        #     overlay_mask_current_name = name
        overlay_mask_current_name = name
        try:
            img_idx = input_image_name_to_idx[overlay_mask_current_name]
        except KeyError:
            img_idx = -1
        model_process.overlay_mask_current_idx.value = img_idx
        print("Changed OVERLAY mask FACE to", overlay_mask_current_name, img_idx)
    
    def change_overlay_mask_extra(name: str):
        global overlay_mask_extra_current_name
        if overlay_mask_extra_current_name == name:
            overlay_mask_extra_current_name = 'none'
        else:
            overlay_mask_extra_current_name = name

        try:
            img_idx = input_image_name_to_idx[overlay_mask_extra_current_name]
        except KeyError:
            img_idx = -1
        model_process.overlay_mask_extra_current_idx.value = img_idx
        print("Changed OVERLAY mask EXTRA to", overlay_mask_extra_current_name, img_idx)


    def change_overlay_mask_incremental(command: str | None, name: str):
        if command is None:
            print("WARN: change_overlay_mask_incremental", "command is None")
            return
        global overlay_mask_incremental_current
        stat = overlay_mask_incremental_current[name]
        if command == 'reset':
            stat['value'] = 0
            stat['name'] = ''
        elif command == 'increment':
            stat['value'] += 1
        elif command == 'decrement':
            stat['value'] -= 1
        
        if stat['value'] > stat['max']:
            print("Incremental: MAX reached on", name)
            stat['value'] = stat['max']
        if stat['value'] < 0:
            print("Incremental: MIN reached on", name)
            stat['value'] = 0
        stat['name'] = name + str(stat['value'])
        print("change_overlay_mask_incremental", name, stat['value'], command)
        try:
            img_idx = input_image_name_to_idx[stat['name']]
            # TODO set input_image_name_to_idx for incremental
        except KeyError:
            img_idx = -1
        model_process_idx_name_map = {
            "tally": model_process.overlay_mask_incremental_current_idx,
            "condoms": model_process.overlay_mask_condoms_incremental_current_idx,
        }
        model_process_idx_name_map[name].value = img_idx
        print("Changed OVERLAY INCREMENTAL mask to", stat['value'], stat['name'], img_idx)


    global expression_current_name
    def change_expression(name: str):
        global expression_current_name
        # if expression_current_name == name:
        #     expression_current_name = ''
        # else:
        #     expression_current_name = name
        expression_current_name = name
        print("Changed expression to", expression_current_name)


    # print("Hotkeys image_name")
    # for k, v in hotkeys_numpad_image_name.items():
    #     keyboard.add_hotkey(
    #         numpad_scan_code_conversion[k],
    #         change_img,
    #         args=(v,),
    #     )
    #     print("Added hotkey numpad", k, ":", v)

    # print("Hotkeys expressions")
    # for k, v in hotkeys_numpad_expressions.items():
    #     keyboard.add_hotkey(
    #         numpad_scan_code_conversion[k],
    #         change_expression,
    #         args=(v,),
    #     )
    #     print("Added hotkey numpad", k, ":", v)


    mouth_corner_scaling = 1.6
    smirk = True
    if smirk:
        mouth_corner_scaling = 2.5
    smirk_min_smile = -0.2
    smirk_min_open = 0.2
    # smirk_max_open = 0.20
    # smirk_max_open = 0.85
    smirk_max_open = 1.0

    MouthOpen_max = 0.625
    MouthWide_max = 1.0

    # EyebrowUpDown_limit_lower = -1.0
    # EyebrowUpDown_limit_upper = 0.9
    EyebrowUpDown_limit_lower = -1.2
    EyebrowUpDown_limit_upper = 0.3

    eyebrow_surprised_threshold = 0.6
    eyebrow_surprised_surprised_max = 0.95

    eyeRotationY_camera_calibration_offset  = -0.15
    # eyeRotationX_camera_calibration_offset  = +0.1
    eyeRotationX_camera_calibration_offset  = +0.0

    # # Head
    # rotationX: positive => looking up, negative => looking down
    # blender_data['rotationX'] < 0 = looking down
    # blender_data['rotationX'] > 0 = looking up
    # rotationY: positive => looking right, negative => looking left
    # blender_data['rotationY'] < 0 = looking right # mirror of tilt ('rotationZ')
    # blender_data['rotationY'] > 0 = looking left
    # rotationZ: positive => tilt left, negative => tilt right
    # blender_data['rotationZ'] < 0 = tilt left # mirror of looking to sides ('rotationY')
    # blender_data['rotationZ'] > 0 = tilt right
    # left = viewer's right
    # right = viewer's left

    # Head rotation limits:
    head_rotationZ_limit_lower = -20 # max tilt left
    head_rotationZ_limit_upper = 20 # max tilt right

    head_rotationX_limit_lower = -30 # max looking down
    head_rotationX_limit_upper = 30 # max looking up

    head_rotationY_limit_lower = -30 # max left,
    head_rotationY_limit_upper = 30 # max right,

    head_glance_limit = max(abs(head_rotationY_limit_lower), head_rotationY_limit_upper)

    Head_divide_tilt_by_rotation = False
    Head_add_glance_to_tilt = True

    # # These are offsets, which means that if the value is positive, it will increase the negative effect (to offset)
    # # Its offsets so that you can take a neutral frame and copy the values over to the offsets, the settings will cancel the camera position to the neutral position (looking straight forward ahead)
    # # rotationX: positive => looking up, negative => looking down
    rotationX_calibration_offset = +30
    # rotationX_calibration_offset = 0
    # # rotationY: positive => looking right, negative => looking left
    rotationY_calibration_offset = -12
    # rotationY_calibration_offset = -6
    # # rotationZ: positive => tilt left, negative => tilt right
    rotationZ_calibration_offset = 4

    # Iris
    iris_limit_up = 1.0
    iris_limit_down = -1.0
    # iris_limit_up = 0.7
    # iris_limit_down = -0.7
    iris_limit_right = 1.0
    iris_limit_left = -1.0

    eye_open_norm_min = 0.2
    eye_open_norm_max = 1.0
    eyeOpen_exp_smoothing = 0.0

    second_screen_coord = [int(x.strip()) for x in args.second_screen_coord.split(',')]

    ## Record Keyboard events until 'esc' is pressed.
    # print("Recording keys!!")
    # recorded = keyboard.record(until='esc')
    # print()
    # print(recorded)
    # for r in recorded:
    #     print("scan_code =", r.scan_code)
    # return


    output_fps = FPS()

    loop_counter = 0
    var_history = defaultdict(list)
    var_history_max = 10**4

    mm = AnimationStatesTha()
    arr = None

    plotter_process = None
    if args.plot_params_capture:
        plotter_process = PlotterProcess()
        if args.plot_params_capture:
            plotter_process.daemon = True
            plotter_process.start()
            print("PlotterProcess Running")

    import threading
    from flask import Flask, request
    app = Flask(__name__)
    # Flask route to receive requests
    @app.route('/movement', methods=['POST'])
    def movement():
        data = request.get_json()
        r = {}
        def e(name, l):
            """
            Use this to register errors and successes on `r`
            """
            try:
                l()
                r[name] = True
            except Exception as e:
                logger.exception(name)
                r[name] = str(e)
        # Call methods to update animation
        for key, value in data.items():
            # By default, allow calling any method on mm
            l = lambda: getattr(mm, key)(**value)
            # Specific calls (overrides default)
            if key == 'sentiments':
                l =  lambda: mm.set_sentiments(data['sentiments'])
            elif key == 'mouth_keyframes':
                l = lambda: mm.start_mouth_keyframes(data['mouth_keyframes'])
            # Safely call it, registering errors
            e(key, l)
        return r

    receive_requests_thread = threading.Thread(target=app.run, kwargs={'port': 7880}, daemon=True)
    receive_requests_thread.start()

    cap = None

    if not args.debug_input:

        if args.ifm is not None:
            client_process = IFMClientProcess()
            client_process.daemon = True
            client_process.start()
            print("iFacialMocap Service Running:", args.ifm)

        elif args.osf is not None:
            client_process = OSFClientProcess()
            client_process.daemon = True
            client_process.start()
            print("OpenSeeFace Service Running:", args.osf)

        elif args.mouse_input is not None:
            mouse_client_process = MouseClientProcess()
            mouse_client_process.daemon = True
            mouse_client_process.start()
            print("Mouse Input Running")

        else:
            if args.input == 'cam':
                for backend in [cv2.CAP_DSHOW, cv2.CAP_FFMPEG, cv2.CAP_VFW, cv2.CAP_MSMF, cv2.CAP_AVFOUNDATION]:
                    try:
                        cap = cv2.VideoCapture(0 + backend)
                        ret, frame = cap.read()
                        if ret is None or not ret:
                            raise Exception("Can't find Camera")
                        break
                    except Exception as e:
                        logging.exception(f"Tried cv2.VideoCapture backend={backend}; Exception={e};")
                if cap is None:
                    return
            elif args.input == 'auto':
                from config_auto import getKwargs
                kwargs = getKwargs()
                mm = AnimationStatesTha(**kwargs)
            else:
                cap = cv2.VideoCapture(args.input)
                frame_count = 0
                os.makedirs(os.path.join('dst', args.character, args.output_dir), exist_ok=True)
                print("Webcam Input Running")

    facemesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    if args.output_webcam:
        cam_scale = 1
        cam_width_scale = 1
        if args.anime4k:
            cam_scale = 2
        if args.alpha_split:
            cam_width_scale = 2
        cam = pyvirtualcam.Camera(width=args.output_w * cam_scale * cam_width_scale, height=args.output_h * cam_scale,
                                  fps=60,
                                  backend=args.output_webcam,
                                  fmt=
                                  {'unitycapture': pyvirtualcam.PixelFormat.RGBA, 'obs': pyvirtualcam.PixelFormat.RGB}[
                                      args.output_webcam])
        print(f'Using virtual camera: {cam.device}')

    a = None

    if args.anime4k:
        parameters = ac.Parameters()
        # enable HDN for ACNet
        parameters.HDN = True

        a = ac.AC(
            managerList=ac.ManagerList([ac.CUDAManager(dID=0)]),
            type=ac.ProcessorType.Cuda_ACNet,
        )

        # a = ac.AC(
        #     managerList=ac.ManagerList([ac.OpenCLACNetManager(pID=6, dID=0)]),
        #     type=ac.ProcessorType.OpenCL_ACNet,
        # )
        a.set_arguments(parameters)
        print("Anime4K Loaded")

    position_vector = [0, 0, 0, 1]
    position_vector_0 = None
    pose_vector_0 = None

    pose_queue = []

    blender_data={}
    if args.ifm:
        blender_data = create_default_blender_data()

    mouse_data = {
        'eye_l_h_temp': 0,
        'eye_r_h_temp': 0,
        'mouth_ratio': 0,
        'eye_y_ratio': 0,
        'eye_x_ratio': 0,
        'x_angle': 0,
        'y_angle': 0,
        'z_angle': 0,
    }

    model_output_first = True
    model_output = None
    model_process = ModelClientProcess(input_images_list)
    model_process.daemon = True
    model_process.start()
    INPUT_RESOLUTION = ModelClientProcess.INPUT_RESOLUTION

    model_input_arr_avg = None
    mouth_eye_vector_c_avg = None
    mouse_controller = mouse.Controller()

    print("Starting loop. Close this console to exit.")

    print_counter = 0
    time_counter_last = time.perf_counter()
    breathing_adder = 0.0
    while True:
        time_counter = time.perf_counter()
        elapsed = time_counter - time_counter_last
        time_counter_last = time_counter

        if 'main' in args.perf:
            tic = time.perf_counter()

        try:
            if args.debug_input:
                eyebrow_vector_c = [0.0] * 12
                mouth_eye_vector_c = [0.0] * 27
                pose_vector_c = [0.0] * 6

                mouth_eye_vector_c[2] = math.sin(time.perf_counter() * 3)
                mouth_eye_vector_c[3] = math.sin(time.perf_counter() * 3)

                mouth_eye_vector_c[14] = 0

                mouth_eye_vector_c[25] = math.sin(time.perf_counter() * 2.2) * 0.2
                mouth_eye_vector_c[26] = math.sin(time.perf_counter() * 3.5) * 0.8

                pose_vector_c[0] = math.sin(time.perf_counter() * 1.1)
                pose_vector_c[1] = math.sin(time.perf_counter() * 1.2)
                pose_vector_c[2] = math.sin(time.perf_counter() * 1.5)

                eyebrow_vector_c[6]=math.sin(time.perf_counter() * 1.1)
                eyebrow_vector_c[7]=math.sin(time.perf_counter() * 1.1)

                pose_vector_c[body_y] = math.sin(time.perf_counter() * 2 + 1)
                pose_vector_c[body_z] = math.sin(time.perf_counter() * 5 + 2)
                pose_vector_c[breathing] = math.sin(time.perf_counter() * 3 + 3)

            elif args.mediapipe is not None:
                eyebrow_vector_c, mouth_eye_vector_c, pose_vector_c = map_detection_restults_face_blendshapes(detection_results_face_blendshapes)

                # plotter_process needs reworking lol
                # if args.plot_params_capture:
                #     plotter_process.queue_push({
                #         detection_results_face_blendshapes['rotation'],
                #     }, time_counter, loop_counter)

            elif args.osf is not None:
                nowaits = 0
                blender_data = None
                try:
                    blender_data = client_process.queue.get()
                    while not client_process.queue.empty():
                        nowaits += 1
                        blender_data = client_process.queue.get_nowait()
                except queue.Empty:
                    pass

                if blender_data is None or len(blender_data) == 0:
                    print("Skipped", blender_data)
                    continue
                # print("elapsed", round(1/elapsed,2), round(elapsed*1000, 2), )
                if 'main' in args.perf:
                    inspect_frame = inspect.getframeinfo(inspect.currentframe())
                    print(f"\t\t main: blender_data   \n\t\t elapsed = {(time.perf_counter() - tic) * 1000:>6.2g}   {Path(inspect_frame.filename).as_posix()}:{inspect_frame.lineno}", f"nowaits={nowaits}")
                    tic = time.perf_counter()

                eyebrow_vector_c = [0.0] * 12
                mouth_eye_vector_c = [0.0] * 27
                pose_vector_c = [0.0] * 6

                # blender_data.keys() =
                # 'id', 'cameraResolutionW', 'cameraResolutionH', 'rightEyeOpen', 'leftEyeOpen', 'got3DPoints', 'fit3DError', 'rawQuaternionX', 'rawQuaternionY', 'rawQuaternionZ', 'rawQuaternionW', 'rawEulerX', 'rawEulerY', 'rawEulerZ', 'translationY', 'translationX', 'translationZ', 'rotationY', 'rotationX', 'rotationZ',
                # ...
                # 'EyeLeft', 'EyeRight', 'EyebrowSteepnessLeft', 'EyebrowUpDownLeft', 'EyebrowQuirkLeft', 'EyebrowSteepnessRight', 'EyebrowUpDownRight', 'EyebrowQuirkRight', 'MouthCornerUpDownLeft', 'MouthCornerInOutLeft', 'MouthCornerUpDownRight', 'MouthCornerInOutRight', 'MouthOpen', 'MouthWide', 'eyeRotationX', 'eyeRotationY',

                EyebrowUpDownLeftScaled = rescale_01(
                    blender_data['EyebrowUpDownLeft'],
                    EyebrowUpDown_limit_lower,
                    EyebrowUpDown_limit_upper,
                )
                EyebrowUpDownLeftScaled = EyebrowUpDownLeftScaled * 2 - 1

                EyebrowUpDownRightScaled = rescale_01(
                    blender_data['EyebrowUpDownRight'],
                    EyebrowUpDown_limit_lower,
                    EyebrowUpDown_limit_upper,
                )
                EyebrowUpDownRightScaled = EyebrowUpDownRightScaled * 2 - 1

                EyebrowUpDown = (
                    EyebrowUpDownLeftScaled ** 3 + EyebrowUpDownRightScaled ** 3
                ) + (
                    (EyebrowUpDownLeftScaled + EyebrowUpDownRightScaled) / 2
                )

                EyebrowUpDownLeftScaled = clamp(EyebrowUpDownLeftScaled, -1.0, 1.0)
                EyebrowUpDownRightScaled = clamp(EyebrowUpDownRightScaled, -1.0, 1.0)

                EyebrowUpDown_min = min(EyebrowUpDownLeftScaled, EyebrowUpDownRightScaled)
                EyebrowUpDown_max = max(EyebrowUpDownLeftScaled, EyebrowUpDownRightScaled)
                EyebrowUpDownClamped = clamp(EyebrowUpDown, EyebrowUpDown_min, EyebrowUpDown_max)

                blender_data['EyebrowUpDownClamped'] = EyebrowUpDownClamped
                blender_data['EyebrowUpDownLeftScaled'] = EyebrowUpDownLeftScaled
                blender_data['EyebrowUpDownRightScaled'] = EyebrowUpDownRightScaled

                EyebrowUpDown_exp_smoothing = 0.5
                if len(var_history['blender_data']) > 0:
                    prev = var_history['blender_data'][-1]['EyebrowUpDownExpS']
                    blender_data['EyebrowUpDownExpS'] = (1 - EyebrowUpDown_exp_smoothing) * EyebrowUpDownClamped + (EyebrowUpDown_exp_smoothing) * prev

                    EyebrowUpDownClamped = blender_data['EyebrowUpDownExpS']
                else:
                    blender_data['EyebrowUpDownExpS'] = EyebrowUpDownClamped


                eyebrows_raised = max(
                    blender_data['EyebrowUpDownLeft'],
                    blender_data['EyebrowUpDownRight'],
                )

                mouth_narrow_open = mouth_uuu
                # mouth_narrow_open = mouth_delta
                # mouth_narrow_open = mouth_ooo
                # data_mouth_open = rescale_01(max(blender_data['MouthOpen'], 0), 0.0, MouthOpen_max)
                data_mouth_open = rescale_01(blender_data['MouthOpen'], 0.0, MouthOpen_max)
                MouthWide = rescale_01(blender_data['MouthWide'], 0.0, MouthWide_max)

                # print("blender_data['MouthOpen']" ,
                #     print_number(blender_data['MouthOpen']),
                #     " (",
                #     print_number(data_mouth_open),
                #     ")",
                # )

                if eyebrows_raised > eyebrow_surprised_threshold:
                    if MouthWide < 0.4:
                        mouth_eye_vector_c[mouth_delta] = data_mouth_open
                    else:
                        mouth_eye_vector_c[mouth_aaa] = (data_mouth_open * (MouthWide))
                        mouth_eye_vector_c[mouth_narrow_open] = (data_mouth_open * (1.0 - MouthWide))
                else:
                    # MouthOpen_uuu = rescale_01(blender_data['MouthOpen'], 0.0, 0.8)
                    # if MouthWide < 0.2:
                    #     mouth_eye_vector_c[mouth_narrow_open] = MouthOpen_uuu
                    # elif MouthWide > 0.8:
                    #     mouth_eye_vector_c[mouth_aaa] = (data_mouth_open)
                    # else:
                    #     mouth_eye_vector_c[mouth_aaa] = (data_mouth_open * (MouthWide))
                    #     mouth_eye_vector_c[mouth_narrow_open] = (MouthOpen_uuu * (1.0 - MouthWide))
                    mouth_eye_vector_c[mouth_aaa] = (data_mouth_open * (MouthWide))
                    mouth_eye_vector_c[mouth_narrow_open] = (data_mouth_open * (1.0 - MouthWide))

                if expression_current_name:
                    if expression_current_name == 'angry':
                        eyebrow_vector_c[eyebrow_angry_left] = 1.0
                        eyebrow_vector_c[eyebrow_angry_right] = 1.0
                    if expression_current_name == 'serious':
                        eyebrow_vector_c[eyebrow_serious_left] = 1.0
                        eyebrow_vector_c[eyebrow_serious_right] = 1.0


                # negative = smile # yup
                max_smile = max(-blender_data['MouthCornerUpDownLeft'], -blender_data['MouthCornerUpDownRight'])
                # # the negative of maximums is not equivalent to the maximums of negatives
                # max_frown = max(blender_data['MouthCornerUpDownLeft'], blender_data['MouthCornerUpDownRight'])

                if blender_data['MouthCornerUpDownLeft'] < 0:
                    mouth_eye_vector_c[mouth_raised_corner_left] = -blender_data['MouthCornerUpDownLeft'] * (
                        mouth_corner_scaling)
                if blender_data['MouthCornerUpDownRight'] < 0:
                    mouth_eye_vector_c[mouth_raised_corner_right] = -blender_data['MouthCornerUpDownRight'] * (
                        mouth_corner_scaling)

                # if blender_data['MouthCornerUpDownLeft'] > 0:
                #     mouth_eye_vector_c[mouth_lowered_corner_left] = blender_data['MouthCornerUpDownLeft'] * (
                #         mouth_corner_scaling)
                # if blender_data['MouthCornerUpDownRight'] > 0:
                #     mouth_eye_vector_c[mouth_lowered_corner_right] = blender_data['MouthCornerUpDownRight'] * (
                #         mouth_corner_scaling)


                if smirk and (
                    (max_smile > smirk_min_smile) and (
                        blender_data['MouthOpen'] > smirk_min_open and
                        blender_data['MouthWide'] > 0.15
                    )
                ) and blender_data['MouthOpen'] <= smirk_max_open:
                    smirk_scaled = rescale_01(max_smile + blender_data['MouthOpen'], 0.0, 0.7)

                    mouth_eye_vector_c[mouth_smirk] = smirk_scaled
                    mouth_eye_vector_c[mouth_iii] = smirk_scaled

                    # mouth_eye_vector_c[mouth_raised_corner_left] = 0.0
                    # mouth_eye_vector_c[mouth_lowered_corner_left] = 0.0
                    # mouth_eye_vector_c[mouth_raised_corner_right] = 0.0
                    # mouth_eye_vector_c[mouth_lowered_corner_right] = 0.0
                    mouth_eye_vector_c[mouth_delta] = 0.0
                    mouth_eye_vector_c[mouth_aaa] = 0.0
                    mouth_eye_vector_c[mouth_ooo] = 0.0
                    mouth_eye_vector_c[mouth_uuu] = 0.0


                # elif blender_data['MouthWide'] > 0.5:
                #     mouth_eye_vector_c[mouth_aaa] = data_mouth_open
                # else:
                #     mouth_eye_vector_c[mouth_ooo] = data_mouth_open


                # * positive = webcam above the screen
                # * negative = webcam below the screen
                blender_data['eyeRotationY'] -= eyeRotationY_camera_calibration_offset
                blender_data['eyeRotationX'] -= eyeRotationX_camera_calibration_offset

                var_history['eyeRotationY'].append(blender_data['eyeRotationY'])
                var_history['eyeRotationX'].append(blender_data['eyeRotationX'])

                # def calculate_jitter(history):
                #     if len(history) < 2:
                #         return 0.0
                #     differences = np.diff(history)
                #     jitter = np.std(differences) if differences.size > 0 else 0
                #     return jitter

                # # Calculate jitter after the loop has finished collecting data
                # jitter_last_x = 20
                # eyeRotationY_jitter = calculate_jitter(var_history['eyeRotationY'][-jitter_last_x:])
                # eyeRotationX_jitter = calculate_jitter(var_history['eyeRotationX'][-jitter_last_x:])

                # print(f"EyeY J {print_number(eyeRotationY_jitter)}")
                # print(f"EyeX J {print_number(eyeRotationX_jitter)}")

                # iris_rotation_x: positive = up, negative = down
                mouth_eye_vector_c[iris_rotation_x] = (
                #     (-blender_data['eyeRotationY'] * 3)
                #     - (blender_data['rotationX'] * to_rad_m * 1.5)
                    (-blender_data['eyeRotationY'] * 1/0.2)
                    # - (blender_data['rotationX'] * to_rad_m * 1.5)
                )
                mouth_eye_vector_c[iris_rotation_y] = (
                #     (+blender_data['eyeRotationX'] * 3)
                #     + (blender_data['rotationY'] * to_rad_m )
                    (+blender_data['eyeRotationX'] * 1/0.4)
                    # + (blender_data['rotationY'] * to_rad_m )
                )

                # mouth_eye_vector_c[iris_rotation_x] = rescale_01(mouth_eye_vector_c[iris_rotation_x], -0.5, 0.5) * 2 - 1.0


                mouth_eye_vector_c[iris_rotation_x] = clamp(
                    mouth_eye_vector_c[iris_rotation_x], iris_limit_down, iris_limit_up)
                mouth_eye_vector_c[iris_rotation_y] = clamp(
                    mouth_eye_vector_c[iris_rotation_y], iris_limit_left, iris_limit_right)


                eye_wink_current_left = eye_wink_left
                eye_wink_current_right = eye_wink_right
                # eye_wink_current_left = eye_happy_wink_left
                # eye_wink_current_right = eye_happy_wink_right
                if expression_current_name == 'relaxed':
                    eye_wink_current_left = eye_relaxed_left
                    eye_wink_current_right = eye_relaxed_right
                elif expression_current_name == 'happy':
                    eye_wink_current_left = eye_happy_wink_left
                    eye_wink_current_right = eye_happy_wink_right
                # elif expression_current_name == '':
                #     eye_wink_current_left = eye_wink_left
                #     eye_wink_current_right = eye_wink_right

                eyebrows_set = True
                if args.eyebrows_sync:
                    if args.eyebrows_sync == 'cube':
                        # print("EyebrowUpDownClamped=", print_number(EyebrowUpDownClamped))
                        if EyebrowUpDownClamped > 0.0:
                            eyebrow_vector_c[eyebrow_raised_left] = EyebrowUpDownClamped
                            eyebrow_vector_c[eyebrow_raised_right] = EyebrowUpDownClamped
                        else:
                            eyebrow_vector_c[eyebrow_lowered_left] = -EyebrowUpDownClamped
                            eyebrow_vector_c[eyebrow_lowered_right] = -EyebrowUpDownClamped
                            eyebrow_troubled_min = -0.0
                            eyebrow_troubled_max = -0.8
                            if EyebrowUpDownClamped < eyebrow_troubled_min:
                                EyebrowTroubled = rescale_01(
                                    EyebrowUpDownClamped,
                                    eyebrow_troubled_min,
                                    eyebrow_troubled_max,
                                )
                                if expression_current_name == 'angry':
                                    if EyebrowTroubled > 0.3:
                                        EyebrowTroubled = 0.3
                                eyebrow_vector_c[eyebrow_troubled_left] = EyebrowTroubled
                                eyebrow_vector_c[eyebrow_troubled_right] = EyebrowTroubled
                    elif args.eyebrows_sync == 'set_max':
                        eyebrow_vector_c[eyebrow_raised_left] = eyebrows_raised
                        eyebrow_vector_c[eyebrow_raised_right] = eyebrows_raised
                    elif args.eyebrows_sync == 'set_max_avg':
                        eyebrow_vector_c[eyebrow_raised_left] = (
                            eyebrows_raised + blender_data['EyebrowUpDownLeft']) *0.5
                        eyebrow_vector_c[eyebrow_raised_right] = (
                            eyebrows_raised + blender_data['EyebrowUpDownRight']) *0.5
                    else:
                        eyebrows_set = False
                if not args.eyebrows_sync or not eyebrows_set:
                    eyebrow_vector_c[eyebrow_raised_left] = blender_data['EyebrowUpDownLeft']
                    eyebrow_vector_c[eyebrow_raised_right] = blender_data['EyebrowUpDownRight']


                if (
                    eyebrow_vector_c[eyebrow_troubled_left] == 0 and
                    eyebrow_vector_c[eyebrow_troubled_right] == 0
                ) and (
                    eyebrow_vector_c[eyebrow_angry_left] == 0 and
                    eyebrow_vector_c[eyebrow_angry_right] == 0
                ):
                    eyebrow_vector_c[eyebrow_happy_left] = max_smile
                    eyebrow_vector_c[eyebrow_happy_right] = max_smile

                surprised = rescale_01(eyebrows_raised,
                    eyebrow_surprised_threshold,
                    eyebrow_surprised_surprised_max,
                )
                surprised = clamp(surprised)

                # Eye Open
                if len(var_history['blender_data']) > 0:
                    prev = var_history['blender_data'][-1]['leftEyeOpenExpS']
                    blender_data['leftEyeOpenExpS'] = (1 - eyeOpen_exp_smoothing) * blender_data['leftEyeOpen'] + (eyeOpen_exp_smoothing) * prev

                    prev = var_history['blender_data'][-1]['rightEyeOpenExpS']
                    blender_data['rightEyeOpenExpS'] = (1 - eyeOpen_exp_smoothing) * blender_data['rightEyeOpen'] + (eyeOpen_exp_smoothing) * prev
                else:
                    blender_data['leftEyeOpenExpS'] = blender_data['leftEyeOpen']
                    blender_data['rightEyeOpenExpS'] = blender_data['rightEyeOpen']

                leftEyeOpen = 'leftEyeOpenExpS'
                rightEyeOpen = 'rightEyeOpenExpS'

                eye_open_set = True
                blender_eye_open_left = rescale_01(blender_data[leftEyeOpen],
                    eye_open_norm_min,
                    eye_open_norm_max,
                )
                blender_eye_open_right = rescale_01(blender_data[rightEyeOpen],
                    eye_open_norm_min,
                    eye_open_norm_max,
                )
                blender_eye_open = max(blender_eye_open_left, blender_eye_open_right)

                # EyeOpen_factor = 2.1
                # EyeOpen_factor = 0.4
                # blender_eye_open = (1.0 - blender_eye_open) ** EyeOpen_factor
                # blender_eye_open = (1.0 - blender_eye_open)

                blender_data['eye_open'] = blender_eye_open

                if args.eye_open_sync:

                    if args.eye_open_sync == 'set_max':
                        mouth_eye_vector_c[eye_wink_current_left] = (1.0 - blender_eye_open)
                        mouth_eye_vector_c[eye_wink_current_right] = (1.0 - blender_eye_open)

                    elif args.eye_open_sync == 'set_max_avg':
                        mouth_eye_vector_c[eye_wink_current_left] = (
                            blender_eye_open + blender_eye_open_left) * 0.5
                        mouth_eye_vector_c[eye_wink_current_right] =  (
                            blender_eye_open + blender_eye_open_right) * 0.5

                    elif args.eye_open_sync == 'apart_or_close':
                        eye_open_difference = abs(blender_data[leftEyeOpen] - blender_data[rightEyeOpen])
                        if eye_open_difference < 0.4:
                            mouth_eye_vector_c[eye_wink_current_left] = (1 - blender_eye_open)
                            mouth_eye_vector_c[eye_wink_current_right] = (1 - blender_eye_open)
                        else:
                            eye_open_set = False

                    else:
                        eye_open_set = False
                if not args.eye_open_sync or not eye_open_set:
                    mouth_eye_vector_c[eye_wink_current_left] = (1 - blender_eye_open_left)
                    mouth_eye_vector_c[eye_wink_current_right] = (1 - blender_eye_open_right)


                if True:
                    unimpressed_max = 0.05
                    def eye_unimpressed_set(wink, unim):
                        if (
                            mouth_eye_vector_c[wink] <= 0.4
                        ):
                            mouth_eye_vector_c[unim] = rescale_01(mouth_eye_vector_c[wink],
                                0.0,
                                unimpressed_max,
                            )
                            mouth_eye_vector_c[wink] = 0.0
                        elif (
                            mouth_eye_vector_c[wink] > 0.4 and
                            mouth_eye_vector_c[wink] <= 0.6
                        ):
                            mouth_eye_vector_c[unim] = rescale_01(mouth_eye_vector_c[wink],
                                0.0,
                                unimpressed_max,
                            )
                            mouth_eye_vector_c[wink] = rescale_01(mouth_eye_vector_c[wink],
                                0.3,
                                1.0,
                            )

                    eye_unimpressed_set(eye_wink_current_left, eye_unimpressed_left)
                    eye_unimpressed_set(eye_wink_current_right, eye_unimpressed_right)
                    

                if ((expression_current_name == 'relaxed') and surprised < 0.25):
                    mouth_eye_vector_c[eye_unimpressed_left] = max(0.5, mouth_eye_vector_c[eye_unimpressed_left])
                    mouth_eye_vector_c[eye_unimpressed_right] = max(0.5, mouth_eye_vector_c[eye_unimpressed_right])
                
                if expression_current_name == 'eyebrow_raise':
                    mouth_eye_vector_c[eye_unimpressed_left] = max(0.5, mouth_eye_vector_c[eye_unimpressed_left])
                    mouth_eye_vector_c[eye_unimpressed_right] = max(0.5, mouth_eye_vector_c[eye_unimpressed_right])
                    
                    eyebrow_vector_c[eyebrow_troubled_right] = max(1.0, eyebrow_vector_c[eyebrow_troubled_right])
                    
                if expression_current_name == 'troubled':
                    eyebrow_vector_c[eyebrow_troubled_left] = max(0.9, eyebrow_vector_c[eyebrow_troubled_left])
                    eyebrow_vector_c[eyebrow_troubled_right] = max(0.9, eyebrow_vector_c[eyebrow_troubled_right])


                # eyebrow_troubled_left = eyebrow_vector_c_name_to_idx['eyebrow_troubled_left']
                # eyebrow_troubled_right = eyebrow_vector_c_name_to_idx['eyebrow_troubled_right']

                if eyebrows_raised > eyebrow_surprised_threshold:
                    if mouth_eye_vector_c[eye_wink_current_left] <= 0.05 and mouth_eye_vector_c[eye_unimpressed_left] <= 0.0:
                        mouth_eye_vector_c[eye_surprised_left] = surprised * 2.0
                    if mouth_eye_vector_c[eye_wink_current_right] <= 0.05 and mouth_eye_vector_c[eye_unimpressed_right] <= 0.0:
                        mouth_eye_vector_c[eye_surprised_right] = surprised * 2.0

                    mouth_eye_vector_c[iris_small_left] = surprised / 3
                    mouth_eye_vector_c[iris_small_right] = surprised / 3

                    if print_counter % 10 == 0:
                        print("surprised" ,
                            print_number(surprised),
                            " (",
                            print_number(mouth_eye_vector_c[eye_surprised_left]),
                            print_number(mouth_eye_vector_c[eye_surprised_right]),
                            ")",
                            "  eyebrows_raised=", print_number(eyebrows_raised),
                        )
                    
                print_counter += 1
                
                # # Head
                # if pose_vector_0==None:
                #     pose_vector_0=[0,0,0]
                #     pose_vector_0[0] = blender_data['rotationX']
                #     pose_vector_0[1] = blender_data['rotationY']
                #     pose_vector_0[2] = blender_data['rotationZ']
                # pose_vector_c[head_x] = (blender_data['rotationX']-pose_vector_0[0]) * to_rad_m * 3
                # pose_vector_c[head_y] = -(blender_data['rotationY']-pose_vector_0[1]) * to_rad_m * 3
                # pose_vector_c[neck_z] = (blender_data['rotationZ']-pose_vector_0[2]) * to_rad_m
                # pose_vector_c[head_x] = (blender_data['rotationX']) * to_rad_m * 3
                # pose_vector_c[head_y] = -(blender_data['rotationY']) * to_rad_m * 3
                # pose_vector_c[neck_z] = (blender_data['rotationZ']) * to_rad_m * 2

                blender_data['rotationX'] -= rotationX_calibration_offset
                blender_data['rotationY'] -= rotationY_calibration_offset
                blender_data['rotationZ'] -= rotationZ_calibration_offset


                pose_vector_c[head_x] = rescale_01(
                    blender_data['rotationX'],
                    head_rotationX_limit_lower,
                    head_rotationX_limit_upper,
                )
                pose_vector_c[head_x] = pose_vector_c[head_x] * 2 - 1

                pose_vector_c[head_y] = rescale_01(
                    blender_data['rotationY'],
                    head_rotationY_limit_lower,
                    head_rotationY_limit_upper,
                )
                pose_vector_c[head_y] = pose_vector_c[head_y] * 2 - 1

                rotationZ = blender_data['rotationZ']
                if Head_add_glance_to_tilt:
                    rotationZ = min(rotationZ, rotationZ + blender_data['rotationY'])
                dd = 1.0
                dd = ((abs(blender_data['rotationY']) ** 2) * 20) / (head_glance_limit * head_glance_limit)
                dd = max(dd, 1.0)
                if Head_divide_tilt_by_rotation:
                    rotationZ = rotationZ / dd

                pose_vector_c[neck_z] = rescale_01(
                    rotationZ,
                    head_rotationZ_limit_lower,
                    head_rotationZ_limit_upper,
                )
                pose_vector_c[neck_z] = pose_vector_c[neck_z] * 2 - 1

                # print("pose_vector_c[neck_z]" ,
                #     print_number(pose_vector_c[neck_z]),
                #     " (",
                #     print_number(blender_data['rotationZ']),
                #     print_number(rotationZ),
                #     ")",
                #     "  dd=", print_number(dd),
                #     "  rotationY=", print_number(blender_data['rotationY']),
                #     "  head_y=", print_number(pose_vector_c[head_y]),
                # )


                # Not sure why, but 'rotationY' is the mirror of head_y
                pose_vector_c[head_y] = -pose_vector_c[head_y]

                # if position_vector_0==None:
                #     position_vector_0=[0,0,0,1]
                #     position_vector_0[0] = blender_data['translationX']
                #     position_vector_0[1] = blender_data['translationY']
                #     position_vector_0[2] = blender_data['translationZ']
                # position_vector[0] = -(blender_data['translationX']-position_vector_0[0])*0.1
                # position_vector[1] = -(blender_data['translationY']-position_vector_0[1])*0.1
                # position_vector[2] = -(blender_data['translationZ']-position_vector_0[2])*0.1

                # Automatic breathing. TODO: heart monitor
                breathing_adder += elapsed * (3.0 * surprised) # if 1.0 surprised, breathing rate will 1+3x
                pose_vector_c[breathing] = (math.sin(time_counter + breathing_adder) + 1) / 2
                # print("breathing", print_number(pose_vector_c[breathing]))

                pose_vector_c[head_x] = clamp(
                    pose_vector_c[head_x], -1.0, 1.0)
                pose_vector_c[head_y] = clamp(
                    pose_vector_c[head_y], -1.0, 1.0)
                pose_vector_c[neck_z] = clamp(
                    pose_vector_c[neck_z], -1.0, 1.0)

                if True:
                    # print("neck_z", print_number(pose_vector_c[neck_z]))
                    pose_vector_c[body_z] += pose_vector_c[neck_z] / 2

                if not args.osf_mouse_body:
                    pose_vector_c[body_y] = pose_vector_c[head_y]
                    pose_vector_c[body_z] = pose_vector_c[neck_z]
                else:
                    monitor_width_max = args.monitor_width_max
                    monitor_height_max = args.monitor_height_max
                    monitor_width_min = args.monitor_width_min
                    monitor_height_min = args.monitor_height_min

                    mouse_pos = mouse_controller.position

                    mouse_pos_x = monitor_width_max / 2
                    mouse_pos_y = monitor_height_max / 2
                    if mouse_pos is None:
                        if len(var_history['mouse_pos']) > 0:
                            var_history['mouse_pos'][-1]
                            mouse_pos_x = var_history['mouse_pos'][-1]['x']
                            mouse_pos_y = var_history['mouse_pos'][-1]['y']
                    else:
                        mouse_pos_x, mouse_pos_y = mouse_controller.position
                        # print("mouse_controller.position =", mouse_controller.position)

                    var_history['mouse_pos'].append({
                        'x': mouse_pos_x,
                        'y': mouse_pos_y,
                    })
                    map_01_to_neg1to1 = lambda x: (x * 2 - 1)
                    mouse_pos_x_prop = mouse_pos_x * (1 / monitor_width_max)
                    pose_vector_c[body_y] += map_01_to_neg1to1(mouse_pos_x_prop)
                    mouse_pos_y_prop = mouse_pos_y * (1 / monitor_height_max)
                    pose_vector_c[body_z] += map_01_to_neg1to1(mouse_pos_y_prop)

                    # WARNING: body_y is HORIZONTAL avatar movement, weirdly; body_z is VERTICAL
                    if args.osf_mouse_body == 'looking_right':
                        pose_vector_c[body_z] = -pose_vector_c[body_z]
                        # pose_vector_c[body_y] = -pose_vector_c[body_y]

                    if args.osf_mouse_mirror == 'axis':
                        pose_vector_c[body_z] *= -pose_vector_c[body_y]
                    elif args.osf_mouse_mirror == 'second_screen_coord':
                        # # `after_second_screen >= 1` when mouse is after second_screen_coord horizontally
                        # print("pose_vector_c[body_z]=\n", pose_vector_c[body_z])
                        after_second_screen = (mouse_pos_x_prop / (second_screen_coord[0]/monitor_width_max))
                        pose_vector_c[body_z] *= -(after_second_screen - 1)
                        # print("after_second_screen=\n", after_second_screen)
                        # print("-(after_second_screen - 1)=\n", -(after_second_screen - 1))
                        # print("pose_vector_c[body_z]=\n", pose_vector_c[body_z])


                var_history['blender_data'].append(blender_data)

                if args.camera_input_to_file:
                    # Write blender_data to a json file
                    filename = f"camera_input blender_data.json"
                    with open(filename, 'w') as f:
                        json.dump(blender_data, f, indent=4)

                if plotter_process:
                    plotter_process.input_queue.put_nowait({
                        'loop_counter': loop_counter,
                        'var': {
                            'time_counter_blender': time_counter,
                            'blender_data': blender_data,
                        }
                    })

                if False:
                    print(
                        # "[head_x]=", print_number(pose_vector_c[head_x]),
                        # "[head_y]=", print_number(pose_vector_c[head_y]),
                        # "[neck_z]=", print_number(pose_vector_c[neck_z]),
                        "[eyebrow_raised_left]=", print_number(eyebrow_vector_c[eyebrow_raised_left]),
                        "[eyebrow_lowered_left]=", print_number(eyebrow_vector_c[eyebrow_lowered_left]),
                        "[eyebrow_troubled_left]=", print_number(eyebrow_vector_c[eyebrow_troubled_left]),
                        "[eye_unimpressed_left]=", print_number(mouth_eye_vector_c[eye_unimpressed_left]),
                        "[eye_wink_current_left]=", print_number(mouth_eye_vector_c[eye_wink_current_left]),
                        "[mouth_uuu]=", print_number(mouth_eye_vector_c[mouth_uuu]),
                        "[mouth_aaa]=", print_number(mouth_eye_vector_c[mouth_aaa]),
                        "[mouth_smirk]=", print_number(mouth_eye_vector_c[mouth_smirk]),
                        "max_smile=", print_number(max_smile),
                    #     "blender_eye_open_right=", print_number(blender_eye_open_right), print_number(blender_data[rightEyeOpen]),
                    #     "eye_open_difference=", print_number(eye_open_difference),
                        "\n",
                        {
                            key: print_number(blender_data[key])
                            for key in [
                                'leftEyeOpen',
                                'rightEyeOpen',
                                leftEyeOpen,
                                rightEyeOpen,
                            ]
                        },
                        "\n",
                        {
                            key: print_number(blender_data[key])
                            for key in [
                                'eyeRotationY',
                                'eyeRotationX',

                                'rotationX',
                                'rotationY',
                                'rotationZ',
                            ]
                        },
                    #     "\n",
                    #     {
                    #         key: print_number(blender_data[key])
                    #         for key in [
                    #             'EyebrowUpDownLeft',
                    #             'EyebrowUpDownRight',
                    #             'EyebrowSteepnessLeft',
                    #             'EyebrowSteepnessRight',
                    #             'EyebrowQuirkLeft',
                    #             'EyebrowQuirkRight',
                    #         ]
                    #     },
                        "\n",
                        {
                            key: print_number(blender_data[key])
                            for key in [
                                # 'MouthCornerUpDownLeft',
                                # 'MouthCornerUpDownRight',
                                # 'MouthCornerInOutLeft',
                                # 'MouthCornerInOutRight',
                                'MouthOpen',
                                'MouthWide',
                            ]
                        },
                    #     "\n",
                    #     ";  ".join([
                    #         # f"[mouth_delta]={print_number(mouth_eye_vector_c[mouth_delta])}",
                    #         # f"[mouth_aaa]={print_number(mouth_eye_vector_c[mouth_aaa])}",
                    #         # f"[mouth_ooo]={print_number(mouth_eye_vector_c[mouth_ooo])}",
                    #         # f"[mouth_smirk]={print_number(mouth_eye_vector_c[mouth_smirk])}",
                    #         # f"eyebrows_raised={print_number(eyebrows_raised)}",
                    #         #
                    #         f"[iris_rotation_x]={print_number(mouth_eye_vector_c[iris_rotation_x])}",
                    #         f"[iris_rotation_y]={print_number(mouth_eye_vector_c[iris_rotation_y])}",
                    #     ]),
                    #     "\n",
                    )

            elif args.ifm is not None:
                # get pose from ifm
                try:
                    new_blender_data = blender_data
                    while not client_process.should_terminate.value and not client_process.queue.empty():
                        new_blender_data = client_process.queue.get_nowait()
                    blender_data = new_blender_data
                except queue.Empty:
                    pass

                ifacialmocap_pose_converted = ifm_converter.convert(blender_data)

                # ifacialmocap_pose = blender_data
                #
                # eye_l_h_temp = ifacialmocap_pose[EYE_BLINK_LEFT]
                # eye_r_h_temp = ifacialmocap_pose[EYE_BLINK_RIGHT]
                # mouth_ratio = (ifacialmocap_pose[JAW_OPEN] - 0.10)*1.3
                # x_angle = -ifacialmocap_pose[HEAD_BONE_X] * 1.5 + 1.57
                # y_angle = -ifacialmocap_pose[HEAD_BONE_Y]
                # z_angle = ifacialmocap_pose[HEAD_BONE_Z] - 1.57
                #
                # eye_x_ratio = (ifacialmocap_pose[EYE_LOOK_IN_LEFT] -
                #                ifacialmocap_pose[EYE_LOOK_OUT_LEFT] -
                #                ifacialmocap_pose[EYE_LOOK_IN_RIGHT] +
                #                ifacialmocap_pose[EYE_LOOK_OUT_RIGHT]) / 2.0 / 0.75
                #
                # eye_y_ratio = (ifacialmocap_pose[EYE_LOOK_UP_LEFT]
                #                + ifacialmocap_pose[EYE_LOOK_UP_RIGHT]
                #                - ifacialmocap_pose[EYE_LOOK_DOWN_RIGHT]
                #                + ifacialmocap_pose[EYE_LOOK_DOWN_LEFT]) / 2.0 / 0.75

                eyebrow_vector_c = [0.0] * 12
                mouth_eye_vector_c = [0.0] * 27
                pose_vector_c = [0.0] * 6
                for i in range(0, 12):
                    eyebrow_vector_c[i] = ifacialmocap_pose_converted[i]
                for i in range(12, 39):
                    mouth_eye_vector_c[i - 12] = ifacialmocap_pose_converted[i]
                for i in range(39, 42):
                    pose_vector_c[i - 39] = ifacialmocap_pose_converted[i]

                position_vector = blender_data[HEAD_BONE_QUAT]

            elif args.mouse_input is not None:

                try:
                    new_blender_data = mouse_data
                    while not mouse_client_process.queue.empty():
                        new_blender_data = mouse_client_process.queue.get_nowait()
                    mouse_data = new_blender_data
                except queue.Empty:
                    pass

                eye_l_h_temp = mouse_data['eye_l_h_temp']
                eye_r_h_temp = mouse_data['eye_r_h_temp']
                mouth_ratio = mouse_data['mouth_ratio']
                eye_y_ratio = mouse_data['eye_y_ratio']
                eye_x_ratio = mouse_data['eye_x_ratio']
                x_angle = mouse_data['x_angle']
                y_angle = mouse_data['y_angle']
                z_angle = mouse_data['z_angle']

                eyebrow_vector_c = [0.0] * 12
                mouth_eye_vector_c = [0.0] * 27
                pose_vector_c = [0.0] * 6

                mouth_eye_vector_c[2] = eye_l_h_temp
                mouth_eye_vector_c[3] = eye_r_h_temp

                mouth_eye_vector_c[14] = mouth_ratio * 1.5

                mouth_eye_vector_c[25] = eye_y_ratio
                mouth_eye_vector_c[26] = eye_x_ratio

                pose_vector_c[0] = x_angle
                pose_vector_c[1] = y_angle
                pose_vector_c[2] = z_angle

            elif args.input == 'auto':
                try:
                    arr = mm.update(time_counter)
                except Exception as e:
                    logger.exception(f"Exception on mm.update")
                vecs = model_input_split(arr, time_counter)
                eyebrow_vector_c = vecs['eyebrow_vector_c']
                mouth_eye_vector_c = vecs['mouth_eye_vector_c']
                pose_vector_c = vecs['pose_vector_c']

            else:
                # args.input == 'cam'
                ret, frame = cap.read()

                input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = facemesh.process(input_frame)

                if results.multi_face_landmarks is None:
                    continue

                facial_landmarks = results.multi_face_landmarks[0].landmark

                if args.debug:
                    pose, debug_image = get_pose(facial_landmarks, frame)
                else:
                    pose = get_pose(facial_landmarks)

                if len(pose_queue) < 3:
                    pose_queue.append(pose)
                    pose_queue.append(pose)
                    pose_queue.append(pose)
                else:
                    pose_queue.pop(0)
                    pose_queue.append(pose)

                np_pose = np.average(np.array(pose_queue), axis=0, weights=[0.6, 0.3, 0.1])

                eye_l_h_temp = np_pose[0]
                eye_r_h_temp = np_pose[1]
                mouth_ratio = np_pose[2]
                eye_y_ratio = np_pose[3]
                eye_x_ratio = np_pose[4]
                x_angle = np_pose[5]
                y_angle = np_pose[6]
                z_angle = np_pose[7]

                eyebrow_vector_c = [0.0] * 12
                mouth_eye_vector_c = [0.0] * 27
                pose_vector_c = [0.0] * 6

                mouth_eye_vector_c[2] = eye_l_h_temp
                mouth_eye_vector_c[3] = eye_r_h_temp

                mouth_eye_vector_c[14] = mouth_ratio * 1.5

                mouth_eye_vector_c[25] = eye_y_ratio
                mouth_eye_vector_c[26] = eye_x_ratio

                pose_vector_c[0] = (x_angle - 1.5) * 1.6
                pose_vector_c[1] = y_angle * 2.0  # temp weight
                pose_vector_c[2] = (z_angle + 1.5) * 2  # temp weight

        except Exception as e:
            logging.exception(e)
            continue

        # This wasn't commented
        # it complete broke 'body_y' and 'body_z' by overwriting them...
        # maybe it was here for a reason, who knows
        # pose_vector_c[3] = pose_vector_c[1]
        # pose_vector_c[4] = pose_vector_c[2]

        smooth = args.exponential_smoothing_eye_rotation
        if smooth > 0 and smooth < 1:
            if mouth_eye_vector_c_avg is None:
                mouth_eye_vector_c_avg = mouth_eye_vector_c
            else:
                mouth_eye_vector_c_avg[iris_rotation_x] = (
                    ((smooth) * mouth_eye_vector_c_avg[iris_rotation_x]) +
                    ((1-smooth) * mouth_eye_vector_c[iris_rotation_x])
                )
                mouth_eye_vector_c[iris_rotation_x] = mouth_eye_vector_c_avg[iris_rotation_x]

                mouth_eye_vector_c_avg[iris_rotation_y] = (
                    ((smooth) * mouth_eye_vector_c_avg[iris_rotation_y]) +
                    ((1-smooth) * mouth_eye_vector_c[iris_rotation_y])
                )
                mouth_eye_vector_c[iris_rotation_y] = mouth_eye_vector_c_avg[iris_rotation_y]


        model_input_arr = [
            *eyebrow_vector_c,
            *mouth_eye_vector_c,
            *pose_vector_c,
        ]

        if 'main' in args.perf:
            inspect_frame = inspect.getframeinfo(inspect.currentframe())
            print(f"\t\t main: model_input_arr   \n\t\t elapsed = {(time.perf_counter() - tic) * 1000:>6.2g}   {Path(inspect_frame.filename).as_posix()}:{inspect_frame.lineno}")
            tic = time.perf_counter()

        smooth = args.exponential_smoothing
        if smooth > 0 and smooth < 1:
            if model_input_arr_avg is None:
                model_input_arr_avg = model_input_arr
            else:
                model_input_arr_avg = (
                    ((smooth) * np.array(model_input_arr_avg)) +
                    ((1-smooth) * np.array(model_input_arr))
                )

                model_input_arr = model_input_arr_avg

        # smooth = args.exponential_smoothing_eye_rotation
        # if smooth > 0 and smooth < 1:
        #     print("exponential_smoothing_eye_rotation", smooth)
        #     if model_input_arr_avg is None:
        #         model_input_arr_avg = model_input_arr
        #     else:
        #         x_idx = len(eyebrow_vector_c) + iris_rotation_x
        #         model_input_arr_avg[x_idx] = (
        #             ((smooth) * model_input_arr_avg[x_idx]) +
        #             ((1-smooth) * model_input_arr[x_idx])
        #         )
        #         y_idx = len(eyebrow_vector_c) + iris_rotation_y
        #         model_input_arr_avg[y_idx] = (
        #             ((smooth) * model_input_arr_avg[y_idx]) +
        #             ((1-smooth) * model_input_arr[y_idx])
        #         )
        #         model_input_arr = model_input_arr_avg


        if args.plot_params:
            import matplotlib.pyplot as plt
            var_history['time_counter'].append(time_counter)
            var_history['model_input_arr'].append(model_input_arr)

            plt_h_scale = 1
            plt_config = {
                'interval': 60 * 6 * plt_h_scale,
                'hist_size': 60 * 6 * plt_h_scale,
            }
            if loop_counter > 1 and (loop_counter % plt_config['interval']) == 0:
                print(f"Plot movement parameters {loop_counter}")
                # Clear axes and re-plot
                plt.clf()
                for i in range(len(var_history['model_input_arr'][0])):
                    plt.plot(var_history["time_counter"][-plt_config['hist_size']:],
                        [
                            v[i]
                            for v in var_history['model_input_arr'][-plt_config['hist_size']:]
                        ],
                        label=model_input_arr_names[i]
                    )
                plt.tight_layout()
                plt.legend(loc=2, prop={'size': 3})
                plt.ylim(-1.0, 1.0) # Add y-axis limits
                plt.savefig("plt_movement_parameters.pdf")


            if 'main' in args.perf:
                inspect_frame = inspect.getframeinfo(inspect.currentframe())
                print(f"\t\t main: args.plot_params   \n\t\t elapsed = {(time.perf_counter() - tic) * 1000:>6.2g}   {Path(inspect_frame.filename).as_posix()}:{inspect_frame.lineno}")
                tic = time.perf_counter()

        # input_image_current_idx = input_image_name_to_idx[input_image_current_name]
        # model_process.input_image_current_idx.value = input_image_current_idx
        if 'main' in args.perf:
            inspect_frame = inspect.getframeinfo(inspect.currentframe())
            print(f"\t\t main: model_process.input_queue.put_nowait()   \n\t\t elapsed = {(time.perf_counter() - tic) * 1000:>6.2g}   {Path(inspect_frame.filename).as_posix()}:{inspect_frame.lineno}")
            tic = time.perf_counter()
        model_process.input_queue.put_nowait(model_input_arr)


        for k in var_history.keys():
            var_history[k] = var_history[k][-var_history_max:]


        nowaits = 0
        if not model_output_first:
            try:
                model_output = model_process.output_queue.get()
                while not model_process.output_queue.empty():
                    nowaits += 1
                    model_output = model_process.output_queue.get_nowait()
            except queue.Empty:
                pass
        model_output_first = False
        if model_output is None:
            continue

        postprocessed_image = model_output

        if 'main' in args.perf:
            inspect_frame = inspect.getframeinfo(inspect.currentframe())
            print(f"\t\t main: got model_output   \n\t\t elapsed = {(time.perf_counter() - tic) * 1000:>6.2g}   {Path(inspect_frame.filename).as_posix()}:{inspect_frame.lineno}", f"nowaits={nowaits}")
            tic = time.perf_counter()

        if extra_image is not None:
            postprocessed_image = cv2.vconcat([postprocessed_image, extra_image])

        k_scale = 1
        rotate_angle = 0
        dx = 0
        dy = 0
        if args.extend_movement:
            k_scale = position_vector[2] * math.sqrt(args.extend_movement) + 1
            rotate_angle = -position_vector[0] * 10 * args.extend_movement
            dx = position_vector[0] * 400 * k_scale * args.extend_movement
            dy = -position_vector[1] * 600 * k_scale * args.extend_movement
        if args.bongo:
            rotate_angle -= 5
        rm = cv2.getRotationMatrix2D((INPUT_RESOLUTION / 2, INPUT_RESOLUTION / 2), rotate_angle, k_scale)
        rm[0, 2] += dx + args.output_w / 2 - INPUT_RESOLUTION / 2
        rm[1, 2] += dy + args.output_h / 2 - INPUT_RESOLUTION / 2

        postprocessed_image = cv2.warpAffine(
            postprocessed_image,
            rm,
            (args.output_w, args.output_h))

        if 'main' in args.perf:
            inspect_frame = inspect.getframeinfo(inspect.currentframe())
            print(f"\t\t main: extendmovement   \n\t\t elapsed = {(time.perf_counter() - tic) * 1000:>6.2g}   {Path(inspect_frame.filename).as_posix()}:{inspect_frame.lineno}")
            tic = time.perf_counter()

        output_fps_number = output_fps()

        if args.anime4k:
            alpha_channel = postprocessed_image[:, :, 3]
            alpha_channel = cv2.resize(alpha_channel, None, fx=2, fy=2)

            # a.load_image_from_numpy(cv2.cvtColor(postprocessed_image, cv2.COLOR_RGBA2RGB), input_type=ac.AC_INPUT_RGB)
            # img = cv2.imread("character/test41.png")
            img1 = cv2.cvtColor(postprocessed_image, cv2.COLOR_RGBA2BGR)
            # a.load_image_from_numpy(img, input_type=ac.AC_INPUT_BGR)
            a.load_image_from_numpy(img1, input_type=ac.AC_INPUT_BGR)
            a.process()
            postprocessed_image = a.save_image_to_numpy()
            postprocessed_image = cv2.merge((postprocessed_image, alpha_channel))
            postprocessed_image = cv2.cvtColor(postprocessed_image, cv2.COLOR_BGRA2RGBA)

            if 'main' in args.perf:
                inspect_frame = inspect.getframeinfo(inspect.currentframe())
                print(f"\t\t main: args.anime4k   \n\t\t elapsed = {(time.perf_counter() - tic) * 1000:>6.2g}   {Path(inspect_frame.filename).as_posix()}:{inspect_frame.lineno}")
                tic = time.perf_counter()

        if args.alpha_split:
            alpha_image = cv2.merge(
                [postprocessed_image[:, :, 3], postprocessed_image[:, :, 3], postprocessed_image[:, :, 3]])
            alpha_image = cv2.cvtColor(alpha_image, cv2.COLOR_RGB2RGBA)
            postprocessed_image = cv2.hconcat([postprocessed_image, alpha_image])

            if 'main' in args.perf:
                inspect_frame = inspect.getframeinfo(inspect.currentframe())
                print(f"\t\t main: args.alpha_split   \n\t\t elapsed = {(time.perf_counter() - tic) * 1000:>6.2g}   {Path(inspect_frame.filename).as_posix()}:{inspect_frame.lineno}")
                tic = time.perf_counter()

        if args.debug:
            output_frame = cv2.cvtColor(postprocessed_image, cv2.COLOR_RGBA2BGRA)
            # resized_frame = cv2.resize(output_frame, (np.min(debug_image.shape[:2]), np.min(debug_image.shape[:2])))
            # output_frame = np.concatenate([debug_image, resized_frame], axis=1)
            cv2.putText(output_frame, str('OUT_FPS:%.1f' % output_fps_number), (0, 16), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 255, 0), 1)
            if args.max_cache_len > 0:
                cv2.putText(output_frame, str(
                    'GPU_FPS:%.1f / %.1f' % (model_process.model_fps_number.value, model_process.gpu_fps_number.value)),
                            (0, 32),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            else:
                cv2.putText(output_frame, str(
                    'GPU_FPS:%.1f' % (model_process.model_fps_number.value)),
                            (0, 32),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            if args.ifm is not None:
                cv2.putText(output_frame, str('IFM_FPS:%.1f' % client_process.ifm_fps_number.value), (0, 48),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            if args.max_cache_len > 0:
                cv2.putText(output_frame, str('MEMCACHED:%.1f%%' % (model_process.cache_hit_ratio.value * 100)),
                            (0, 64),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            if args.max_gpu_cache_len > 0:
                cv2.putText(output_frame, str('GPUCACHED:%.1f%%' % (model_process.gpu_cache_hit_ratio.value * 100)),
                            (0, 80),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            cv2.imshow("frame", output_frame)
            # cv2.imshow("camera", debug_image)
            cv2.waitKey(1)

            if 'main' in args.perf:
                inspect_frame = inspect.getframeinfo(inspect.currentframe())
                print(f"\t\t main: args.debug   \n\t\t elapsed = {(time.perf_counter() - tic) * 1000:>6.2g}   {Path(inspect_frame.filename).as_posix()}:{inspect_frame.lineno}")
                tic = time.perf_counter()

        if args.output_webcam:
            # result_image = np.zeros([720, 1280, 3], dtype=np.uint8)
            # result_image[720 - 512:, 1280 // 2 - 256:1280 // 2 + 256] = cv2.resize(
            #     cv2.cvtColor(postprocessing_image(output_image.cpu()), cv2.COLOR_RGBA2RGB), (512, 512))
            result_image = postprocessed_image
            if args.output_webcam == 'obs':
                result_image = cv2.cvtColor(result_image, cv2.COLOR_RGBA2RGB)
            cam.send(result_image)
            cam.sleep_until_next_frame()

        if 'main' in args.perf:
            inspect_frame = inspect.getframeinfo(inspect.currentframe())
            print(f"\t\t main: output   \n\t\t elapsed = {(time.perf_counter() - tic) * 1000:>6.2g}   {Path(inspect_frame.filename).as_posix()}:{inspect_frame.lineno}")
            tic = time.perf_counter()

        loop_counter += 1


if __name__ == '__main__':
    main()
