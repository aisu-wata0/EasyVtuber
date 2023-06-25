states = ['eyebrow_troubled_left', 'eyebrow_troubled_right', 'eyebrow_angry_left', 'eyebrow_angry_right', 'eyebrow_lowered_left', 'eyebrow_lowered_right', 'eyebrow_raised_left', 'eyebrow_raised_right', 'eyebrow_happy_left', 'eyebrow_happy_right', 'eyebrow_serious_left', 'eyebrow_serious_right', 'eye_wink_left', 'eye_wink_right', 'eye_happy_wink_left', 'eye_happy_wink_right', 'eye_surprised_left', 'eye_surprised_right', 'eye_relaxed_left', 'eye_relaxed_right', 'eye_unimpressed_left', 'eye_unimpressed_right', 'eye_raised_lower_eyelid_left', 'eye_raised_lower_eyelid_right', 'iris_small_left', 'iris_small_right', 'mouth_aaa', 'mouth_iii', 'mouth_uuu', 'mouth_eee', 'mouth_ooo', 'mouth_delta', 'mouth_lowered_corner_left', 'mouth_lowered_corner_right', 'mouth_raised_corner_left', 'mouth_raised_corner_right', 'mouth_smirk', 'iris_rotation_x', 'iris_rotation_y', 'head_x', 'head_y', 'neck_z', 'body_y', 'body_z', 'breathing']


from collections import defaultdict
from dataclasses import dataclass, field

import struct
from typing import Any, Dict, List, Optional, Set
from abc import ABC, abstractmethod

from matplotlib import markers

from AnimationStates.animation import BezierCurveCubic

import time
timers = {}

def logTime(name, elapsed):
    if name not in timers:
        timers[name] = []
    timers[name].append(elapsed)
    # print(f"elapsed {elapsed}  {name}")


def timerPrint(keys=None):
    if keys is None:
        keys = timers.keys()
    for k in keys:
        print(k)
        print(np.mean(timers[k]))
        print(np.std(timers[k]))


# from AnimationStates.animation import AnimationStates, CurveT, Animation, Transition
# from AnimationsTha.animations import Animation_idle_eye_glance_random_tha
# from python_utils_aisu.utils import Cooldown, CooldownVarU


# c = CurveT.build(**{'name': 'linear_delays'})

# a = Animation_idle_eye_glance_random_tha(interval={'time': 12})
# print(a.interval)

# Transition.build({'name': 'linear'}, {'time': 0})

# print((CooldownVarU(4, variance=5) * 4).seconds)


# from config_auto import getKwargs
# mm = AnimationStates(**getKwargs())


# import matplotlib.pyplot as plt

# presets = {
#     "ease"       : ".25,.1,.25,1",
#     "linear"     : "0,0,1,1",
#     "ease-in"    : ".42,0,1,1",
#     "ease-out"   : "0,0,.58,1",
#     "ease-in-out": ".42,0,.58,1",
#     "snek": "0.99,0.0,0.0,0.99",
# }

# for name, args in presets.items():
#     args = list(map(float, args.split(",")))
#     print(args)
#     curve = cubic_bezier_css(*args)
#     xs = [float(i) / 100 for i in range(100)]
#     ys = [curve(x) for x in xs]

#     plt.plot(xs, ys, label=name)

# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Cubic Bezier Curves')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# resolution = 0.01

# x = [0, 1, 0, 1]
# y = [0, 0, 1, 1]

# bcc = BezierCurveCubic(x, y, resolution)

# # Plotting the curve
# plt.plot(np.arange(0.0, 1.0, resolution), [bcc.y(x) for x in np.arange(0.0, 1.0, resolution)], label='precalc')
# plt.plot(np.arange(0.0, 1.0, resolution), [bcc.y_i(x) for x in np.arange(0.0, 1.0, resolution)], label='precalc i')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.tight_layout()
# plt.show()
