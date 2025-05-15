import argparse
import re

def convert_to_byte(size):
    result = re.search('(\d+\.?\d*)(b|kb|mb|gb|tb)', size.lower())
    if (result and result.groups()):
        unit = result.groups()[1]
        amount = float(result.groups()[0])
        index = ['b', 'kb', 'mb', 'gb', 'tb'].index(unit)
        return amount * pow(1024, index)
    raise ValueError("Invalid size provided, value is " + size)

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--eyebrow', action='store_true')
parser.add_argument('--extend_movement', type=float)
parser.add_argument('--input', type=str, default='cam')
parser.add_argument('--character', type=str, default='y')
parser.add_argument('--output_dir', type=str)
parser.add_argument('--output_webcam', type=str)
parser.add_argument('--output_size', type=str, default='512x512')
parser.add_argument('--model', type=str, default='standard_float')
parser.add_argument('--debug_input', action='store_true')
parser.add_argument('--mouse_input', type=str)
parser.add_argument(
    '--perf',
    type=list[str],
    default=[
        # 'model',
        # 'main',
        # 'plotter',
    ],
)
parser.add_argument('--skip_model', action='store_true')
parser.add_argument('--ifm', type=str)
parser.add_argument('--osf', type=str)
parser.add_argument('--mediapipe', type=str)
parser.add_argument('--anime4k', action='store_true')
parser.add_argument('--alpha_split', action='store_true')
parser.add_argument('--bongo', action='store_true')
parser.add_argument('--cache', type=str, default='256mb')
parser.add_argument('--gpu_cache', type=str, default='512mb')
parser.add_argument('--simplify', type=int, default=1)


# Plotting
parser.add_argument('--camera_input_to_file', type=int, default=0)
parser.add_argument('--plot_params', type=bool, default=False)
parser.add_argument('--plot_params_capture', type=bool, default=False)
# parser.add_argument('--plot_params', type=bool, default=True)
# parser.add_argument('--plot_params_capture', type=bool, default=True)

# CALIBRATION VARS
# `eyeRotationY_camera_offset`
# `head_rotationZ_limit_lower`
# `head_rotationZ_limit_upper`
# `head_rotationX_limit_lower`
# `head_rotationX_limit_upper`
# `head_rotationY_limit_lower`
# `head_rotationY_limit_upper`

# OSF
parser.add_argument('--exponential_smoothing', type=float, default=0.15)
parser.add_argument('--exponential_smoothing_eye_rotation', type=float, default=0.75) # 0.96 = ~1 second on 30 fps (1-1/30)
parser.add_argument('--eyebrows_sync', type=str, default='cube') # 'cube', 'set_max', 'set_max_avg', or ''
parser.add_argument('--eye_open_sync', type=str, default='apart_or_close')
parser.add_argument('--osf_mouse_body', type=str, default='looking_right') # 'looking_right', 'looking_left'
parser.add_argument('--osf_mouse_mirror', type=str, default='second_screen_coord') # 'axis', 'second_screen_coord'
# parser.add_argument('--monitor_width_max', type=int, default=3840)
# parser.add_argument('--monitor_height_max', type=int, default=1272)
parser.add_argument('--monitor_width_max', type=int, default=5759)
parser.add_argument('--monitor_height_max', type=int, default=1079)
parser.add_argument('--monitor_width_min', type=int, default=0)
parser.add_argument('--monitor_height_min', type=int, default=-177)

parser.add_argument('--second_screen_coord', type=str, default='3840,0')

args = parser.parse_args()
args.output_w = int(args.output_size.split('x')[0])
args.output_h = int(args.output_size.split('x')[1])
if args.cache is not None:
    args.max_cache_len=int(convert_to_byte(args.cache)/262144/4)
else:
    args.max_cache_len=0
if args.gpu_cache is not None:
    args.max_gpu_cache_len=int(convert_to_byte(args.gpu_cache)/589824/4)
else:
    args.max_gpu_cache_len=0
if args.output_webcam is None and args.output_dir is None: args.debug = True