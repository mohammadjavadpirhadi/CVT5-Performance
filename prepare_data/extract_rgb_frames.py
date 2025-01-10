import os
import math
import glob
import argparse
import cv2
from tqdm.auto import tqdm

import decord
decord.bridge.set_bridge('native')

parser = argparse.ArgumentParser()

parser.add_argument('--base_dir',
                    required=True,type=str,
                    help='path to base directory containing data folder')

args = parser.parse_args()

video_pathes = glob.glob(f"{args.base_dir}/data/*/*/*/*224p_h264.mp4")
for video_path in tqdm(video_pathes):
    output_dir = f"{video_path.rsplit('.', 1)[0]}_rgb"
    os.makedirs(output_dir, exist_ok=True)
    decord_reader = decord.VideoReader(video_path)
    num_digits = math.ceil(math.log10(len(decord_reader)))
    for frame_index, frame in tqdm(enumerate(decord_reader), total=len(decord_reader), position=1, leave=False):
        frame_path = f"{output_dir}/{str(frame_index).rjust(num_digits, '0')}.jpg"
        cv2.imwrite(frame_path, cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR))
