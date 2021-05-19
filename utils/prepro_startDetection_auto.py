from pyAudioAnalysis import MidTermFeatures as aFm
from pyAudioAnalysis import audioBasicIO as aIO
import moviepy.editor as mp
import numpy as np
import argparse
import json


def get_index(list_dict, vid_name):
    """helper to read the json file."""
    for i in range(len(list_dict)):
        if list_dict['videos'][i]['name'] == vid_name:
            return i


def extract_time_start(video_path, bip_ref_path="ref_bip_isolated.wav"):
    # features of the ref
    # extract short-term features using a 50msec non-overlapping windows
    fs, s_ref = aIO.read_audio_file(bip_ref_path)
    duration = len(s_ref) / float(fs)
    win, step = 0.05, 0.05
    win_mid, step_mid = duration, 0.5
    mt_ref, st_ref, mt_n_ref = aFm.mid_feature_extraction(s_ref, fs, win_mid * fs, step_mid * fs,
                                                          win * fs, step * fs)
    # extraction on the long signal
    my_clip1 = mp.VideoFileClip(video_path)
    fs = 44100
    s_long = my_clip1.audio.to_soundarray(fps=fs)
    s_long = s_long[:, 0]
    duration_long = len(s_long) / float(fs)

    # extract short-term features using a 50msec non-overlapping windows
    win, step = 0.05, 0.05
    win_mid, step_mid = 0.4, 0.05
    mt_long, st_long, mt_n_long = aFm.mid_feature_extraction(s_long, fs, win_mid * fs, step_mid * fs,
                                                             win * fs, step * fs)

    # compute the distance and get the minimum
    distances = np.linalg.norm(mt_long - mt_ref, axis=0)
    time_start = np.argmin(distances) * duration_long / mt_long.shape[1]
    return time_start


if __name__ == "__main__":
    # print(extract_time_start('videos/50_brasse_stevens.mp4'))
    parser = argparse.ArgumentParser(description='Parser for automatic start detection.')
    parser.add_argument('--json', help='Path of the json of the race')
    parser.add_argument('--video', help='Path of the video')
    parser.add_argument('--ref', help='Reference to use as the noise for the start.', default='ref_bip_isolated.wav')
    args = parser.parse_args()
    start_time = extract_time_start(args.video, bip_ref_path=args.ref)

    with open(args.json) as json_file:
        json_course = json.load(json_file)

    name_of_video = args.video.split('/')[-1]
    index_vid = get_index(json_course, name_of_video)
    json_course['videos'][index_vid]['start_moment'] = start_time

    with open(args.json, 'w') as outfile:
        json.dump(json_course, outfile, indent=4)
    # Todo reflechir quels arguments on met : plutot race, json path, video path ou folder, race, video
