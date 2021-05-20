import os
import cv2
import pandas as pd
import numpy as np


def get_framename(video_path, repo) :
    file_name = video_path.split('/')[-1]
    raw_name = file_name.split('.')[0]
    unique_name = raw_name #repo + '_' + raw_name
    return unique_name


def save_frames_rate(video_path, frames_per_seconds, frames_save_path, repo, one_directory_per_video, csv_path, start=0, end=0):
    all_labels = extract_and_linearise_csv(csv_path)
    labels_to_keep = []

    cap = cv2.VideoCapture(video_path)
    frame_nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_max = frame_nb - end*fps

    if frames_per_seconds != 0:
        frames_rate = int(fps*frames_per_seconds) # int(fps / frames_per_seconds)
    else: frames_rate = 1

    not_end_of_video, current_frame = cap.read()

    generic_frame_name = get_framename(video_path, repo)
    if one_directory_per_video: frames_save_path = os.path.join(frames_save_path, generic_frame_name)

    if not os.path.isdir(frames_save_path): os.mkdir(frames_save_path)

    frame_nb = 0

    while not_end_of_video:
        if frame_nb >= start*fps:
            if frame_nb % frames_rate == frames_rate-1 :
                # current_frame = cv2.resize(current_frame, (1600, 900))
                frame_name = generic_frame_name+'_'+str(frame_nb) + '.jpg'
                frame_path = os.path.join(frames_save_path, frame_name)
                cv2.imwrite(frame_path, current_frame)
                labels_frame = [generic_frame_name+'_'+str(frame_nb)+ '.jpg'] + [tuple(elmt[frame_nb]) for elmt in all_labels]
                labels_to_keep.append(labels_frame)

                # for test and visualisation
                test_image = current_frame
                for elmt in all_labels:
                    test_image = cv2.circle(test_image, tuple(elmt[frame_nb]), 2, (255, 0, 0), -1)
                cv2.imwrite(os.path.join('./new_images/test', frame_name), test_image)


        not_end_of_video, current_frame = cap.read()
        frame_nb += 1
        if frame_nb == frame_max: break
    pd.DataFrame(np.array(labels_to_keep)).to_csv("./new_images/label/" + generic_frame_name + ".csv", header=None, index=None)

def extract_and_linearise_csv(csv_path):
    directory = os.path.join(csv_path)
    all_swimmers = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                swimmer = pd.read_csv(os.path.join(csv_path,file))
                swimmer = swimmer.set_index('Indice')
                swimmer = swimmer.reindex(range(0, max(swimmer.index) + 1))
                swimmer.loc[0] = 0
                # print(swimmer)
                data_to_print = swimmer[['Distance (Pix) x', 'Distance (Pix) y']].interpolate(method='index')
                data_to_print['Distance (Pix) x'] = data_to_print['Distance (Pix) x'].astype(int)
                data_to_print['Distance (Pix) y'] = data_to_print['Distance (Pix) y'].astype(int)
                data_to_print = data_to_print.to_numpy()
                # print(data_to_print.shape)
                all_swimmers.append(data_to_print)
    print(np.array(all_swimmers)[0].shape)
    return np.array(all_swimmers)


# first : every frames in one repo
if __name__=='__main__' :

    # for 1, created 1070 frames for 17 videos
    frames_per_seconds = 4 # 0 for every frames, N > 0 for N frames per second
    every_videos_path = './excel_david/vid'
    frames_save_path = './new_images/images'
    csv_path = './excel_david/100NL_FAH'
    one_directory_per_video = False
    delete_folder = True

    # extract_and_linearise_csv(csv_path)
    if delete_folder:
        os.system('rm ' + frames_save_path + "/*")
        os.system("rm ./new_images/test/*")

    if not os.path.isdir(frames_save_path): os.mkdir(frames_save_path)

    for root, dirs, files in os.walk(every_videos_path) :
        dirs.sort()
        for file in files :
            video_path = os.path.join(root, file)
            repo = root.split('/')[-1]
            print(video_path)
            save_frames_rate(video_path, frames_per_seconds, frames_save_path, repo,
                             one_directory_per_video, csv_path, start=10, end=3)