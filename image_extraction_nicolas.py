import os
import cv2


def get_framename(video_path, repo) :
    file_name = video_path.split('/')[-1]
    raw_name = file_name.split('.')[0]
    unique_name = repo + '_' + raw_name
    return unique_name


def save_frames_rate(video_path, frames_per_seconds, frames_save_path, repo, one_directory_per_video, start=0, end=0):
    cap = cv2.VideoCapture(video_path)
    frame_nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_max = frame_nb - end*fps

    if frames_per_seconds != 0 :
        frames_rate = int(fps / frames_per_seconds)
    else : frames_rate = 1

    not_end_of_video, current_frame = cap.read()

    generic_frame_name = get_framename(video_path, repo)
    if one_directory_per_video : frames_save_path = os.path.join(frames_save_path, generic_frame_name)

    if not os.path.isdir(frames_save_path): os.mkdir(frames_save_path)
    frame_nb = 0

    while not_end_of_video :
        if frame_nb >= start*fps  :
            if frame_nb % frames_rate == frames_rate-1 :
                current_frame = cv2.resize(current_frame, (1600, 900))
                frame_name = generic_frame_name+'_'+str(frame_nb) + '.jpg'
                frame_path = os.path.join(frames_save_path, frame_name)
                cv2.imwrite(frame_path, current_frame)
                # print(frame_path)
        not_end_of_video, current_frame = cap.read()
        frame_nb += 1
        if frame_nb == frame_max : break


# first : every frames in one repo
if __name__=='__main__' :

    frames_per_seconds = 0 # 0 for every frames, N > 0 for N frames per second
    every_videos_path = '/home/amigo/Bureau/data/saint-raph/deuxieme_reunion/50_dos_dames_serie1_f122020/vid'
    frames_save_path = '/home/amigo/Bureau/data/saint-raph/deuxieme_reunion/50_dos_dames_serie1_f122020/image'
    one_directory_per_video = False

    if not os.path.isdir(frames_save_path) : os.mkdir(frames_save_path)

    for root, dirs, files in os.walk(every_videos_path) :
        dirs.sort()
        for file in files :
            video_path = os.path.join(root, file)
            repo = root.split('/')[-1]
            print(video_path)
            save_frames_rate(video_path, frames_per_seconds, frames_save_path, repo, one_directory_per_video, start=4, end=4)
