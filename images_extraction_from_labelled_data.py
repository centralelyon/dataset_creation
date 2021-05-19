import os
import cv2


def get_framename(video_path, repo):
    file_name = video_path.split('/')[-1]
    raw_name = file_name.split('.')[0]
    unique_name = raw_name #repo + '_' + raw_name
    return unique_name


def save_labelled_frames(video_path, labels, frames_save_path, repo, one_directory_per_video, start=0, end=0):

    cap = cv2.VideoCapture(video_path)
    frame_nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_max = frame_nb - end*fps

    not_end_of_video, current_frame = cap.read()

    generic_frame_name = get_framename(video_path, repo)
    if one_directory_per_video:
        frames_save_path = os.path.join(frames_save_path, generic_frame_name)

    if not os.path.isdir(frames_save_path):
        os.mkdir(frames_save_path)

    frame_nb = 0

    while not_end_of_video:
        if frame_nb >= start*fps:
            frame_name_radical = generic_frame_name + '_' + str(frame_nb)
            if frame_name_radical in labels :
                # current_frame = cv2.resize(current_frame, (1600, 900))
                frame_name = generic_frame_name+'_'+str(frame_nb) + '.jpg'
                frame_path = os.path.join(frames_save_path, frame_name)
                cv2.imwrite(frame_path, current_frame)

        not_end_of_video, current_frame = cap.read()
        frame_nb += 1
        if frame_nb == frame_max: break


if __name__=='__main__' :

    every_videos_path = '/home/amigo/Bureau/data/video_for_extracting/videos_raw_compressed'
    frames_save_path = '/home/amigo/Bureau/data/video_for_extracting/images_from_label'
    labels_path = '/home/amigo/Bureau/data/video_for_extracting/label'

    all_labels = [elmt.split('.')[0] for elmt in os.listdir(labels_path)] #[files.split('.')[0] for root, dirs, files in os.walk(labels_path)]
    # print(all_labels)
    # print('100_nl_dames_finaleA_f122020_droite_1149' in all_labels)
    one_directory_per_video = False
    delete_folder = True

    # extract_and_linearise_csv(csv_path)
    if delete_folder:
        os.system('rm ' + frames_save_path + "/*")

    if not os.path.isdir(frames_save_path): os.mkdir(frames_save_path)

    for root, dirs, files in os.walk(every_videos_path) :
        dirs.sort()
        for file in files :
            video_path = os.path.join(root, file)
            repo = root.split('/')[-1]
            print(file + ' extracting')
            save_labelled_frames(video_path, all_labels, frames_save_path, repo,
                             one_directory_per_video, start=0, end=3)