import pickle
import numpy as np
import pandas as pd
import os
import torch
import xml.etree.ElementTree as et

def get_style(file_name) :
    return file_name.split("_")[0]


def get_frame_nb(file_name) :
    nb_jpg = file_name.split("_")[1]
    nb = nb_jpg.split(".")[0]
    return int(nb)


def get_barycenter(node) :
    xmin = int(node[4][0].text)
    ymin = int(node[4][1].text)
    xmax = int(node[4][2].text)
    ymax = int(node[4][3].text)

    x = int((xmin + xmax) * 0.5)
    y = int((ymin + ymax) * 0.5)
    return x, y


def get_jpg_name(file_name) :
    name = file_name.split('.')[0]
    name += '.jpg'
    return name


def xml_to_dataframe(xml_repo_path) :
    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []
    frameName_list = []
    crawl_list = []
    breaststroke_list = []

    print('Files with less than 8 swimmers :')
    for root, dirs, files in os.walk(xml_repo_path):
        for file_name in files:
            frame_name = get_jpg_name(file_name)
            style = get_style(file_name)
            xml_path = os.path.join(root, file_name)

            xtree = et.parse(xml_path)
            xroot = xtree.getroot()

            # if len(xroot) == 14 : # ie 8 swimmmers
            for node in xroot:
                if node.tag == 'object':
                    xmin = int(node[4][0].text)
                    ymin = int(node[4][1].text)
                    xmax = int(node[4][2].text)
                    ymax = int(node[4][3].text)

                    xmin_list.append(xmin)
                    ymin_list.append(ymin)
                    xmax_list.append(xmax)
                    ymax_list.append(ymax)
                    frameName_list.append(frame_name)
                    crawl_list.append(1 if style == 'crawl' else 0)
                    breaststroke_list.append(1 if style == 'breaststroke' else 0)
            # else : print('\t', file_name, len(xroot))
    print('End of files list\n')

    data = pd.DataFrame()
    data['xmin'] = xmin_list
    data['ymin'] = ymin_list
    data['xmax'] = xmax_list
    data['ymax'] = ymax_list
    data['frame'] = frameName_list
    data['crawl'] = crawl_list
    data['breaststroke'] = breaststroke_list
    data['direction'] = pd.Series(0 for d in range(data.shape[0])) # No direction for now
    data['confidence'] = pd.Series(1 for d in range(data.shape[0]))

    return data


def specify_cells_and_coords(data, model_size, last_layer_base) :
    assert model_size[0] % last_layer_base[0] == 0 and model_size[1] % last_layer_base[1] == 0
    x_cell_size = np.int16(model_size[0] / last_layer_base[0])
    y_cell_size = np.int16(model_size[1] / last_layer_base[1])

    x_barycenter = np.int16(data['xmin'] + data['xmax']) / 2
    y_barycenter = np.int16(data['ymin'] + data['ymax']) / 2
    data['cell_x'] = np.int16(x_barycenter / x_cell_size)
    data['cell_y'] = np.int16(y_barycenter / y_cell_size)

    x_cell_origin = x_cell_size * data['cell_x']
    y_cell_origin = y_cell_size * data['cell_y']
    delta_xmin = np.float64(data['xmin'] - x_cell_origin) / x_cell_size
    delta_ymin = np.float64(data['ymin'] - y_cell_origin) / y_cell_size
    delta_xmax = np.float64(data['xmax'] - x_cell_origin) / x_cell_size
    delta_ymax = np.float64(data['ymax'] - y_cell_origin) / y_cell_size

    data['xmin'] = delta_xmin
    data['ymin'] = delta_ymin
    data['xmax'] = delta_xmax
    data['ymax'] = delta_ymax

    return data


def path_to_name(path) :
    name = path.split('/')[-1]
    return name


def assign_out(out, line, coords, shift=(0, 0)) :
    out[coords[0]+shift[0], coords[1]+shift[1], 0] = line['crawl']
    out[coords[0]+shift[0], coords[1]+shift[1], 1] = line['breaststroke']
    out[coords[0]+shift[0], coords[1]+shift[1], 2] = line['direction']
    out[coords[0]+shift[0], coords[1]+shift[1], 3] = line['xmin']-shift[0]
    out[coords[0]+shift[0], coords[1]+shift[1], 4] = line['ymin']-shift[1]
    out[coords[0]+shift[0], coords[1]+shift[1], 5] = line['xmax']-shift[0]
    out[coords[0]+shift[0], coords[1]+shift[1], 6] = line['ymax']-shift[1]
    out[coords[0]+shift[0], coords[1]+shift[1], 7] = line['confidence'] * shift[2]
    return out


def find_pertinent_shifts(line) :
    shifts = [(0, 0, 1)]
    x_bary = (line['xmin'] + line['xmax']) / 2
    if x_bary < 0.2 :
        shifts.append((-1, 0, 1-x_bary))
    if x_bary > 0.8 :
        shifts.append((1, 0, x_bary))
    return shifts


def tensors_generation(data, last_layer_base) :
    output_depth = 8  # crawl, breaststroke, direction, xmin, ymin, xmax, ymax, confidence
    tensor_shape = (last_layer_base[0], last_layer_base[1], output_depth)
    reverse_mask_list = {}
    output_list = {}
    confidence_lambda_tensor_list = {}

    unique_frames_list = list(dict.fromkeys(data['frame']))

    for frame_name in unique_frames_list :
        out = torch.zeros(tensor_shape)
        reverse_mask = torch.ones(tensor_shape)
        reverse_mask[:, :, 7] = 0.0  # set every confidence to 0
        confidence_lambda_tensor = torch.ones(tensor_shape)
        confidence_lambda_tensor[:, :, 7] = 0.5

        matching_elt = data[data['frame'] == frame_name]
        if matching_elt.size > 0 :
            if frame_name == 'breaststroke_665.jpg' :
                pass
            for _, line in matching_elt.iterrows():
                coords = np.int8(line['cell_x']), np.int8(line['cell_y'])
                shifts = find_pertinent_shifts(line)
                for shift in shifts :
                    if 0 < coords[0] + shift[0] < out.shape[0] and 0 < coords[1] + shift[1] < out.shape[1]:
                        out = assign_out(out, line, coords, shift)
                        reverse_mask[coords[0]+shift[0], coords[1]+shift[1], :] = 0.0
                        if line['confidence'] > 0.5:
                            confidence_lambda_tensor[coords[0]+shift[0], coords[1]+shift[1], 7] = 5

                key = frame_name
                output_list[key] = out
                reverse_mask_list[key] = reverse_mask
                confidence_lambda_tensor_list[key] = confidence_lambda_tensor

    return output_list, reverse_mask_list, confidence_lambda_tensor_list


def save_obj(obj, name) :
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def get_nb(frame_names) :
    frame_nb = []
    for name in frame_names :
        nb = name.split('_')[-1].split('.')[0]
        frame_nb.append(int(nb))
    return frame_nb


# ya comme un truc qui va pas parce que j'ai pas trié par y. Mais en fait il semblerait que ça soit rattrapé par le fait que les données ont été rentrées dans l'ordre des y (ou l'ordre inverse mais peu importe)
def interpolate_gaps(data) :
    data['frame_nb'] = get_nb(data['frame'])
    data = data.sort_values(by=['crawl', 'frame_nb', 'ymin'])

    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []
    frame_name_list = []
    crawl_list = []
    breaststroke_list = []

    for (_, current), (_, next) in zip(data[:-8].iterrows(), data[8:].iterrows()) :
        if current.crawl == next.crawl :
            style = 'breaststroke_' if next.crawl == 0 else 'crawl_'

            nb_steps = next.frame_nb - current.frame_nb
            xmin_step = (next.xmin - current.xmin) / nb_steps
            ymin_step = (next.ymin - current.ymin) / nb_steps
            xmax_step = (next.xmax - current.xmax) / nb_steps
            ymax_step = (next.ymax - current.ymax) / nb_steps
            for i in range(1, nb_steps) :
                xmin = int(current.xmin + i * xmin_step)
                ymin = int(current.ymin + i * ymin_step)
                xmax = int(current.xmax + i * xmax_step)
                ymax = int(current.ymax + i * ymax_step)
                frame_name = style + str(current.frame_nb + i) + '.jpg'

                xmin_list.append(xmin)
                ymin_list.append(ymin)
                xmax_list.append(xmax)
                ymax_list.append(ymax)
                frame_name_list.append(frame_name)
                crawl_list.append(current.crawl)
                breaststroke_list.append(current.breaststroke)

    data2 = pd.DataFrame()
    data2['xmin'] = xmin_list
    data2['ymin'] = ymin_list
    data2['xmax'] = xmax_list
    data2['ymax'] = ymax_list
    data2['frame'] = frame_name_list
    data2['crawl'] = crawl_list
    data2['breaststroke'] = breaststroke_list
    data2['direction'] = pd.Series(0 for d in range(data2.shape[0]))  # No direction for now
    data2['confidence'] = pd.Series(1 for d in range(data2.shape[0]))

    data = data.drop(columns=['frame_nb'])
    data = pd.concat([data, data2])
    data = data.sort_values(by=['crawl', 'frame', 'ymin'])

    return data


if __name__=='__main__' :
    last_layer_base = (10, 10)
    xml_repo_path = '/home/amigo/Bureau/data/video_for_extracting/label'
    output_tensors_save_path = '/home/amigo/Bureau/data/video_for_extracting/label' # '../data/images/boxes_annotated_runs_'+str(last_layer_base[0])+'x'+str(last_layer_base[1])
    # model_size = (640, 320, 3)
    model_size = (1000, 500)

    data = xml_to_dataframe(xml_repo_path)
    print(data[data['frame'] == '100_nl_dames_finaleA_f122020_gauche_99.jpg'])
    data.to_pickle("save.pkl")
    with open(os.path.join(xml_repo_path + "save.pkl"), 'wb') as f:
        pickle.dump(data, f)
    # data = interpolate_gaps(data)
    #
    # # Vizvid stuff
    # data['frame_number'] = [get_frame_nb(name) for name in data['frame']]
    # data['x1'] = [v*50/640 for v in data['xmin']]
    # data['y1'] = [int(v*1000/320) for v in data['ymin']]
    # data['x2'] = [v*50/640 for v in data['xmax']]
    # data['y2'] = [int(v*1000/320) for v in data['ymax']]
    # data = data[data['crawl'] == 1]
    # data = data[['x1', 'y1', 'x2', 'y2', 'direction', 'frame_number']]
    # data.to_csv(output_tensors_save_path)

    # data = specify_cells_and_coords(data, model_size, last_layer_base)
    # output_list, reverse_mask_list, confidence_lambda_tensor_list = tensors_generation(data, last_layer_base)
    # save_obj((output_list, reverse_mask_list, confidence_lambda_tensor_list), output_tensors_save_path)
