# from tools.read_and_write import read_obj
# from tools.read_and_write import write_json
import re
import numpy as np

def read_obj(model_path, flags = ('v')):
    fid = open(model_path, 'r', encoding="utf-8")

    data = {}

    for head in flags:
        data[head] = []

    for line in fid:
        line = line.strip()
        if not line:
            continue
        line = re.split('\s+', line)
        if line[0] in flags:
            data[line[0]].append(line[1:])

    fid.close()

    if 'v' in data.keys():
        data['v'] = np.array(data['v']).astype(np.float)

    if 'vt' in data.keys():
        data['vt'] = np.array(data['vt']).astype(np.float)

    if 'vn' in data.keys():
        data['vn'] = np.array(data['vn']).astype(np.float)

    return data


def normalize_points(in_points, padding = 0.1):
    '''
    normalize points into [-0.5, 0.5]^3, and output point set and scale info.
    :param in_points: input point set, Nx3.
    :return:
    '''
    vertices = in_points.copy()

    bb_min = vertices.min(0)
    bb_max = vertices.max(0)
    total_size = (bb_max - bb_min).max() / (1 - padding)

    # centroid
    centroid = (bb_min + bb_max)/2.
    vertices = vertices - centroid

    vertices = vertices/total_size

    return vertices, total_size, centroid

def normalize_obj_file(input_obj_file, output_obj_file, padding = 0.1):
    """
    normalize vertices into [-0.5, 0.5]^3, and write the result into another .obj file.
    :param input_obj_file: input .obj file
    :param output_obj_file: output .obj file
    :return:
    """
    vertices = read_obj(input_obj_file)['v']

    vertices, total_size, centroid = normalize_points(vertices, padding=padding)

    input_fid = open(input_obj_file, 'r', encoding='utf-8')
    output_fid = open(output_obj_file, 'w', encoding='utf-8')

    v_id = 0
    for line in input_fid:
        if line.strip().split(' ')[0] != 'v':
            output_fid.write(line)
        else:
            output_fid.write(('v' + ' %f' * len(vertices[v_id]) + '\n') % tuple(vertices[v_id]))
            v_id = v_id + 1

    output_fid.close()
    input_fid.close()

    return total_size, centroid


input_file_path = './AnyConv.com__oventrop_101581431.obj'
output_obj_file = './normal.obj'
total_size, centroid = normalize_obj_file(input_file_path, output_obj_file,
                                            padding=0)

# size_centroid = {'size': total_size.tolist(), 'centroid': centroid.tolist()}

# write_json(size_centroid_file, size_centroid)