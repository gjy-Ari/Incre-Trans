'''
The prediction probability images of n stages is assembled to obtain the final classification results.
'''

import os
import numpy as np
from skimage import io

# Path of probability images predicted by n stage models
# (Stitching into a complete image in advance).
predict_probability_paths = r' '

# Save path of assemble-decision results.
outpaths = f' '

stage_number = 3

images_file = os.listdir(os.path.join(predict_probability_paths, 'probability_stage0'))
images_file_tif = []
for i in range(len(images_file)):
    if ".tif" in images_file[i]:
        images_file_tif.append(images_file[i])

for i in range(len(images_file_tif)):
    building_sigmoid = []
    background_sigmoid = []
    for j in range(stage_number):
        predict_stageN = io.imread(os.path.join(predict_probability_paths, f'probability_stage{j}', f'{images_file_tif[i]}'))
        for m in range(predict_stageN.shape[0]):
            for n in range(predict_stageN.shape[1]):
                if predict_stageN[m][n]>=0.5:
                    building_sigmoid.append(predict_stageN[m][n])
                else:
                    background_sigmoid.append(predict_stageN[m][n])

    building_sigmoid_mean = np.mean(building_sigmoid)
    background_sigmoid_mean = np.mean(background_sigmoid)
    building_sigmoid_std = np.std(building_sigmoid)
    background_sigmoid_std = np.std(background_sigmoid)
    building_threshold = building_sigmoid_mean - building_sigmoid_std
    background_threshold = background_sigmoid_mean + background_sigmoid_std

    # print(f'{images_file_tif[i]} building_threshold:', building_threshold)
    # print(f'{images_file_tif[i]} background_threshold:', background_threshold)

    predict_stage0 = io.imread(os.path.join(predict_probability_paths, f'probability_stage0', f'{images_file_tif[i]}'))
    predict_stage_all = np.zeros(shape=[predict_stage0.shape[0], predict_stage0.shape[1] * stage_number])
    for j in range(stage_number):
        predict_stageN = io.imread(os.path.join(predict_probability_paths, f'probability_stage{j}', f'{images_file_tif[i]}'))
        result_sigmoid = np.zeros(shape = predict_stageN.shape, dtype = np.float32)
        result = np.zeros(shape = predict_stageN.shape, dtype = np.uint8)
        predict_stage_all[:predict_stageN.shape[0],predict_stageN.shape[1]*j:predict_stageN.shape[1]*(j+1)] = predict_stageN

    for m in range(predict_stage0.shape[0]):
        for n in range(predict_stage0.shape[1]):
            m_n_value = []
            for a in range(stage_number):
                m_n_predict = predict_stage_all[m][a*predict_stage0.shape[1]+n]
                m_n_value.append(m_n_predict)
            if max(m_n_value)<=building_threshold and min(m_n_value)>=background_threshold:
                result_sigmoid[m][n] = max(m_n_value)
            elif max(m_n_value)>building_threshold and min(m_n_value)>=background_threshold:
                result_sigmoid[m][n] = max(m_n_value)
            elif max(m_n_value)<=building_threshold and min(m_n_value)<background_threshold:
                result_sigmoid[m][n] = min(m_n_value)
            else:
                result_sigmoid[m][n] = np.mean(m_n_value)

            if result_sigmoid[m][n]>=0.5:
                result[m][n] = 255

    save_file_name = images_file_tif[i].split('.')[0]

    output_dir = os.path.join(outpaths, 'result')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    io.imsave(os.path.join(output_dir, f'{save_file_name}.tif'), result)

    print(f'{i+1}th image is done')