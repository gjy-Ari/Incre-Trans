'''stageN_file_names.json'''
import os
import json

# Path of training dataset
file_path = r'/home/train_image'
file_name = os.listdir(file_path)

with open('/home/file_name/stage2_file_names.json', 'w', encoding='utf-8') as file:
    json.dump(file_name, file, ensure_ascii=False)