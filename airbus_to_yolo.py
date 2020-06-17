import os
import csv
import shutil

from PIL import Image


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


csv_file_path = '/home/william/Downloads/airbus-ship-detection/train_ship_segmentations_v2.csv'

image_folder_path = '/home/william/Downloads/airbus-ship-detection/train_v2'
output_folder = '/home/william/Datasets/Airbus'

############ READ CSV ############

images_index = {}
csv_file = open(csv_file_path)
csv_file_reader = csv.reader(csv_file, delimiter=',')
for line in csv_file_reader:
    if line[0] in images_index:
        images_index[line[0]].append(line[1])
    else:
        images_index[line[0]] = [line[1]]
csv_file.close()

if os.path.exists(output_folder):
    shutil.rmtree(output_folder, ignore_errors=True)
os.makedirs(output_folder)
output_folder_img = f'{output_folder}/Images'
output_folder_ann = f'{output_folder}/Images'
os.makedirs(output_folder_img)

for image_file_name in os.listdir(image_folder_path):
    image_path = f'{image_folder_path}/{image_file_name}'
    image_file = Image.open(image_path)
    width, height = image_file.size

    if image_file:
        out_file = open(
            f'{output_folder_ann}/{image_file_name.split(".")[0]}.txt', "w")
        airbus_coordinates = images_index[image_file_name]
        for airbus_coordinate in airbus_coordinates:
            airbus_coordinate = [int(i) for i in airbus_coordinate.split()]
            airbus_coordinate = zip(
                airbus_coordinate[0:-1:2], airbus_coordinate[1::2])
            x_values = []
            y_values = []
            for start, length in airbus_coordinate:
                for pixel_position in range(start, start + length):
                    x_values.append(pixel_position // width)
                    y_values.append(pixel_position % height)
            if x_values and y_values:
                point = (min(x_values), min(y_values),
                        max(x_values), max(y_values))
                bounding_box = convert(image_file.size, point)
                out_file.write(
                    f'0 {bounding_box[0]} {bounding_box[1]} {bounding_box[2]} {bounding_box[3]}\n')
        shutil.copy(f'{image_folder_path}/{image_file_name}',
                    f'{output_folder_img}/{image_file_name}')
        out_file.close()
