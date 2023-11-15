#
# @Description:
# @Author:虚幻的元亨利贞
# @Time: 2023-11-13 18:02
#
import time
from FlaskServerUtils import *

data_map = {}


def spike(file_name):
    update_dir()
    source_image_path = file_name
    destination_image_path = 'data_flask/images/' + file_name.split("/")[-1]
    shutil.copy(source_image_path, destination_image_path)

    n = yolo_detect(weights=ROOT / 'spike.pt')
    number = n.item()
    txt_name = "static/result/" + file_name.split("/")[-1].split(".")[0] + ".txt"
    with open(txt_name, 'w+') as file:
        file.write(str(number))


def do_request(file_name, flag):
    spike(file_name)
    print(f"图片地址: {file_name}, 做法: {flag}")


def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        return lines


def update_data_map(new_data):
    new_data_map = {}  # 创建临时的新map
    for line in new_data:
        line = line.strip().split()
        if line:
            key = line[0]
            value = line[1]
            if key not in data_map:
                new_data_map[key] = value
                print(f"新增数据: {key}: {value}")

    return new_data_map


def check_for_changes(filename):
    while True:
        data = read_file(filename)
        new_lines = [line for line in data if line.split()[0] not in data_map]
        if new_lines:
            new_data_map = update_data_map(new_lines)
            data_map.update(new_data_map)
            ## 取出更新内容
            if new_data_map:
                for key, value in new_data_map.items():
                    do_request(key, value)
        else:
            print("文件没有新增行数")

        time.sleep(1)
        print("过了1s")


if __name__ == '__main__':
    # 启动文件监控
    check_for_changes('loop_check.txt')
