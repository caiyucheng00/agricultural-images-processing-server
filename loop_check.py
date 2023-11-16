#
# @Description:
# @Author:虚幻的元亨利贞
# @Time: 2023-11-13 18:02
#
import os
import time
from FlaskServerUtils import *

data_map = {}


##======================================================================================================================
def LAI(file_name):
    pass


def NDVI(file_name):
    pass


def EXG(file_name):
    pass


def daofu(file_name):
    pass


def shifei(file_name):
    pass


def spike(file_name):
    source = "data_flask/flask_spike"
    project = "static/result_spike"
    Path(source).mkdir(parents=True, exist_ok=True)
    Path(project).mkdir(parents=True, exist_ok=True)

    flag_name = file_name.split(os.sep)[-1].split(".")[0]
    update_dir("data_flask/flask_spike")  # 每次清空
    unzip(file_name, "data_flask", "flask_spike")

    n = yolo_detect(weights=ROOT / 'spike.pt', source=source, project=project)
    number = []
    for tensor in n:
        number.append(tensor.item())
    txt_name = "static/result_spike/detect/" + flag_name + ".txt"
    with open(txt_name, 'w+') as file:
        for i in number:
            file.write(str(i) + '\n')

    dst_dir = "/home/downloads/results/"
    zip("static/result_spike/detect", dst_dir, "result_" + flag_name)


def rice(file_name):
    source = "data_flask/flask_rice"
    project = "static/result_rice"
    Path(source).mkdir(parents=True, exist_ok=True)
    Path(project).mkdir(parents=True, exist_ok=True)

    flag_name = file_name.split(os.sep)[-1].split(".")[0]
    update_dir("data_flask/flask_rice")  # 每次清空
    unzip(file_name, "data_flask", "flask_rice")

    n = yolo_detect(weights=ROOT / 'rice.pt', source=source, project=project)
    number = []
    for tensor in n:
        number.append(tensor.item())
    txt_name = "static/result_rice/detect/" + flag_name + ".txt"
    with open(txt_name, 'w+') as file:
        for i in number:
            file.write(str(i) + '\n')

    dst_dir = "/home/downloads/results/"
    zip("static/result_rice/detect", dst_dir, "result_" + flag_name)


def seedling(file_name):
    source = "data_flask/flask_seedling"
    project = "static/result_seedling"
    Path(source).mkdir(parents=True, exist_ok=True)
    Path(project).mkdir(parents=True, exist_ok=True)

    flag_name = file_name.split(os.sep)[-1].split(".")[0]
    update_dir("data_flask/flask_seedling")  # 每次清空
    unzip(file_name, "data_flask", "flask_seedling")

    n = yolo_detect(weights=ROOT / 'seedling.pt', source=source, project=project)
    number = []
    for tensor in n:
        number.append(tensor.item())
    txt_name = "static/result_seedling/detect/" + flag_name + ".txt"
    with open(txt_name, 'w+') as file:
        for i in number:
            file.write(str(i) + '\n')

    dst_dir = "/home/downloads/results/"
    zip("static/result_seedling/detect", dst_dir, "result_" + flag_name)


def phe(file_name):
    pass


def do_request(file_name, flag):
    if flag == "01":
        pass
    elif flag == "02":
        pass
    elif flag == "03":
        pass
    elif flag == "04":
        pass
    elif flag == "05":
        pass
    elif flag == "06":
        pass
    elif flag == "07":
        pass
    elif flag == "08":
        spike(file_name)
    elif flag == "09":
        rice(file_name)
    elif flag == "10":
        seedling(file_name)
    elif flag == "11":
        time.sleep(3)

    print(f"图片地址: {file_name}, 做法: {flag}")
    time.sleep(1)


##======================================================================================================================
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
                    do_request(key, value)  ## 新数据
        else:
            print("文件没有新增行数")

        time.sleep(10)
        print("过了10s")


if __name__ == '__main__':
    # 启动文件监控
    check_for_changes('/root/Downloads/upload/upload.txt')
