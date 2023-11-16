import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from datetime import timedelta, datetime
import time
from flask import Flask, render_template, request, jsonify, url_for, send_file
from werkzeug.utils import secure_filename
import cv2
import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image, ImageFont, ImageDraw
import torch.nn.functional as F
from thop import profile
import zipfile

from prediction import detect_forword
from SimpleTrainAndClassify import *
from FlaskServerUtils import *
import matplotlib.pyplot as plt
import numpy as np

from loop_check import check_for_changes

from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor()

app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# 配置允许上传的文件类型和保存目录
UPLOAD_FOLDER = '/static/phe_folder'  # 请替换为你想要保存文件夹的实际路径
TEMP_UPLOAD_DIR = os.path.abspath('/static/phe_folder/temp_upload/')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}  # 允许的图片文件扩展名

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/check')
def loop_check():
    # 启动文件监控
    executor.submit(check_for_changes, 'loop_check.txt')
    return render_template('upload.html', filename='wait.png', folder='style')

@app.route('/ai')
def hello_world():
    return render_template('ai.html', filename='none.png', folder='style')



## 生育期检测 #################################################################################################
@app.route('/phe', methods=['POST', 'GET'])
def phe():
    if 'file' in request.files:
        image_folder = request.files['file']

        shutil.rmtree('./data_flask/phe')
        os.mkdir('./data_flask/phe')
        shutil.rmtree('./show_time')
        os.mkdir('./show_time')
        extract_folder = './show_time/'

        base_path = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(base_path, 'data_flask/phe',
                                   'show_time.zip')  # 一定要先创建该文件夹，不然会提示没有该路径
        image_folder.save(upload_path)  # 保存文件

        # 解压缩文件
        with zipfile.ZipFile(upload_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)

        # 在解压缩后更改文件名
        extracted_files = os.listdir(extract_folder)
        for filename in extracted_files:
            # 生成新的文件名
            new_filename = 'show_time'
            # 构建旧文件的完整路径和新文件的完整路径
            old_file_path = os.path.join(extract_folder, filename)
            new_file_path = os.path.join(extract_folder, new_filename)
            # 重命名文件
            os.rename(old_file_path, new_file_path)

            list_dir = os.listdir(new_file_path)
            nums = len(list_dir)
            if nums < 30:
                return jsonify({"status": 10001, "message": 'error',
                                "data": "图像帧数据小于30"})
            elif nums > 30:
                list_dir.sort()
                # 保留前30个文件，删除多余的文件
                files_to_keep = list_dir[:30]
                files_to_delete = list_dir[30:]

                for file_to_delete in files_to_delete:
                    file_path_to_delete = os.path.join(new_file_path, file_to_delete)
                    os.remove(file_path_to_delete)

                # 指定要遍历的文件夹路径
            folder_path = os.path.join(extract_folder, new_filename)

            # 获取文件夹名称
            folder_name = os.path.basename(folder_path)

            # 初始化计数器
            counter = 1

            # 遍历文件夹中的所有文件
            for filename in sorted(os.listdir(folder_path)):
                # 构建新的文件名
                new_filename = f"{folder_name}_{counter:02d}.jpg"

                # 构建完整的文件路径
                old_file_path = os.path.join(folder_path, filename)
                new_file_path = os.path.join(folder_path, new_filename)

                # 重命名文件
                os.rename(old_file_path, new_file_path)

                # 增加计数器
                counter += 1

        phenology_name = detect_forword()

        return jsonify({"status": 10000, "message": phenology_name,
                        "data": ""})

    return render_template('upload.html', filename='wait.png', folder='style')


## 麦穗检测 #################################################################################################
@app.route('/spike', methods=['POST', 'GET'])
def spike():
    if request.method == 'POST':
        a = time.time()
        file = request.files.get('file')  # 通过file标签获取文件
        if not (file and allowed_file(file.filename)):
            return jsonify({"status": 10001, "message": "wrong image type"})

        update_dir()
        base_path = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(base_path, 'data_flask/images',
                                   secure_filename(file.filename))  # 一定要先创建该文件夹，不然会提示没有该路径
        file.save(upload_path)  # 保存文件

        n = yolo_detect(weights=ROOT / 'spike.pt')

        b = time.time()
        cal = int(round(b * 1000)) - int(round(a * 1000))
        print('================cal:' + str(cal))
        res = ""
        try:
            res = str(n.item())
        except AttributeError:
            res = str(0)

            # return render_template('upload.html', filename=file.filename, folder='detect',
        #                        number=res + "个")  # 返回上传成功界面
        return jsonify({"status": 10000, "message": res,
                        "data": "/static/detect/" + file.filename})
    # 重新返回上传界面
    return render_template('upload.html', filename='wait.png', folder='style')


## 稻穗检测 #################################################################################################
@app.route('/rice', methods=['POST', 'GET'])
def rice():
    if request.method == 'POST':
        a = time.time()
        file = request.files.get('file')  # 通过file标签获取文件
        if not (file and allowed_file(file.filename)):
            return jsonify({"status": 10001, "message": "wrong image type"})

        update_dir()
        base_path = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(base_path, 'data_flask/images',
                                   secure_filename(file.filename))  # 一定要先创建该文件夹，不然会提示没有该路径
        file.save(upload_path)  # 保存文件

        n = yolo_detect(weights=ROOT / 'rice.pt')

        b = time.time()
        cal = int(round(b * 1000)) - int(round(a * 1000))
        print('================cal:' + str(cal))
        res = ""
        try:
            res = str(n.item())
        except AttributeError:
            res = str(0)

            # return render_template('upload.html', filename=file.filename, folder='detect',
        #                        number=res + "个")  # 返回上传成功界面
        return jsonify({"status": 10000, "message": res,
                        "data": "/static/detect/" + file.filename})
    # 重新返回上传界面
    return render_template('upload.html', filename='wait.png', folder='style')


## 麦苗检测 #################################################################################################
@app.route('/seedling', methods=['POST', 'GET'])
def seedling():
    if request.method == 'POST':
        file = request.files.get('file')  # 通过file标签获取文件
        if not (file and allowed_file(file.filename)):
            return jsonify({"status": 10001, "message": "wrong image type"})

        update_dir()
        base_path = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(base_path, 'data_flask/images',
                                   secure_filename(file.filename))  # 一定要先创建该文件夹，不然会提示没有该路径
        file.save(upload_path)  # 保存文件

        n = yolo_detect(weights=ROOT / 'seedling.pt')

        res = ""
        try:
            res = str(n.item())
        except AttributeError:
            res = str(0)

        # return render_template('upload.html', filename=file.filename, folder='detect',
        #                        number=res + "个")  # 返回上传成功界面
        return jsonify({"status": 10000, "message": res,
                        "data": "/static/detect/" + file.filename})
    # 重新返回上传界面
    return render_template('upload.html', filename='wait.png', folder='style')

    ## 草情检测 #################################################################################################


@app.route('/weeds', methods=['POST'])
def weeds():
    file = request.files.get('file')  # 通过file标签获取文件
    if not (file and allowed_file(file.filename)):
        return jsonify({"status": 10001, "message": "wrong image type"})

    update_dir()
    base_path = os.path.dirname(__file__)  # 当前文件所在路径
    upload_path = os.path.join(base_path, 'data_flask/images',
                               secure_filename(file.filename))  # 一定要先创建该文件夹，不然会提示没有该路径
    file.save(upload_path)  # 保存文件

    # exg方法注释，未启用
    # exg_data = exg(upload_path)
    result_path = os.path.join(base_path, 'static/detect',
                               secure_filename(file.filename))
    n = yolo_detect(weights=ROOT / 'seedling.pt')
    # 二值化
    erzhihua(result_path)
    return jsonify({"status": 10000, "message": "分析成功", "data": result_path})


## 水稻穗数检测 #################################################################################################
@app.route('/ricedemo', methods=['POST', 'GET'])
def ricedemo():
    if request.method == 'POST':
        file = request.files.get('file')  # 通过file标签获取文件
        if not (file and allowed_file(file.filename)):
            return jsonify({"status": 10001, "message": "wrong image type"})

        update_dir()
        base_path = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(base_path, 'data_flask/images',
                                   secure_filename(file.filename))  # 一定要先创建该文件夹，不然会提示没有该路径
        file.save(upload_path)  # 保存文件

        n = yolo_detect(weights=ROOT / 'best_rice.pt')

        res = ""
        try:
            res = str(n.item())
        except AttributeError:
            res = str(0)

        # return render_template('upload.html', filename=file.filename, folder='detect',
        #                        number=res + "个")  # 返回上传成功界面
        return jsonify({"status": 10000, "message": res,
                        "data": "/static/detect/" + file.filename})
    # 重新返回上传界面
    return render_template('upload.html', filename='wait.png', folder='style')


## 粮食检测 #################################################################################################
@app.route('/cropdemo', methods=['POST', 'GET'])
def cropdemo():
    if request.method == 'POST':
        file = request.files.get('file')  # 通过file标签获取文件
        if not (file and allowed_file(file.filename)):
            return jsonify({"status": 10001, "message": "wrong image type"})

        update_dir()
        base_path = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(base_path, 'data_flask/images',
                                   secure_filename(file.filename))  # 一定要先创建该文件夹，不然会提示没有该路径
        file.save(upload_path)  # 保存文件

        n = yolo_detect(weights=ROOT / 'crop4.pt')

        res = ""
        try:
            res = str(n.item())
        except AttributeError:
            res = str(0)

        return render_template('ai.html', filename=file.filename, folder='detect', des='测试结果')  # 返回上传成功界面
        # return jsonify({"status": 10000, "message": res,
        #                "data": "/static/detect/" + file.filename})
    # 重新返回上传界面
    return render_template('ai.html', filename='none.png', folder='style')


## 虫害检测 #################################################################################################
@app.route('/insectdemo', methods=['POST', 'GET'])
def insectdemo():
    if request.method == 'POST':
        file = request.files.get('file')  # 通过file标签获取文件
        if not (file and allowed_file(file.filename)):
            return jsonify({"status": 10001, "message": "wrong image type"})

        update_dir()
        base_path = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(base_path, 'data_flask/images',
                                   secure_filename(file.filename))  # 一定要先创建该文件夹，不然会提示没有该路径
        file.save(upload_path)  # 保存文件

        n = yolo_detect(weights=ROOT / 'insect.pt')

        res = ""
        try:
            res = str(n.item())
        except AttributeError:
            res = str(0)

        return render_template('ai.html', filename=file.filename, folder='detect', des='测试结果')  # 返回上传成功界面
        # return jsonify({"status": 10000, "message": res,
        #                "data": "/static/detect/" + file.filename})
    # 重新返回上传界面
    return render_template('ai.html', filename='none.png', folder='style')


## 草害检测 #################################################################################################
@app.route('/diseasedemo', methods=['POST', 'GET'])
def diseasedemo():
    if request.method == 'POST':
        file = request.files.get('file')  # 通过file标签获取文件
        if not (file and allowed_file(file.filename)):
            return jsonify({"status": 10001, "message": "wrong image type"})

        update_dir()
        base_path = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(base_path, 'data_flask/images',
                                   secure_filename(file.filename))  # 一定要先创建该文件夹，不然会提示没有该路径
        file.save(upload_path)  # 保存文件

        n = yolo_detect(weights=ROOT / 'plantdisease.pt')

        res = ""
        try:
            res = str(n.item())
        except AttributeError:
            res = str(0)

        return render_template('ai.html', filename=file.filename, folder='detect', des='测试结果')  # 返回上传成功界面
        # return jsonify({"status": 10000, "message": res,
        #                "data": "/static/detect/" + file.filename})
    # 重新返回上传界面
    return render_template('ai.html', filename='none.png', folder='style')


## 病害识别 #################################################################################################
@app.route('/diseaseClassifydemo', methods=['POST', 'GET'])
def diseaseClassifydemo():
    if request.method == 'POST':
        file = request.files.get('file')  # 通过file标签获取文件
        if not (file and allowed_file(file.filename)):
            return jsonify({"status": 10001, "message": "wrong image type"})

        update_dir()
        base_path = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(base_path, 'data_flask/diseaseClassifydemo',
                                   secure_filename(file.filename))  # 一定要先创建该文件夹，不然会提示没有该路径
        file.save(upload_path)  # 保存文件
        sc = SimpleClassification('disease')
        res = sc.classify('Plantdisease1.pt', 'data_flask/diseaseClassifydemo/' + file.filename,
                          'idx_to_labels_disease.npy')
        return render_template('ai.html', filename='img_pred.jpg', folder='detect', des='测试结果: ' + res)  # 返回上传成功界面
    # 重新返回上传界面
    return render_template('ai.html', filename='none.png', folder='style')


## 虫害识别 #################################################################################################
@app.route('/insectClassifydemo', methods=['POST', 'GET'])
def insectClassifydemo():
    if request.method == 'POST':
        file = request.files.get('file')  # 通过file标签获取文件
        if not (file and allowed_file(file.filename)):
            return jsonify({"status": 10001, "message": "wrong image type"})

        update_dir()
        base_path = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(base_path, 'data_flask/insectClassifydemo',
                                   secure_filename(file.filename))  # 一定要先创建该文件夹，不然会提示没有该路径
        file.save(upload_path)  # 保存文件
        sc = SimpleClassification('scene')
        res = sc.classify('Insect1.pt', 'data_flask/insectClassifydemo/' + file.filename, 'idx_to_labels_insect.npy')
        return render_template('ai.html', filename='img_pred.jpg', folder='detect', des='测试结果: ' + res)  # 返回上传成功界面
    # 重新返回上传界面
    return render_template('ai.html', filename='none.png', folder='style')


## 场景识别 #################################################################################################
@app.route('/sceneClassifydemo', methods=['POST', 'GET'])
def sceneClassifydemo():
    if request.method == 'POST':
        file = request.files.get('file')  # 通过file标签获取文件
        if not (file and allowed_file(file.filename)):
            return jsonify({"status": 10001, "message": "wrong image type"})

        update_dir()
        base_path = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(base_path, 'data_flask/sceneClassifydemo',
                                   secure_filename(file.filename))  # 一定要先创建该文件夹，不然会提示没有该路径
        file.save(upload_path)  # 保存文件

        sc = SimpleClassification('scene')
        res = sc.classify('scene.pt', 'data_flask/sceneClassifydemo/' + file.filename, 'idx_to_labels_scene.npy')
        return render_template('ai.html', filename='img_pred.jpg', folder='detect', des='测试结果: ' + res)  # 返回上传成功界面
    # 重新返回上传界面
    return render_template('ai.html', filename='none.png', folder='style')


## 叶龄识别 #################################################################################################
@app.route('/leafageClassifydemo', methods=['POST', 'GET'])
def leafageClassifydemo():
    if request.method == 'POST':
        file = request.files.get('file')  # 通过file标签获取文件
        if not (file and allowed_file(file.filename)):
            return jsonify({"status": 10001, "message": "wrong image type"})

        update_dir()
        base_path = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(base_path, 'data_flask/leafageClassifydemo',
                                   secure_filename(file.filename))  # 一定要先创建该文件夹，不然会提示没有该路径
        file.save(upload_path)  # 保存文件

        sc = SimpleClassification('Insect')
        res = sc.classify('LeafAge.pt', 'data_flask/leafageClassifydemo/' + file.filename, 'idx_to_labels_leaf.npy')
        return render_template('ai.html', filename='img_pred.jpg', folder='detect', des='测试结果: ' + res)  # 返回上传成功界面
    # 重新返回上传界面
    return render_template('ai.html', filename='none.png', folder='style')


## 穴数检测 #################################################################################################
@app.route('/holedemo', methods=['POST', 'GET'])
def holedemo():
    if request.method == 'POST':
        file = request.files.get('file')  # 通过file标签获取文件
        if not (file and allowed_file(file.filename)):
            return jsonify({"status": 10001, "message": "wrong image type"})

        update_dir()
        base_path = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(base_path, 'data_flask/images',
                                   secure_filename(file.filename))  # 一定要先创建该文件夹，不然会提示没有该路径
        file.save(upload_path)  # 保存文件

        n = yolo_detect(weights=ROOT / 'best_hole.pt')

        res = ""
        try:
            res = str(n.item())
        except AttributeError:
            res = str(0)

        # return render_template('upload.html', filename=file.filename, folder='detect',
        #                        number=res + "个")  # 返回上传成功界面
        return jsonify({"status": 10000, "message": res,
                        "data": "/static/detect/" + file.filename})
    # 重新返回上传界面
    return render_template('upload.html', filename='wait.png', folder='style')


## 图像分类 #################################################################################################
@app.route('/imageClassifydemo', methods=['POST', 'GET'])
def imageClassifydemo():
    if request.method == 'POST':
        file = request.files.get('file')  # 通过file标签获取文件
        if not (file and allowed_file(file.filename)):
            return jsonify({"status": 10001, "message": "wrong image type"})

        update_dir()
        base_path = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(base_path, 'data_flask/imageClassifydemo',
                                   secure_filename(file.filename))  # 一定要先创建该文件夹，不然会提示没有该路径
        file.save(upload_path)  # 保存文件

        # 导入中文字体，指定字号
        # font = ImageFont.truetype('SimHei.ttf', 32)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 载入预训练图像分类模型
        model = models.resnet18(pretrained=True)
        model = model.eval()
        model = model.to(device)
        # 测试集图像预处理-RCTN：缩放裁剪、转 Tensor、归一化
        test_transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(
                                                 mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                                             ])

        ## 图片路径
        img_path = 'data_flask/imageClassifydemo/' + file.filename
        img_pil = Image.open(img_path)

        input_img = test_transform(img_pil)  # 预处理
        input_img = input_img.unsqueeze(0).to(device)
        # 执行前向预测，得到所有类别的 logit 预测分数
        pred_logits = model(input_img)
        pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算

        n = 2
        top_n = torch.topk(pred_softmax, n)
        # 解析出类别
        pred_ids = top_n[1].cpu().detach().numpy().squeeze()
        # 解析出置信度
        confs = top_n[0].cpu().detach().numpy().squeeze()
        # 查表
        df = pd.read_csv('register_table/imagenet_class_index.csv')
        idx_to_labels = {}
        for idx, row in df.iterrows():
            idx_to_labels[row['ID']] = [row['wordnet'], row['Chinese']]

        # 保存结果
        res = ''
        img_bgr = cv2.imread(img_path)
        draw = ImageDraw.Draw(img_pil)
        for i in range(n):
            if i == 0:  ## 置信度最大
                class_name = idx_to_labels[pred_ids[i]][1]  # 获取类别名称
                confidence = confs[i] * 100  # 获取置信度
                # print(class_name)
                # 文字坐标，中文字符串，字体，rgba颜色
                # draw.text((50, 50), class_name, font=font, fill=(255, 0, 0, 1))
                res = class_name
        # 保存图像
        img_pil.save('static/detect/img_pred.jpg')
        return render_template('ai.html', filename='img_pred.jpg', folder='detect', des='测试结果: ' + res)  # 返回上传成功界面
    # 重新返回上传界面
    return render_template('ai.html', filename='none.png', folder='style')


# lai函数======================================================
def lai(img):
    file_path = getResultFilePath()
    original_path = file_path + img.filename
    img.save(original_path)
    image = cv2.imread(original_path, cv2.IMREAD_COLOR)
    # 转换成int型，不然会导致数据溢出
    img1 = np.array(image, dtype='int')
    # 超绿灰度图
    r, g, b = cv2.split(img1)
    ExR = 1.4 * r - g
    LAI = 4.297 * np.exp(-6.09 * 1.4 * r - g)
    # 确保LAI中的值在0~255之间
    LAI = np.clip(LAI, 0, 255)
    LAI = np.array(LAI, dtype='uint8')  # 重新转换成uint8类型

    plt.subplot(132), plt.imshow(cv2.cvtColor(LAI, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    LAI_path = file_path + img.filename.split('.')[0] + '_LAI.png'
    plt.savefig(LAI_path, dpi=800, bbox_inches='tight', pad_inches=0)
    plt.close()
    img2 = plt.imread(LAI_path)
    # 直接读入的img为3通道，这里用直接赋值的方法转为单通道
    img_s2 = img2[:, :, 0]
    sc2 = plt.imshow(img_s2)
    sc2.set_cmap('nipy_spectral')  # 这里可以设置多种模式
    plt.colorbar()  # 显示色度条
    # plt.rcParams['axes.unicode_minus'] = False

    plt.title('LAI')
    plt.axis('off')
    plt.savefig(LAI_path, dpi=800, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    file_list = set()
    img_list = [x for x in os.listdir(file_path) if x != img.filename]
    for num, i in enumerate(img_list):
        file_list.add(file_path + i)
    return file_list


# EXG函数
def exg(img):
    file_path = getResultFilePath()
    original_path = file_path + img.filename
    img.save(original_path)
    image = cv2.imread(original_path, cv2.IMREAD_COLOR)
    img1 = np.array(image, dtype='int')  # 转换成int型，不然会导致数据溢出
    # 超绿灰度图
    r, g, b = cv2.split(img1)
    # ExG_sub = cv2.subtract(2*g,r)
    # ExG = cv2.subtract(ExG_sub,b )
    ExG = 2 * g - r - b
    [m, n] = ExG.shape

    for i in range(m):
        for j in range(n):
            if ExG[i, j] < 0:
                ExG[i, j] = 0
            elif ExG[i, j] > 255:
                ExG[i, j] = 255

    ExG = np.array(ExG, dtype='uint8')  # 重新转换成uint8类型
    ret2, th2 = cv2.threshold(ExG, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    plt.subplot(132), plt.imshow(cv2.cvtColor(ExG, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    exg_path = file_path + img.filename.split('.')[0] + '_EXG.png'
    plt.savefig(exg_path, dpi=800, bbox_inches='tight', pad_inches=0)
    plt.close()
    img2 = plt.imread(exg_path)
    img_s2 = img2[:, :, 0]  # 直接读入的img为3通道，这里用直接赋值的方法转为单通道
    sc2 = plt.imshow(img_s2)
    sc2.set_cmap('nipy_spectral')  # 这里可以设置多种模式
    plt.colorbar()  # 显示色度条
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.title(u'光谱指数模型')
    plt.title('EXG')
    plt.axis('off')
    plt.savefig(exg_path, dpi=800, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    file_list = set()
    img_list = [x for x in os.listdir(file_path) if x != img.filename]
    for num, i in enumerate(img_list):
        file_list.add(file_path + i)
    return file_list


# 施肥
def shifei1(img, MAXExg, shifeizongliang, proportion):
    file_path = getResultFilePath()
    original_path = file_path + img.filename
    img.save(original_path)
    image = cv2.imread(original_path, cv2.IMREAD_COLOR)
    img1 = np.array(image, dtype='int')  # 转换成int型，不然会导致数据溢出
    # 超绿灰度图
    r, g, b = cv2.split(img1)
    # ExG_sub = cv2.subtract(2*g,r)
    # ExG = cv2.subtract(ExG_sub,b )
    ExG = 2 * g - r - b
    # shifei = (MAXExg - Exg) / (MAXExg * shifeizongliang * proportion)
    [m, n] = ExG.shape

    for i in range(m):
        for j in range(n):
            if ExG[i, j] < 0:
                ExG[i, j] = 0
            elif ExG[i, j] > 255:
                ExG[i, j] = 255
    ExG1 = np.array(ExG, dtype='uint8')
    shifei2 = (270 - ExG1) / (MAXExg * shifeizongliang * proportion)
    ExG11 = np.array(shifei2, dtype='uint8')  # 重新转换成uint8类型
    # shifei2 = (MAXExg - ExG) / (MAXExg * shifeizongliang * proportion)
    # ret2, th2 = cv2.threshold(ExG11, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # cv2.imshow('s',ExG11)
    # cv2.waitKey(0)
    plt.subplot(132), plt.imshow(cv2.cvtColor(ExG11, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    exg_path = file_path + img.filename.split('.')[0] + '_shifeidemo.png'
    plt.savefig(exg_path, dpi=800, bbox_inches='tight', pad_inches=0)
    plt.close()
    img2 = plt.imread(exg_path)
    img_s2 = img2[:, :, 0]  # 直接读入的img为3通道，这里用直接赋值的方法转为单通道
    sc2 = plt.imshow(ExG11)
    sc2.set_cmap('nipy_spectral')  # 这里可以设置多种模式
    plt.colorbar()  # 显示色度条
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.title(u'光谱指数模型')
    plt.title('chufang')
    plt.axis('off')
    plt.savefig(exg_path, dpi=800, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    file_list = set()
    img_list = [x for x in os.listdir(file_path) if x != img.filename]
    for num, i in enumerate(img_list):
        file_list.add(file_path + i)
    return file_list


# 倒伏函数
def daofu(img):
    file_path = getResultFilePath()
    original_path = file_path + img.filename
    img.save(original_path)
    # 读取原始图像的信息
    img0 = cv2.imread(original_path)  # 读取图像
    img1 = cv2.resize(img0, fx=0.9, fy=0.9, dsize=None)  # 调整图像大小
    img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 将图像转化为灰度图

    height = img1.shape[0]  # shape[0] 图像第一维度，高度
    width = img1.shape[1]  # shape[1] 图像第二维度，宽度
    plt.rcParams['font.family'] = 'SimHei'
    img11 = img2[0:-1:10, 0:-1:10]
    blur = cv2.blur(img11, (3, 3))  # 取3*3的矩阵 一般取奇数矩阵 均值滤波

    median = cv2.medianBlur(blur, 5)  # 中值滤波
    median1 = np.array(median, dtype='uint8')
    ret2, th2 = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sc = plt.imshow(th2, 'gray')
    sc.set_cmap('rainbow')  # 这里可以设置多种模式
    plt.colorbar()  # 显示色度条
    plt.xticks([]), plt.yticks([])

    plt.savefig(original_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    file_list = set()
    img_list = [x for x in os.listdir(file_path)]
    for num, i in enumerate(img_list):
        file_list.add(file_path + i)
    return file_list


def getResultFilePath():
    file_path = 'index_result/' + str(int(time.time())) + '/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    return file_path


@app.route("/LAI", methods=['POST'])
def LAI():
    img = request.files.get("file")
    if None == img:
        json_result = {
            "status": 10002,
            "message": "请求数据为空"
        }
        return jsonify(json_result)
    file_list = list(lai(img))

    data = {"file_list": file_list}
    json_result = {
        "status": 10000,
        "message": "图像分析成功",
        "data": data
    }
    return jsonify(json_result)


@app.route("/EXG", methods=['POST'])
def EXG():
    img = request.files.get("file")
    if None == img:
        json_result = {
            "status": 10002,
            "message": "请求数据为空"
        }
        return jsonify(json_result)
    file_list = list(exg(img))

    data = {"file_list": file_list}
    json_result = {
        "status": 10000,
        "message": "图像分析成功",
        "data": data
    }
    return jsonify(json_result)


@app.route("/df", methods=['POST'])
def df():
    img = request.files.get("file")
    if None == img:
        json_result = {
            "status": 10002,
            "message": "请求数据为空"
        }
        return jsonify(json_result)
    file_list = list(daofu(img))
    data = {"file_list": file_list}
    json_result = {
        "status": 10000,
        "message": "图像分析成功",
        "data": data
    }
    return jsonify(json_result)


@app.route("/result", methods=['POST'])
def result():
    file_path = request.form.get("file_path")
    if not os.path.exists(file_path):
        json_result = {
            "status": 10001,
            "message": "无有效分析结果数据"
        }
        return jsonify(json_result)
    return send_file(file_path, mimetype='image/png')


# 施肥
@app.route("/shifei", methods=['POST'])
def shifei():
    MAXExg = float(request.form.get("MAXExg"))
    shifeizongliang = float(request.form.get("shifeizongliang"))
    proportion = float(request.form.get("proportion"))

    img = request.files.get("file")
    if None == img:
        json_result = {
            "status": 10002,
            "message": "请求数据为空"
        }
        return jsonify(json_result)
    file_list = list(shifei1(img, MAXExg, shifeizongliang, proportion))

    data = {"file_list": file_list}
    json_result = {
        "status": 10000,
        "message": "图像分析成功",
        "data": data
    }
    return jsonify(json_result)


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':

        file = request.files['file']  # 通过file标签获取文件
        if not (file and allowed_file(file.filename)):
            return jsonify({"error": 1001, "msg": "图片类型：png、PNG、jpg、JPG、bmp"})

        update_dir()
        base_path = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(base_path, 'data_flask/images',
                                   secure_filename(file.filename))  # 一定要先创建该文件夹，不然会提示没有该路径
        file.save(upload_path)  # 保存文件

        n = yolo_detect(weights=ROOT / 'spike.pt')
        res = ""
        try:
            res = str(n.item())
        except AttributeError:
            res = str(0)
        return res
    return ""


if __name__ == '__main__':
    ## from werkzeug.contrib.fixers import ProxyFix

    ## app.wsgi_app = ProxyFix(app.wsgi_app)
    app.run()
