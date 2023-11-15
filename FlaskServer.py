import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from datetime import timedelta, datetime
import time
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from FlaskServerUtils import *
import matplotlib.pyplot as plt
import numpy as np
from loop_check import check_for_changes

app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


@app.route('/')
def loop_check():
    # 启动文件监控
    check_for_changes('loop_check.txt')


## 麦穗检测
@app.route('/spike', methods=['POST', 'GET'])
def spike():
    if request.method == 'POST':
        a = time.time()
        file = request.files.get('file')  # 通过file标签获取文件
        if not (file and allowed_file(file.filename)):
            return jsonify({"status": 10001, "message": "wrong image type"})
        area = request.values.get("area")
        if not area:
            return jsonify({"status": 10001, "message": "no area param"})

        update_dir()
        base_path = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(base_path, 'data_flask/images',
                                   secure_filename(file.filename))  # 一定要先创建该文件夹，不然会提示没有该路径
        file.save(upload_path)  # 保存文件

        n = yolo_detect(weights=ROOT / 'spike.pt')

        b = time.time()
        cal = int(round(b * 1000)) - int(round(a * 1000))
        # print('================cal:' + str(cal))
        res = ""
        try:
            res = str(n.item() / float(area))
        except AttributeError:
            res = str(0)

        return render_template('spike.html', filename=file.filename, folder='detect',
                               number=res + "个/平方米")  # 返回上传成功界面
        # return jsonify({"status": 10000, "message": res,
        #                 "data": "/usr/local/webserver/agricultural-images-processing-server/static/detect/" + file.filename})
    # 重新返回上传界面
    return render_template('spike.html', filename='wait.png', folder='style')


## 麦苗检测
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

        # return render_template('spike.html', filename=file.filename, folder='detect',
        #                        number=res + "个")  # 返回上传成功界面
        return jsonify({{"status": 10000, "message": "spike detection success",
                         "data": "/usr/local/webserver/agricultural-images-processing-server/static/detect/" + file.filename}})
    # 重新返回上传界面
    return render_template('spike.html', filename='wait.png', folder='style')


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

        n = yolo_detect(weights=ROOT / 'mobile.pt')
        res = ""
        try:
            res = str(n.item())
        except AttributeError:
            res = str(0)
        return res
    return ""


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

    plt.subplot(132)
    plt.imshow(cv2.cvtColor(ExG, cv2.COLOR_BGR2RGB))
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


if __name__ == '__main__':
    ## from werkzeug.contrib.fixers import ProxyFix

    ## app.wsgi_app = ProxyFix(app.wsgi_app)
    app.run()
