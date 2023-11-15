#
# @Description:
# @Author:虚幻的元亨利贞
# @Time: 2022-04-24 21:17
#
import shutil
from urllib import request as url_request
from detect import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 设置允许的文件格式
ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG', 'bmp', 'mp4'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# detect.py
def yolo_detect(weights):
    opt = parse_opt()
    opt.weights = weights
    print_args(FILE.stem, opt)
    n = main(opt)
    return n


# 更新目录
def update_dir():
    shutil.rmtree('./data_flask/images')
    ## shutil.rmtree('./static/detect')
    os.mkdir('./data_flask/images')
    ## os.mkdir('./static/detect')


# 根据url下载图片
def url_save(url, filename):
    req = url_request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',
        'Cookie': 'SESSION=765bff9a-d6ff-4538-93ae-1a1fa2e3c327; _ga=GA1.2.833023310.1552467160; UM_distinctid=16adf48e17275d-0be8cf00f7c334-353166-1fa400-16adf48e173580; ASP.NET_SessionId=oitt4mutnu0kou1usezqdbf4'

    })
    res = url_request.urlopen(req)

    with res as response, open("data_flask/images/" + filename, 'wb') as f_save:
        f_save.write(response.read())
        f_save.flush()
        f_save.close()


def exg(upload_path):
    image = cv2.imread(upload_path, cv2.IMREAD_COLOR)
    img1 = np.array(image, dtype='int')  # 转换成int型，不然会导致数据溢出
    # 超绿灰度图
    r, g, b = cv2.split(img1)
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

    plt.figure(figsize=(10, 5), dpi=80)

    plt.subplot(132), plt.imshow(cv2.cvtColor(ExG, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(upload_path, dpi=800, bbox_inches='tight', pad_inches=0)
    plt.close()
    # img2 = plt.imread(upload_path)
    # img_s2 = img2[:, :, 0]  # 直接读入的img为3通道，这里用直接赋值的方法转为单通道
    # sc2 = plt.imshow(img_s2)
    # sc2.set_cmap('rainbow')  # 这里可以设置多种模式
    # plt.colorbar()  # 显示色度条
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.title(u'光谱指数模型')
    # plt.title('EXG')
    # plt.axis('off')
    # plt.savefig(exg_path)
    # plt.close()
    # file_list = set()
    return ExG


def erzhihua(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    img1 = np.array(image, dtype='int')  # 转换成int型，不然会导致数据溢出
    # 超绿灰度图
    r, g, b = cv2.split(img1)

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

    plt.figure(figsize=(10, 5), dpi=80)

    plt.subplot(133), plt.imshow(cv2.cvtColor(th2, cv2.COLOR_BGR2RGB)), \
        # plt.title('OTSU_bw'),
    plt.axis('off')
    plt.savefig(path, dpi=800, bbox_inches='tight', pad_inches=0)

    im_depth = cv2.imread(path)
    # 转换成伪彩色（之前必须是8位图片）
    # 这里有个alpha值，深度图转换伪彩色图的scale可以通过alpha的数值调整，我设置为1，感觉对比度大一些
    im_color = cv2.applyColorMap(cv2.convertScaleAbs(im_depth, alpha=2), cv2.COLORMAP_RAINBOW)
    # 转成png

    im = Image.fromarray(im_color)
    # 保存图片
    im.save(path)
