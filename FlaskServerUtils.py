#
# @Description:
# @Author:虚幻的元亨利贞
# @Time: 2022-04-24 21:17
#
import os
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
def yolo_detect(weights, source="data_flask/images", project='static/result'):
    opt = parse_opt()
    opt.weights = weights
    opt.source = source
    opt.project = project
    print_args(FILE.stem, opt)
    n = main(opt)
    return n


# 更新目录
def update_dir(dir):
    shutil.rmtree(dir)
    os.mkdir(dir)


# 解压缩
def unzip(compressed_file_path, extracted_folder_path, type_name):
    # 压缩文件路径
    compressed_file_path = compressed_file_path
    # 解压缩后的目标文件夹路径
    extracted_folder_path = extracted_folder_path
    # 确保目标文件夹存在，如果不存在则创建
    Path(extracted_folder_path).mkdir(parents=True, exist_ok=True)
    # 解压缩文件到目标文件夹
    shutil.unpack_archive(compressed_file_path, extracted_folder_path)

    origin_name = compressed_file_path.split(os.sep)[-1].split(".")[0]
    source_folder_path = extracted_folder_path + os.sep + origin_name
    target_folder_path = extracted_folder_path + os.sep + type_name
    # 确保目标文件夹存在，如果不存在则创建
    Path(target_folder_path).mkdir(parents=True, exist_ok=True)
    # 遍历源文件夹中的文件和子文件夹
    for root, dirs, files in os.walk(source_folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # 剪切文件到目标文件夹
            shutil.move(file_path, target_folder_path)

    # 删除源文件夹
    os.rmdir(source_folder_path)


##压缩
def zip(detect_path, destination_folder, zip_file_name):
    # 定义要压缩的文件夹路径
    detect_path = detect_path
    # 定义目标文件夹路径
    destination_folder = destination_folder
    # 定义zip文件的名称和路径
    zip_file_name = zip_file_name
    zip_file_path = os.path.join(destination_folder, zip_file_name)

    # 使用shutil库的make_archive方法压缩文件夹
    shutil.make_archive(zip_file_path, 'zip', detect_path)
    # 如果要删除原始文件夹内容，可以使用shutil.rmtree
    shutil.rmtree(detect_path)


# 实现均值滤波
def mean_filter(image, kernel_size):
    rows, cols = image.shape
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    filtered_image = np.zeros_like(image)
    for i in range(rows - kernel_size + 1):
        for j in range(cols - kernel_size + 1):
            image_patch = image[i:i + kernel_size, j:j + kernel_size]
            filtered_image[i + kernel_size // 2, j + kernel_size // 2] = np.sum(image_patch * kernel)
    return filtered_image


# 实现中值滤波
def median_filter(image, kernel_size):
    rows, cols = image.shape
    filtered_image = np.zeros_like(image)
    for i in range(rows - kernel_size + 1):
        for j in range(cols - kernel_size + 1):
            image_patch = image[i:i + kernel_size, j:j + kernel_size]
            median_value = np.median(image_patch)
            filtered_image[i + kernel_size // 2, j + kernel_size // 2] = median_value
    return filtered_image


# 阈值处理
def threshold(image):
    threshold_value = np.mean(image)
    thresholded_image = np.where(image > threshold_value, 255, 0)
    return thresholded_image


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
