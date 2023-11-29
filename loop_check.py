#
# @Description:
# @Author:虚幻的元亨利贞
# @Time: 2023-11-13 18:02
#
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
from FlaskServerUtils import *
from osgeo import gdal
from scipy.ndimage import zoom
import matplotlib
matplotlib.use('TkAgg')  # 切换后端为 TkAgg

data_map = {}
CHECK_FILE = '/root/Downloads/upload/upload.txt'
RESULT_LAI = '/root/Downloads/result/LAI/'
RESULT_EXG = '/root/Downloads/result/EXG/'
RESULT_xiaomaidaofu = '/root/Downloads/result/xiaomaidaofu/'
RESULT_shuidaodaofu = '/root/Downloads/result/shuidaodaofu/'
RESULT_xiaomaishifei = '/root/Downloads/result/xiaomaishifei/'
RESULT_shuidaoshifei = '/root/Downloads/result/shuidaoshifei/'
RESULT_SPIKE = '/root/Downloads/result/xiaomaisuishu/'
RESULT_RICE = '/root/Downloads/result/shuidaosuishu/'
RESULT_SEEDLING = '/root/Downloads/result/xiaomaimiaoshu/'


##======================================================================================================================
def LAI(file_name):
    source = "data_flask/flask_LAI"
    project = "static/result_LAI/index"
    Path(source).mkdir(parents=True, exist_ok=True)
    Path(project).mkdir(parents=True, exist_ok=True)

    flag_name = file_name.split(os.sep)[-1].split(".")[0]
    update_dir(source)  # 每次清空
    unzip(file_name, "data_flask", "flask_LAI")

    for img in os.listdir(source):
        # image = cv2.imread(source + os.sep + img, cv2.IMREAD_COLOR)
        # # 转换成int型，不然会导致数据溢出
        # img1 = np.array(image, dtype='int')
        # # 超绿灰度图
        # r, g, b = cv2.split(img1)
        # ExR = 1.4 * r - g
        # LAI = 4.297 * np.exp(-6.09 * 1.4 * r - g)
        # # 确保LAI中的值在0~255之间
        # LAI = np.clip(LAI, 0, 255)
        # LAI = np.array(LAI, dtype='uint8')  # 重新转换成uint8类型
        #
        # plt.subplot(132), plt.imshow(cv2.cvtColor(LAI, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # LAI_path = project + os.sep + "result_" + str(img)
        # plt.savefig(LAI_path, dpi=800, bbox_inches='tight', pad_inches=0)
        # plt.close()
        # img2 = plt.imread(LAI_path)
        # # 直接读入的img为3通道，这里用直接赋值的方法转为单通道
        # img_s2 = img2[:, :, 0]
        # sc2 = plt.imshow(img_s2)
        # sc2.set_cmap('nipy_spectral')  # 这里可以设置多种模式
        # plt.colorbar()  # 显示色度条
        # # plt.rcParams['axes.unicode_minus'] = False
        #
        # plt.title('LAI')
        # plt.axis('off')
        # plt.savefig(LAI_path, dpi=800, bbox_inches='tight', pad_inches=0.2)
        # plt.close()

        isCrop = False
        index = str(img).split('.')[1]
        if index == 'tif':
            isCrop = True
        # 打开原始图像文件
        dataset = gdal.Open(source + os.sep + img, gdal.GA_ReadOnly)
        if dataset is None:
            print("无法打开文件")
            exit(-1)
        # 获取原始图像的宽度和高度
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        # 定义裁剪的左上角和右下角坐标（这里是裁剪宽高各占原图的70%，即原图中间部分）
        left = int(width * 0.15)
        top = int(height * 0.15)
        right = int(width * 0.85)
        bottom = int(height * 0.85)
        # 计算裁剪后图像的宽度和高度
        new_width = right - left
        new_height = bottom - top
        if isCrop:
            # 读取原始图像数据
            data = dataset.ReadAsArray(left, top, new_width, new_height)
        else:
            data = dataset.ReadAsArray(0, 0, width, height)
        # 数据类型转换到浮点型并归一化到 0-1 范围内
        data = data.astype(np.float32) / 255.0

        # 计算 LAI
        r, g, b = data[0], data[1], data[2]
        ExR = 1.4 * r - g
        LAI = 4.297 * np.exp(-6.09 * 1.4 * r - g)
        # 将 LAI 值限制在 0 到 1 之间
        LAI = np.clip(LAI, 0, 1)
        # 创建一个彩色映射
        cmap = plt.cm.plasma  # 选择彩色映射，nipy_spectral
        LAI_color = (cmap(LAI) * 255).astype(np.uint8)  # 将 ExG 值映射为 RGB 彩色空间

        # 创建输出图像
        driver = gdal.GetDriverByName("GTiff")
        if isCrop:
            out_dataset = driver.Create(project + os.sep + str(img), new_width, new_height, 3, gdal.GDT_Byte)
        else:
            out_dataset = driver.Create(project + os.sep + str(img), width, height, 3, gdal.GDT_Byte)
        if out_dataset is None:
            print("无法创建输出文件")
            exit(-1)

        # 将地理空间信息设置到输出图像中
        geotransform = list(dataset.GetGeoTransform())
        geotransform[0] = geotransform[0] + left * geotransform[1]
        geotransform[3] = geotransform[3] + top * geotransform[5]
        out_dataset.SetGeoTransform(tuple(geotransform))
        out_dataset.SetProjection(dataset.GetProjection())

        # 写入颜色数据到输出图像的对应波段
        for i in range(3):
            out_band = out_dataset.GetRasterBand(i + 1)
            out_band.WriteArray(LAI_color[:, :, i])

        # 创建一个空的 Matplotlib 图像
        fig, ax = plt.subplots(figsize=(6, 6))
        # 在图像上显示 ExG_color 数据
        img_ax = ax.imshow(LAI_color)
        # 添加彩色图例条
        cbar = plt.colorbar(img_ax, ax=ax)
        cbar.set_label('LAI')
        # 隐藏坐标轴
        ax.axis('off')
        # 保存图像
        plt.savefig(project + os.sep + 'axis_' + str(img), bbox_inches='tight')
        # 关闭数据集
        dataset = None
        out_dataset = None

    dst_dir = RESULT_LAI
    zip("static/result_LAI/index", dst_dir, "result_" + flag_name)


def NDVI(file_name):
    pass


def EXG(file_name):
    source = "data_flask/flask_EXG"
    project = "static/result_EXG/index"
    Path(source).mkdir(parents=True, exist_ok=True)
    Path(project).mkdir(parents=True, exist_ok=True)

    flag_name = file_name.split(os.sep)[-1].split(".")[0]
    update_dir(source)  # 每次清空
    unzip(file_name, "data_flask", "flask_EXG")

    for img in os.listdir(source):
        # image = cv2.imread(source + os.sep + img, cv2.IMREAD_COLOR)
        # img1 = np.array(image, dtype='int')  # 转换成int型，不然会导致数据溢出
        # # 超绿灰度图
        # r, g, b = cv2.split(img1)
        # ExG = 2 * g - r - b
        # [m, n] = ExG.shape
        #
        # for i in range(m):
        #     for j in range(n):
        #         if ExG[i, j] < 0:
        #             ExG[i, j] = 0
        #         elif ExG[i, j] > 255:
        #             ExG[i, j] = 255
        #
        # ExG = np.array(ExG, dtype='uint8')  # 重新转换成uint8类型
        # ret2, th2 = cv2.threshold(ExG, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #
        # plt.subplot(132), plt.imshow(cv2.cvtColor(ExG, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # exg_path = project + os.sep + "result_" + str(img)
        # plt.savefig(exg_path, dpi=800, bbox_inches='tight', pad_inches=0)
        # plt.close()
        #
        # img2 = plt.imread(exg_path)
        # img_s2 = img2[:, :, 0]  # 直接读入的img为3通道，这里用直接赋值的方法转为单通道
        # sc2 = plt.imshow(img_s2)
        # sc2.set_cmap('nipy_spectral')  # 这里可以设置多种模式
        # plt.colorbar()  # 显示色度条
        # plt.title('EXG')
        # plt.axis('off')
        # plt.savefig(exg_path, dpi=800, bbox_inches='tight', pad_inches=0.2)
        # plt.close()

        isCrop = False
        index = str(img).split('.')[1]
        if index == 'tif':
            isCrop = True
        # 打开原始图像文件
        dataset = gdal.Open(source + os.sep + img, gdal.GA_ReadOnly)
        if dataset is None:
            print("无法打开文件")
            exit(-1)
        # 获取原始图像的宽度和高度
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        # 定义裁剪的左上角和右下角坐标（这里是裁剪宽高各占原图的70%，即原图中间部分）
        left = int(width * 0.15)
        top = int(height * 0.15)
        right = int(width * 0.85)
        bottom = int(height * 0.85)
        # 计算裁剪后图像的宽度和高度
        new_width = right - left
        new_height = bottom - top
        if isCrop:
            # 读取原始图像数据
            data = dataset.ReadAsArray(left, top, new_width, new_height)
        else:
            data = dataset.ReadAsArray(0, 0, width, height)
        # 数据类型转换到浮点型并归一化到 0-1 范围内
        data = data.astype(np.float32) / 255.0

        # 计算 ExG
        r, g, b = data[0], data[1], data[2]
        ExG = 2 * g - r - b
        # 将 ExG 值限制在 0 到 1 之间
        ExG = np.clip(ExG, 0, 1)
        # 创建一个彩色映射
        cmap = plt.cm.nipy_spectral  # 选择彩色映射，nipy_spectral
        ExG_color = (cmap(ExG) * 255).astype(np.uint8)  # 将 ExG 值映射为 RGB 彩色空间

        # 创建输出图像
        driver = gdal.GetDriverByName("GTiff")
        if isCrop:
            out_dataset = driver.Create(project + os.sep + str(img), new_width, new_height, 3, gdal.GDT_Byte)
        else:
            out_dataset = driver.Create(project + os.sep + str(img), width, height, 3, gdal.GDT_Byte)
        if out_dataset is None:
            print("无法创建输出文件")
            exit(-1)

        # 将地理空间信息设置到输出图像中
        geotransform = list(dataset.GetGeoTransform())
        geotransform[0] = geotransform[0] + left * geotransform[1]
        geotransform[3] = geotransform[3] + top * geotransform[5]
        out_dataset.SetGeoTransform(tuple(geotransform))
        out_dataset.SetProjection(dataset.GetProjection())

        # 写入颜色数据到输出图像的对应波段
        for i in range(3):
            out_band = out_dataset.GetRasterBand(i + 1)
            out_band.WriteArray(ExG_color[:, :, i])

        # 创建一个空的 Matplotlib 图像
        fig, ax = plt.subplots(figsize=(6, 6))
        # 在图像上显示 ExG_color 数据
        img_ax = ax.imshow(ExG_color)
        # 添加彩色图例条
        cbar = plt.colorbar(img_ax, ax=ax)
        cbar.set_label('ExG')
        # 隐藏坐标轴
        ax.axis('off')
        # 保存图像
        plt.savefig(project + os.sep + 'axis_' + str(img), bbox_inches='tight')
        # 关闭数据集
        dataset = None
        out_dataset = None

    dst_dir = RESULT_EXG
    zip("static/result_EXG/index", dst_dir, "result_" + flag_name)


def xiaomaidaofu(file_name):
    source = "data_flask/flask_xiaomaidaofu"
    project = "static/result_xiaomaidaofu/index"
    Path(source).mkdir(parents=True, exist_ok=True)
    Path(project).mkdir(parents=True, exist_ok=True)

    flag_name = file_name.split(os.sep)[-1].split(".")[0]
    update_dir(source)  # 每次清空
    unzip(file_name, "data_flask", "flask_xiaomaidaofu")

    for img in os.listdir(source):
        # 读取原始图像的信息
        # img0 = cv2.imread(source + os.sep + img)  # 读取图像
        # img1 = cv2.resize(img0, fx=0.9, fy=0.9, dsize=None)  # 调整图像大小
        # img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 将图像转化为灰度图
        #
        # height = img1.shape[0]  # shape[0] 图像第一维度，高度
        # width = img1.shape[1]  # shape[1] 图像第二维度，宽度
        # plt.rcParams['font.family'] = 'SimHei'
        # img11 = img2[0:-1:10, 0:-1:10]
        # blur = cv2.blur(img11, (3, 3))  # 取3*3的矩阵 一般取奇数矩阵 均值滤波
        #
        # median = cv2.medianBlur(blur, 5)  # 中值滤波
        # median1 = np.array(median, dtype='uint8')
        # ret2, th2 = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # sc = plt.imshow(th2, 'gray')
        # sc.set_cmap('rainbow')  # 这里可以设置多种模式
        # plt.colorbar()  # 显示色度条
        # plt.xticks([]), plt.yticks([])
        #
        # save_path = project + os.sep + "result_" + str(img)
        # plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
        # plt.close()

        isCrop = False
        index = str(img).split('.')[1]
        if index == 'tif':
            isCrop = True
        # 打开原始图像文件
        dataset = gdal.Open(source + os.sep + img, gdal.GA_ReadOnly)
        if dataset is None:
            print("无法打开文件")
            exit(-1)
        # 获取原始图像的宽度和高度
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        # 定义裁剪的左上角和右下角坐标（这里是裁剪宽高各占原图的70%，即原图中间部分）
        left = int(width * 0.15)
        top = int(height * 0.15)
        right = int(width * 0.85)
        bottom = int(height * 0.85)
        # 计算裁剪后图像的宽度和高度
        new_width = right - left
        new_height = bottom - top
        if isCrop:
            # 读取原始图像数据
            data = dataset.ReadAsArray(left, top, new_width, new_height)
        else:
            data = dataset.ReadAsArray(0, 0, width, height)
        # 将图像转化为灰度图
        gray_data = 0.2989 * data[0] + 0.587 * data[1] + 0.114 * data[2]
        gray_data = np.clip(gray_data, 0, 255)  # 确保灰度值在合理范围内

        # 对灰度图进行下采样处理
        downsampled_data = gray_data[0:-1:10, 0:-1:10]
        # 应用均值滤波
        blurred_data = mean_filter(downsampled_data, 3)
        # 应用中值滤波
        blurred_data = median_filter(blurred_data, 5)
        # 将下采样后的图像上采样回原始尺寸
        upsampled_data = zoom(blurred_data, (10, 10), order=1)
        thresholded_data = threshold(upsampled_data)
        # 创建一个彩色映射
        cmap = plt.cm.rainbow  # 选择彩色映射，nipy_spectral
        color = (cmap(thresholded_data) * 255).astype(np.uint8)  # 将 ExG 值映射为 RGB 彩色空间

        # 创建输出图像
        driver = gdal.GetDriverByName("GTiff")
        out_dataset = driver.Create(project + os.sep + str(img), upsampled_data.shape[1], upsampled_data.shape[0], 3, gdal.GDT_Byte)
        if out_dataset is None:
            print("无法创建输出文件")
            exit(-1)

        # 将地理空间信息设置到输出图像中
        geotransform = list(dataset.GetGeoTransform())
        geotransform[0] = geotransform[0] + left * geotransform[1]
        geotransform[3] = geotransform[3] + top * geotransform[5]
        out_dataset.SetGeoTransform(tuple(geotransform))
        out_dataset.SetProjection(dataset.GetProjection())

        # 写入颜色数据到输出图像的对应波段
        for i in range(3):
            out_band = out_dataset.GetRasterBand(i + 1)
            out_band.WriteArray(color[:, :, i])

        # 创建一个空的 Matplotlib 图像
        fig, ax = plt.subplots(figsize=(6, 6))
        # 在图像上显示 ExG_color 数据
        img_ax = ax.imshow(color)
        # 添加彩色图例条
        cbar = plt.colorbar(img_ax, ax=ax)
        cbar.set_label('ExG')
        # 隐藏坐标轴
        ax.axis('off')
        # 保存图像
        plt.savefig(project + os.sep + 'axis_' + str(img), bbox_inches='tight')
        # 关闭数据集
        dataset = None
        out_dataset = None

    dst_dir = RESULT_xiaomaidaofu
    zip("static/result_xiaomaidaofu/index", dst_dir, "result_" + flag_name)


def shuidaodaofu(file_name):
    source = "data_flask/flask_shuidaodaofu"
    project = "static/result_shuidaodaofu/index"
    Path(source).mkdir(parents=True, exist_ok=True)
    Path(project).mkdir(parents=True, exist_ok=True)

    flag_name = file_name.split(os.sep)[-1].split(".")[0]
    update_dir(source)  # 每次清空
    unzip(file_name, "data_flask", "flask_shuidaodaofu")

    for img in os.listdir(source):
        # 读取原始图像的信息
        # img0 = cv2.imread(source + os.sep + img)  # 读取图像
        # img1 = cv2.resize(img0, fx=0.9, fy=0.9, dsize=None)  # 调整图像大小
        # img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 将图像转化为灰度图
        #
        # height = img1.shape[0]  # shape[0] 图像第一维度，高度
        # width = img1.shape[1]  # shape[1] 图像第二维度，宽度
        # plt.rcParams['font.family'] = 'SimHei'
        # img11 = img2[0:-1:10, 0:-1:10]
        # blur = cv2.blur(img11, (3, 3))  # 取3*3的矩阵 一般取奇数矩阵 均值滤波
        #
        # median = cv2.medianBlur(blur, 5)  # 中值滤波
        # median1 = np.array(median, dtype='uint8')
        # ret2, th2 = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # sc = plt.imshow(th2, 'gray')
        # sc.set_cmap('rainbow')  # 这里可以设置多种模式
        # plt.colorbar()  # 显示色度条
        # plt.xticks([]), plt.yticks([])
        #
        # save_path = project + os.sep + "result_" + str(img)
        # plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
        # plt.close()

        isCrop = False
        index = str(img).split('.')[1]
        if index == 'tif':
            isCrop = True
        # 打开原始图像文件
        dataset = gdal.Open(source + os.sep + img, gdal.GA_ReadOnly)
        if dataset is None:
            print("无法打开文件")
            exit(-1)
        # 获取原始图像的宽度和高度
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        # 定义裁剪的左上角和右下角坐标（这里是裁剪宽高各占原图的70%，即原图中间部分）
        left = int(width * 0.15)
        top = int(height * 0.15)
        right = int(width * 0.85)
        bottom = int(height * 0.85)
        # 计算裁剪后图像的宽度和高度
        new_width = right - left
        new_height = bottom - top
        if isCrop:
            # 读取原始图像数据
            data = dataset.ReadAsArray(left, top, new_width, new_height)
        else:
            data = dataset.ReadAsArray(0, 0, width, height)
        # 将图像转化为灰度图
        gray_data = 0.2989 * data[0] + 0.587 * data[1] + 0.114 * data[2]
        gray_data = np.clip(gray_data, 0, 255)  # 确保灰度值在合理范围内

        # 对灰度图进行下采样处理
        downsampled_data = gray_data[0:-1:10, 0:-1:10]
        # 应用均值滤波
        blurred_data = mean_filter(downsampled_data, 3)
        # 应用中值滤波
        blurred_data = median_filter(blurred_data, 5)
        # 将下采样后的图像上采样回原始尺寸
        upsampled_data = zoom(blurred_data, (10, 10), order=1)
        thresholded_data = threshold(upsampled_data)
        # 创建一个彩色映射
        cmap = plt.cm.rainbow  # 选择彩色映射，nipy_spectral
        color = (cmap(thresholded_data) * 255).astype(np.uint8)  # 将 ExG 值映射为 RGB 彩色空间

        # 创建输出图像
        driver = gdal.GetDriverByName("GTiff")
        out_dataset = driver.Create(project + os.sep + str(img), upsampled_data.shape[1], upsampled_data.shape[0], 3,
                                    gdal.GDT_Byte)
        if out_dataset is None:
            print("无法创建输出文件")
            exit(-1)

        # 将地理空间信息设置到输出图像中
        geotransform = list(dataset.GetGeoTransform())
        geotransform[0] = geotransform[0] + left * geotransform[1]
        geotransform[3] = geotransform[3] + top * geotransform[5]
        out_dataset.SetGeoTransform(tuple(geotransform))
        out_dataset.SetProjection(dataset.GetProjection())

        # 写入颜色数据到输出图像的对应波段
        for i in range(3):
            out_band = out_dataset.GetRasterBand(i + 1)
            out_band.WriteArray(color[:, :, i])

        # 创建一个空的 Matplotlib 图像
        fig, ax = plt.subplots(figsize=(6, 6))
        # 在图像上显示 ExG_color 数据
        img_ax = ax.imshow(color)
        # 添加彩色图例条
        cbar = plt.colorbar(img_ax, ax=ax)
        cbar.set_label('ExG')
        # 隐藏坐标轴
        ax.axis('off')
        # 保存图像
        plt.savefig(project + os.sep + 'axis_' + str(img), bbox_inches='tight')
        # 关闭数据集
        dataset = None
        out_dataset = None

    dst_dir = RESULT_shuidaodaofu
    zip("static/result_shuidaodaofu/index", dst_dir, "result_" + flag_name)


def xiaomaishifei(file_name):
    source = "data_flask/flask_xiaomaishifei"
    project = "static/result_xiaomaishifei/index"
    Path(source).mkdir(parents=True, exist_ok=True)
    Path(project).mkdir(parents=True, exist_ok=True)

    flag_name = file_name.split(os.sep)[-1].split(".")[0]
    update_dir(source)  # 每次清空
    unzip(file_name, "data_flask", "flask_xiaomaishifei")

    for img in os.listdir(source):
        # image = cv2.imread(source + os.sep + img, cv2.IMREAD_COLOR)
        # img1 = np.array(image, dtype='int')  # 转换成int型，不然会导致数据溢出
        # # 超绿灰度图
        # r, g, b = cv2.split(img1)
        # ExG = 2 * g - r - b
        # [m, n] = ExG.shape
        #
        # for i in range(m):
        #     for j in range(n):
        #         if ExG[i, j] < 0:
        #             ExG[i, j] = 0
        #         elif ExG[i, j] > 255:
        #             ExG[i, j] = 255
        # ExG1 = np.array(ExG, dtype='uint8')
        # MAXExg = 2
        # shifeizongliang = 100
        # proportion = 0.3
        # shifei2 = (270 - ExG1) / (MAXExg * shifeizongliang * proportion)
        # ExG11 = np.array(shifei2, dtype='uint8')  # 重新转换成uint8类型
        #
        # plt.subplot(132), plt.imshow(cv2.cvtColor(ExG11, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # exg_path = project + os.sep + "result_" + str(img)
        # plt.savefig(exg_path, dpi=800, bbox_inches='tight', pad_inches=0)
        # plt.close()
        # img2 = plt.imread(exg_path)
        # img_s2 = img2[:, :, 0]  # 直接读入的img为3通道，这里用直接赋值的方法转为单通道
        # sc2 = plt.imshow(ExG11)
        # sc2.set_cmap('nipy_spectral')  # 这里可以设置多种模式
        # plt.colorbar()  # 显示色度条
        #
        # plt.title('chufang')
        # plt.axis('off')
        # plt.savefig(exg_path, dpi=800, bbox_inches='tight', pad_inches=0.2)
        # plt.close()

        isCrop = False
        index = str(img).split('.')[1]
        if index == 'tif':
            isCrop = True
        # 打开原始图像文件
        dataset = gdal.Open(source + os.sep + img, gdal.GA_ReadOnly)
        if dataset is None:
            print("无法打开文件")
            exit(-1)
        # 获取原始图像的宽度和高度
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        # 定义裁剪的左上角和右下角坐标（这里是裁剪宽高各占原图的70%，即原图中间部分）
        left = int(width * 0.15)
        top = int(height * 0.15)
        right = int(width * 0.85)
        bottom = int(height * 0.85)
        # 计算裁剪后图像的宽度和高度
        new_width = right - left
        new_height = bottom - top
        if isCrop:
            # 读取原始图像数据
            data = dataset.ReadAsArray(left, top, new_width, new_height)
        else:
            data = dataset.ReadAsArray(0, 0, width, height)
        # 数据类型转换到浮点型并归一化到 0-255 范围内
        data = data.astype(np.float32) /255.0

        # 计算 ExG
        r, g, b = data[0], data[1], data[2]
        ExG = 2 * g - r - b
        # 将 ExG 值限制在 0 到 1 之间
        ExG = np.clip(ExG, 0, 1)
        MAXExg = 2
        shifeizongliang = 100
        proportion = 0.3
        shifei = (270 - ExG) / (MAXExg * shifeizongliang * proportion)
        # 创建一个彩色映射
        cmap = plt.cm.Greens  # 选择彩色映射，nipy_spectral
        ExG_color = (cmap(ExG) * 255).astype(np.uint8)  # 将 ExG 值映射为 RGB 彩色空间

        # 创建输出图像
        driver = gdal.GetDriverByName("GTiff")
        if isCrop:
            out_dataset = driver.Create(project + os.sep + str(img), new_width, new_height, 3, gdal.GDT_Byte)
        else:
            out_dataset = driver.Create(project + os.sep + str(img), width, height, 3, gdal.GDT_Byte)
        if out_dataset is None:
            print("无法创建输出文件")
            exit(-1)

        # 将地理空间信息设置到输出图像中
        geotransform = list(dataset.GetGeoTransform())
        geotransform[0] = geotransform[0] + left * geotransform[1]
        geotransform[3] = geotransform[3] + top * geotransform[5]
        out_dataset.SetGeoTransform(tuple(geotransform))
        out_dataset.SetProjection(dataset.GetProjection())

        # 写入颜色数据到输出图像的对应波段
        for i in range(3):
            out_band = out_dataset.GetRasterBand(i + 1)
            out_band.WriteArray(ExG_color[:, :, i])

        # 创建一个空的 Matplotlib 图像
        fig, ax = plt.subplots(figsize=(6, 6))
        # 在图像上显示 ExG_color 数据
        img_ax = ax.imshow(ExG_color)
        # 添加彩色图例条
        cbar = plt.colorbar(img_ax, ax=ax)
        cbar.set_label('ExG')
        # 隐藏坐标轴
        ax.axis('off')
        # 保存图像
        plt.savefig(project + os.sep + 'axis_' + str(img), bbox_inches='tight')
        # 关闭数据集
        dataset = None
        out_dataset = None

    dst_dir = RESULT_xiaomaishifei
    zip("static/result_xiaomaishifei/index", dst_dir, "result_" + flag_name)


def shuidaoshifei(file_name):
    source = "data_flask/flask_shuidaoshifei"
    project = "static/result_shuidaoshifei/index"
    Path(source).mkdir(parents=True, exist_ok=True)
    Path(project).mkdir(parents=True, exist_ok=True)

    flag_name = file_name.split(os.sep)[-1].split(".")[0]
    update_dir(source)  # 每次清空
    unzip(file_name, "data_flask", "flask_shuidaoshifei")

    for img in os.listdir(source):
        # image = cv2.imread(source + os.sep + img, cv2.IMREAD_COLOR)
        # img1 = np.array(image, dtype='int')  # 转换成int型，不然会导致数据溢出
        # # 超绿灰度图
        # r, g, b = cv2.split(img1)
        # # ExG_sub = cv2.subtract(2*g,r)
        # # ExG = cv2.subtract(ExG_sub,b )
        # ExG = 2 * g - r - b
        # # shifei = (MAXExg - Exg) / (MAXExg * shifeizongliang * proportion)
        # [m, n] = ExG.shape
        #
        # for i in range(m):
        #     for j in range(n):
        #         if ExG[i, j] < 0:
        #             ExG[i, j] = 0
        #         elif ExG[i, j] > 255:
        #             ExG[i, j] = 255
        # ExG1 = np.array(ExG, dtype='uint8')
        # MAXExg = 2
        # shifeizongliang = 100
        # proportion = 0.3
        # shifei2 = (270 - ExG1) / (MAXExg * shifeizongliang * proportion)
        # ExG11 = np.array(shifei2, dtype='uint8')  # 重新转换成uint8类型
        # # shifei2 = (MAXExg - ExG) / (MAXExg * shifeizongliang * proportion)
        # # ret2, th2 = cv2.threshold(ExG11, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #
        # # cv2.imshow('s',ExG11)
        # # cv2.waitKey(0)
        # plt.subplot(132), plt.imshow(cv2.cvtColor(ExG11, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # exg_path = project + os.sep + "result_" + str(img)
        # plt.savefig(exg_path, dpi=800, bbox_inches='tight', pad_inches=0)
        # plt.close()
        # img2 = plt.imread(exg_path)
        # img_s2 = img2[:, :, 0]  # 直接读入的img为3通道，这里用直接赋值的方法转为单通道
        # sc2 = plt.imshow(ExG11)
        # sc2.set_cmap('nipy_spectral')  # 这里可以设置多种模式
        # plt.colorbar()  # 显示色度条
        # # plt.rcParams['axes.unicode_minus'] = False
        # # plt.title(u'光谱指数模型')
        # plt.title('chufang')
        # plt.axis('off')
        # plt.savefig(exg_path, dpi=800, bbox_inches='tight', pad_inches=0.2)
        # plt.close()

        isCrop = False
        index = str(img).split('.')[1]
        if index == 'tif':
            isCrop = True
        # 打开原始图像文件
        dataset = gdal.Open(source + os.sep + img, gdal.GA_ReadOnly)
        if dataset is None:
            print("无法打开文件")
            exit(-1)
        # 获取原始图像的宽度和高度
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        # 定义裁剪的左上角和右下角坐标（这里是裁剪宽高各占原图的70%，即原图中间部分）
        left = int(width * 0.15)
        top = int(height * 0.15)
        right = int(width * 0.85)
        bottom = int(height * 0.85)
        # 计算裁剪后图像的宽度和高度
        new_width = right - left
        new_height = bottom - top
        if isCrop:
            # 读取原始图像数据
            data = dataset.ReadAsArray(left, top, new_width, new_height)
        else:
            data = dataset.ReadAsArray(0, 0, width, height)
        # 数据类型转换到浮点型并归一化到 0-255 范围内
        data = data.astype(np.float32) / 255.0

        # 计算 ExG
        r, g, b = data[0], data[1], data[2]
        ExG = 2 * g - r - b
        # 将 ExG 值限制在 0 到 1 之间
        ExG = np.clip(ExG, 0, 1)
        MAXExg = 2
        shifeizongliang = 100
        proportion = 0.3
        shifei = (270 - ExG) / (MAXExg * shifeizongliang * proportion)
        # 创建一个彩色映射
        cmap = plt.cm.Greens  # 选择彩色映射，nipy_spectral
        ExG_color = (cmap(ExG) * 255).astype(np.uint8)  # 将 ExG 值映射为 RGB 彩色空间

        # 创建输出图像
        driver = gdal.GetDriverByName("GTiff")
        if isCrop:
            out_dataset = driver.Create(project + os.sep + str(img), new_width, new_height, 3, gdal.GDT_Byte)
        else:
            out_dataset = driver.Create(project + os.sep + str(img), width, height, 3, gdal.GDT_Byte)
        if out_dataset is None:
            print("无法创建输出文件")
            exit(-1)

        # 将地理空间信息设置到输出图像中
        geotransform = list(dataset.GetGeoTransform())
        geotransform[0] = geotransform[0] + left * geotransform[1]
        geotransform[3] = geotransform[3] + top * geotransform[5]
        out_dataset.SetGeoTransform(tuple(geotransform))
        out_dataset.SetProjection(dataset.GetProjection())

        # 写入颜色数据到输出图像的对应波段
        for i in range(3):
            out_band = out_dataset.GetRasterBand(i + 1)
            out_band.WriteArray(ExG_color[:, :, i])

        # 创建一个空的 Matplotlib 图像
        fig, ax = plt.subplots(figsize=(6, 6))
        # 在图像上显示 ExG_color 数据
        img_ax = ax.imshow(ExG_color)
        # 添加彩色图例条
        cbar = plt.colorbar(img_ax, ax=ax)
        cbar.set_label('ExG')
        # 隐藏坐标轴
        ax.axis('off')
        # 保存图像
        plt.savefig(project + os.sep + 'axis_' + str(img), bbox_inches='tight')
        # 关闭数据集
        dataset = None
        out_dataset = None

    dst_dir = RESULT_shuidaoshifei
    zip("static/result_shuidaoshifei/index", dst_dir, "result_" + flag_name)


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

    dst_dir = RESULT_SPIKE
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

    dst_dir = RESULT_RICE
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

    dst_dir = RESULT_SEEDLING
    zip("static/result_seedling/detect", dst_dir, "result_" + flag_name)


def phe(file_name):
    pass


def do_request(file_name, flag):
    if flag == "01":
        LAI(file_name)
    elif flag == "02":
        pass
    elif flag == "03":
        EXG(file_name)
    elif flag == "04":
        xiaomaidaofu(file_name)
    elif flag == "05":
        shuidaodaofu(file_name)
    elif flag == "06":
        xiaomaishifei(file_name)
    elif flag == "07":
        shuidaoshifei(file_name)
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
    check_for_changes(CHECK_FILE)
