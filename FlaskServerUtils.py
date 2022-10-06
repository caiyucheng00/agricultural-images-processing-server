#
# @Description:
# @Author:虚幻的元亨利贞
# @Time: 2022-04-24 21:17
#
import shutil
from urllib import request as url_request
from detect import *

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
    os.mkdir('./data_flask/images')


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
