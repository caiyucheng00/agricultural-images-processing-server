from datetime import timedelta, datetime
import time

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

from FlaskServerUtils import *

app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


@app.route('/')
def hello_world():
    return render_template('spike.html')


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
        #                 "data": "/usr/local/webserver/spike-detection-server/static/detect/" + file.filename})
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
                         "data": "/usr/local/webserver/spike-detection-server/static/detect/" + file.filename}})
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
