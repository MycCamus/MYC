from flask import Flask, render_template
from oss2 import Auth, Bucket
import os
from flask_jwt_extended import JWTManager, create_access_token, jwt_required

app = Flask(__name__)

# 阿里云OSS配置（需要替换为您的实际信息）
OSS_ACCESS_KEY_ID = os.getenv('OSS_KEY_ID', 'your_access_key_id')
OSS_ACCESS_KEY_SECRET = os.getenv('OSS_KEY_SECRET', 'your_access_key_secret')
OSS_ENDPOINT = 'oss-cn-hangzhou.aliyuncs.com'  # 根据实际地域修改
BUCKET_NAME = 'your_bucket_name'

# 初始化OSS客户端
auth = Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
bucket = Bucket(auth, OSS_ENDPOINT, BUCKET_NAME)


app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET', 'your_super_secret')
jwt = JWTManager(app)

@app.route('/login', methods=['POST'])
def login():
    # 简单演示用，实际应接入数据库
    username = request.json.get('username')
    password = request.json.get('password')
    if username == "admin" and password == "admin123":
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token)
    return jsonify({"msg": "Bad credentials"}), 401

@app.route('/')
@jwt_required(optional=True)  # 允许匿名访问
def show_images():
    # 原有逻辑不变，可扩展不同用户权限
    # 获取带元数据的文件列表
    files = bucket.list_objects()

    # 按最后修改时间分类
    image_dict = defaultdict(list)
    for obj in files:
        if obj.key.lower().endswith(('.png', '.jpg', '.jpeg')):
            mod_date = datetime.fromtimestamp(obj.last_modified)
            date_key = mod_date.strftime("%Y-%m-%d")
            image_dict[date_key].append(bucket.sign_url('GET', obj.key, 3600))

    return render_template('gallery.html', images=image_dict)

    # 生成带签名的临时访问URL（有效期3600秒）
    image_urls = [bucket.sign_url('GET', img, 3600) for img in image_files]

    return render_template('gallery.html', images=image_urls)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)