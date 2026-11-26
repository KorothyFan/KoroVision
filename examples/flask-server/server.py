# -*- coding: utf-8 -*-
"""
KoroVision 推理服务器
━━━━━━━━━━━━━━━━━━━━
将 KoroVision 浏览器端导出的 TensorFlow.js 分类头（.json + .weights.bin）
加载到 Python Flask + Keras 服务中，提供统一的 HTTP 推理接口。

本地运行: python server.py
云端部署: 见 flask_app.py (PythonAnywhere WSGI 入口) 与 DEPLOY.md
"""

import glob
import json
import os
import socket

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

# ═══════════════════════════════════════
#  配置
# ═══════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CATEGORIES = ['独角狮', '其他狮子', '其他物品']
PORT = int(os.environ.get('PORT', '5000'))
CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.7'))
IMG_SIZE = 224


def find_model_files(model_dir):
    """自动发现目录中的 TF.js 模型 JSON 与对应权重 .bin 文件。"""
    if not os.path.isdir(model_dir):
        return None, None

    json_candidates = [
        os.path.join(model_dir, 'korovision-model.json'),
        os.path.join(model_dir, 'zumiao-model.json'),
    ]
    for path in sorted(glob.glob(os.path.join(model_dir, '*.json'))):
        if os.path.basename(path) != 'metadata.json' and path not in json_candidates:
            json_candidates.append(path)

    json_path = next((p for p in json_candidates if os.path.isfile(p)), None)
    if not json_path:
        return None, None

    bin_path = None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            spec = json.load(f)
        manifest = spec.get('weightsManifest') or []
        if manifest and manifest[0].get('paths'):
            rel_bin = os.path.basename(manifest[0]['paths'][0])
            candidate = os.path.join(model_dir, rel_bin)
            if os.path.isfile(candidate):
                bin_path = candidate
    except Exception:
        pass

    if not bin_path:
        for name in ('korovision-model.weights.bin', 'zumiao-model.weights.bin'):
            candidate = os.path.join(model_dir, name)
            if os.path.isfile(candidate):
                bin_path = candidate
                break

    if not bin_path:
        bins = sorted(glob.glob(os.path.join(model_dir, '*.bin')))
        bin_path = bins[0] if bins else None

    return json_path, bin_path


def resolve_model_dir():
    """按环境变量 -> 本地 ./model -> 仓库内置 ../../demo-model 顺序解析模型目录。"""
    env_dir = os.environ.get('KOROVISION_MODEL_DIR') or os.environ.get('ZUMIAO_MODEL_DIR')
    if env_dir:
        return os.path.abspath(env_dir)

    local_model_dir = os.path.join(BASE_DIR, 'model')
    if all(find_model_files(local_model_dir)):
        return local_model_dir

    repo_demo_dir = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'demo-model'))
    if all(find_model_files(repo_demo_dir)):
        return repo_demo_dir

    return local_model_dir


def load_categories(model_dir):
    """从 metadata.json 读取类别列表，缺省时回退到默认三分类。"""
    meta_path = os.path.join(model_dir, 'metadata.json')
    if os.path.isfile(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        classes = meta.get('classes')
        if isinstance(classes, list) and classes:
            return classes
    return list(DEFAULT_CATEGORIES)


MODEL_DIR = resolve_model_dir()
CATEGORIES = load_categories(MODEL_DIR)

# 模型变量（由 load_model 初始化）
mobilenet = None
classifier = None
_model_ready = False

# ═══════════════════════════════════════
#  Flask 应用（WSGI 入口在 flask_app.py）
# ═══════════════════════════════════════
app = Flask(__name__)
CORS(app)


def preprocess_pil_image(img, size=IMG_SIZE):
    """对齐前端训练平台的预处理：RGB -> 中心正方形裁剪 -> 双线性缩放到 224x224 -> [-1, 1]。"""
    rgb = img.convert('RGB')
    w, h = rgb.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    cropped = rgb.crop((left, top, left + side, top + side))
    resized = cropped.resize((size, size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)
    # 与 tensorflow.keras.applications.mobilenet_v2.preprocess_input 一致
    return (arr / 127.5) - 1.0


# ═══════════════════════════════════════
#  模型加载
# ═══════════════════════════════════════
def _load_tfjs_weights(model, json_path, bin_path):
    """从 TF.js 格式加载权重到 Keras 模型。"""
    with open(json_path, 'r', encoding='utf-8') as f:
        model_json = json.load(f)

    weight_specs = model_json['weightsManifest'][0]['weights']

    with open(bin_path, 'rb') as f:
        buffer = f.read()

    weights = []
    offset = 0
    for spec in weight_specs:
        shape = spec['shape']
        n = 1
        for d in shape:
            n *= d
        arr = np.frombuffer(buffer, dtype=np.float32, count=n, offset=offset)
        arr = arr.reshape(shape).copy()
        weights.append(arr)
        offset += n * 4

    model.set_weights(weights)
    return weights


def load_model():
    """加载 MobileNet V2 特征提取器 + TF.js 导出的分类头。"""
    global mobilenet, classifier, _model_ready, MODEL_DIR, CATEGORIES
    if _model_ready:
        return

    MODEL_DIR = resolve_model_dir()
    CATEGORIES = load_categories(MODEL_DIR)

    json_path, bin_path = find_model_files(MODEL_DIR)
    if not json_path or not bin_path:
        raise FileNotFoundError(
            f'未在 {MODEL_DIR} 找到完整的 TF.js 模型文件（.json + .bin）。请设置 KOROVISION_MODEL_DIR 或放置到 ./model/'
        )

    with open(json_path, 'r', encoding='utf-8') as f:
        model_json = json.load(f)
    weight_specs = model_json['weightsManifest'][0]['weights']
    hidden_units = weight_specs[0]['shape'][1]
    num_classes = weight_specs[-1]['shape'][0]

    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.models import Sequential

    print('[Server] Loading MobileNet V2 (feature extractor)...')
    mobilenet = MobileNetV2(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    mobilenet.trainable = False
    print('[Server] MobileNet V2 loaded (1280-dim embedding)')

    print(f'[Server] Loading classifier head from {json_path}...')
    classifier = Sequential([
        Dense(hidden_units, activation='relu', input_shape=(1280,)),
        Dropout(0.3),
        Dense(num_classes, activation='softmax'),
    ])
    _load_tfjs_weights(classifier, json_path, bin_path)

    if len(CATEGORIES) != num_classes:
        CATEGORIES = [f'类别 {i + 1}' for i in range(num_classes)]

    # 预热
    dummy = np.zeros((1, 1280), dtype=np.float32)
    classifier.predict(dummy, verbose=0)
    print(f'[Server] Model ready! Classes: {CATEGORIES}')
    _model_ready = True


# ═══════════════════════════════════════
#  Flask 路由
# ═══════════════════════════════════════
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400

    try:
        if not _model_ready:
            load_model()

        file = request.files['file']
        img = Image.open(file.stream)
        img_array = preprocess_pil_image(img, size=IMG_SIZE)
        img_batch = np.expand_dims(img_array, axis=0)

        embedding = mobilenet.predict(img_batch, verbose=0)
        probs = classifier.predict(embedding, verbose=0)[0]

        results = [
            {'name': CATEGORIES[i], 'confidence': float(p)}
            for i, p in enumerate(probs)
        ]
        results.sort(key=lambda x: x['confidence'], reverse=True)

        return jsonify({
            'predicted': results[0]['name'],
            'confidence': float(results[0]['confidence']),
            'isConfident': bool(results[0]['confidence'] >= CONFIDENCE_THRESHOLD),
            'topK': results,
        })
    except Exception as e:
        print(f'[Server] Predict error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'service': 'KoroVision',
        'categories': CATEGORIES,
        'model_dir': MODEL_DIR,
        'model_loaded': _model_ready,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
    })


# ═══════════════════════════════════════
#  本地启动
# ═══════════════════════════════════════
def get_local_ip():
    """获取本机在局域网中的 IP。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'


if __name__ == '__main__':
    load_model()
    local_ip = get_local_ip()
    print()
    print('=' * 50)
    print('  KoroVision 推理服务器已启动')
    print(f'  模型目录:  {MODEL_DIR}')
    print(f'  识别类别:  {CATEGORIES}')
    print(f'  本机地址:  http://127.0.0.1:{PORT}')
    print(f'  局域网IP:  http://{local_ip}:{PORT}')
    print('=' * 50)
    print()
    app.run(host='0.0.0.0', port=PORT, debug=False)
