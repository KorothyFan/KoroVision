# KoroVision Flask 推理服务部署指南

本目录演示如何将 **KoroVision** 浏览器端训练平台导出的 TensorFlow.js 模型（`korovision-model.json` / `zumiao-model.json`、`*.weights.bin`、`metadata.json`）部署为标准的 HTTP 推理 API，供小程序、App 或工作流服务（如 KoroFlow）调用。

## 文件说明

| 文件 | 用途 |
| :--- | :--- |
| `server.py` | Flask 推理服务（含 `/health` 与 `/predict` 接口） |
| `flask_app.py` | PythonAnywhere 等平台的 WSGI 入口 |
| `requirements.txt` | Python 依赖列表 |

> 默认模型解析顺序：`KOROVISION_MODEL_DIR`（兼容 `ZUMIAO_MODEL_DIR`）环境变量 → 当前目录下的 `./model/` → 仓库根目录的 `../../demo-model/`；同时会自动匹配 `korovision-model.json`、`zumiao-model.json` 及对应 `.weights.bin`。

---

## 一、本地快速启动

### 1. 安装依赖

```bash
cd examples/flask-server
pip install -r requirements.txt
```

### 2. （可选）替换为你自己导出的模型

将网页端导出的 3 个文件（`.json` + `.weights.bin` + `metadata.json`）放入 `examples/flask-server/model/`，或通过环境变量指定目录：

```bash
export KOROVISION_MODEL_DIR=/path/to/your-model
```

### 3. 启动服务并测试

```bash
python server.py
```

健康检查与图片推理测试：

```bash
# 1. 检查服务与已加载类别
curl http://127.0.0.1:5000/health

# 2. 上传图片进行推理
curl -X POST -F "file=@/path/to/test.jpg" http://127.0.0.1:5000/predict
```

返回示例：

```json
{
  "predicted": "独角狮",
  "confidence": 0.964,
  "isConfident": true,
  "topK": [
    { "name": "独角狮", "confidence": 0.964 },
    { "name": "其他狮子", "confidence": 0.028 },
    { "name": "其他物品", "confidence": 0.008 }
  ]
}
```

---

## 二、云端部署示例（PythonAnywhere）

1. 注册 PythonAnywhere 免费账号（Beginner Plan）。
2. 在 home 目录下新建 `KoroVision-Server/`，上传 `server.py`、`flask_app.py`、`requirements.txt`，并将模型文件（`korovision-model.json`、`korovision-model.weights.bin`、`metadata.json`）放入 `KoroVision-Server/model/`。
3. 打开 Bash Console 安装依赖：
   ```bash
   pip install --user -r ~/KoroVision-Server/requirements.txt
   ```
4. 在 **Web** 标签页创建 Flask（Python 3.10）Web App。
5. 编辑 WSGI configuration file，将 `flask_app.py` 内容粘贴进去，并把 `YOUR_USERNAME` 替换为你的用户名。
6. 点击绿色 **Reload** 按钮（首次加载需要下载 ImageNet MobileNet V2 权重，约 30 秒），访问 `https://<你的用户名>.pythonanywhere.com/health` 验证即可。

---

## 三、技术细节对齐说明

- **图像预处理一致性**：`server.py` 中的 `preprocess_pil_image()` 与前端 `index.html` 的 `imgToTensor()` 保持完全一致——先取短边做**中心正方形裁剪**，再双线性缩放到 `224×224`，最后归一化到 `[-1, 1]`。
- **TF.js 权重直读**：`_load_tfjs_weights()` 按 `weightsManifest[0].weights` 顺序从 `.weights.bin` 解析 `float32` 张量并直接注入 Keras `Sequential` 分类头，无需额外执行 `tensorflowjs_converter` 转换。
