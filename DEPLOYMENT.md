# sam2_SaaS 部署文档

本文档面向这份仓库里的本地改造版本，重点是 `sam2_segment_api.py` 提供的 FastAPI 服务，以及可选的 `gradio_video_segment.py` 前端。

## 1. 部署前提

- 操作系统建议：Ubuntu 22.04 或更新版本
- Python：3.10 及以上
- GPU：NVIDIA GPU，建议显存 16GB 以上
- CUDA / PyTorch：与当前环境中的 `torch>=2.5.1` 匹配
- 这份仓库不会提交模型权重到 GitHub，因为 GitHub 单文件大小限制为 100MB

未提交的大文件：

- `checkpoints/sam2.1_hiera_large.pt`
- `BiRefNet/model.safetensors`

部署时需要你自行下载或拷贝到服务器。

## 2. 拉取代码并安装依赖

```bash
git clone <your-private-repo-url> sam2_SaaS
cd sam2_SaaS

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
pip install -r requirements.api.txt
```

如果服务器没有可用 `nvcc`，又不需要编译 SAM2 的 CUDA 扩展，可以这样安装：

```bash
SAM2_BUILD_CUDA=0 pip install -e .
```

## 3. 准备模型文件

### SAM2 权重

当前 API 默认读取：

- `checkpoints/sam2.1_hiera_large.pt`
- `configs/sam2.1/sam2.1_hiera_l.yaml`

可直接在仓库根目录执行：

```bash
mkdir -p checkpoints
cd checkpoints
./download_ckpts.sh
cd ..
```

### BiRefNet 权重

当前 API 需要本地目录 `./BiRefNet` 中存在 Hugging Face 导出的模型文件，至少包括：

- `config.json`
- `configuration.json`
- `birefnet.py`
- `BiRefNet_config.py`
- `model.safetensors`

本仓库已经保留了代码和配置文件，但 `model.safetensors` 已被 `.gitignore` 排除，需要你在部署机手动放入 `BiRefNet/` 目录。

## 4. 启动 API

### 方式 A：直接启动

```bash
source .venv/bin/activate
python sam2_segment_api.py
```

默认监听 `0.0.0.0:8000`，文档地址为：

```text
http://<server-ip>:8000/docs
```

### 方式 B：使用带滚动重启的启动脚本

这个方式更适合线上常驻服务，脚本会每 12 小时滚动重启一次 uvicorn 子进程，减少长时间运行后的显存累积风险。

```bash
source .venv/bin/activate
PYTHON_BIN="$(which python)" ./sam2_segment_api.sh
```

支持的环境变量：

- `PYTHON_BIN`：Python 可执行文件路径
- `SAM2_API_HOST`：监听地址，默认 `0.0.0.0`
- `SAM2_API_PORT`：监听端口，默认 `8000`
- `SAM2_API_RESTART_INTERVAL_SECONDS`：滚动重启间隔，默认 `43200`
- `SAM2_API_RESTART_READY_TIMEOUT_SECONDS`：新进程 ready 超时，默认 `180`
- `SAM2_API_OLD_PROCESS_GRACEFUL_SECONDS`：旧进程优雅退出等待时间，默认 `3600`
- `SAM2_API_TIMEOUT_GRACEFUL_SHUTDOWN`：uvicorn 优雅关停超时，默认 `3600`
- `SAM2_CHECKPOINT`：SAM2 权重路径
- `SAM2_MODEL_CFG`：SAM2 配置文件路径
- `BIREFNET_MODEL_DIR`：BiRefNet 模型目录，默认 `./BiRefNet`

示例：

```bash
source .venv/bin/activate
export PYTHON_BIN="$(which python)"
export SAM2_API_PORT=9000
export SAM2_CHECKPOINT="checkpoints/sam2.1_hiera_small.pt"
export SAM2_MODEL_CFG="configs/sam2.1/sam2.1_hiera_s.yaml"
./sam2_segment_api.sh
```

## 5. 使用 systemd 托管

新建 `/etc/systemd/system/sam2-api.service`：

```ini
[Unit]
Description=SAM2 FastAPI Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/sam2_SaaS
Environment=PYTHON_BIN=/opt/sam2_SaaS/.venv/bin/python
Environment=SAM2_API_HOST=0.0.0.0
Environment=SAM2_API_PORT=8000
Environment=SAM2_CHECKPOINT=/opt/sam2_SaaS/checkpoints/sam2.1_hiera_large.pt
Environment=SAM2_MODEL_CFG=configs/sam2.1/sam2.1_hiera_l.yaml
Environment=BIREFNET_MODEL_DIR=/opt/sam2_SaaS/BiRefNet
ExecStart=/opt/sam2_SaaS/sam2_segment_api.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

加载并启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable sam2-api
sudo systemctl start sam2-api
sudo systemctl status sam2-api
```

## 6. Nginx 反向代理

如果要通过域名访问，可加一层 Nginx：

```nginx
server {
    listen 80;
    server_name your-domain.example.com;

    client_max_body_size 200m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 7. 可选 Gradio 页面

如果你想要一个简单的交互标注页面：

```bash
source .venv/bin/activate
python gradio_video_segment.py
```

默认访问地址：

```text
http://<server-ip>:7860
```

它会调用同机的 `http://localhost:8000/segment`。

## 8. 接口说明

接口调用说明见：

- `sam2_segment_api.md`

在线自检可访问：

- `/docs`
- `/openapi.json`

## 9. 常见问题

### 1. 推送到 GitHub 失败，提示文件过大

原因通常是误把以下文件加入 Git：

- `checkpoints/*.pt`
- `BiRefNet/model.safetensors`
- 生成结果目录，例如 `notebooks/videos/test/`

这些文件已经在 `.gitignore` 中排除，不要强行提交。

### 2. 启动后显存长期增长

优先使用 `./sam2_segment_api.sh`，它会定期滚动重启 uvicorn 子进程。

### 3. 想换更小模型

把环境变量切换到 small/tiny 对应的 checkpoint 和 config 即可，无需改代码。
