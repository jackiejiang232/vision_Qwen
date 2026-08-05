# 机器人相机实时检测与官方镜像部署

本章说明如何把已经能够处理本地图片的 GroundingDINO + SAM 模块接入比赛机器人的 ROS 2 相机，并在官方镜像环境中运行实时推理节点。

## 1. 已确认的官方环境

官方机器人相机接口：

```text
ROS 版本：ROS 2 Humble
图像话题：/head_camera/color/image_raw
消息类型：sensor_msgs/msg/Image
OpenCV 转换：CvBridge().imgmsg_to_cv2(message, desired_encoding="bgr8")
```

官方镜像实测环境：

```text
Python：3.10
PyTorch：2.7.1+cu118
torchvision：0.22.1+cu118
CUDA Toolkit：11.8
GPU：NVIDIA GeForce RTX 4090 D
```

官方镜像已经包含：

- `rclpy`
- `cv_bridge`
- `sensor_msgs`
- `torch`
- `torchvision`
- `opencv-python`

官方镜像没有包含：

- `groundingdino`
- `segment_anything`
- `transformers`

因此，仅把 `grounded_sam.py`、模型源码和权重挂载进官方镜像不能直接运行。宿主机 `vision` Conda 环境也不会自动进入容器。

推荐结构：

```text
官方服务端容器
  ├── 场景
  ├── 机器人
  ├── 相机发布器
  └── 裁判系统
          │
          │ ROS 2 DDS（host 网络）
          ▼
视觉客户端容器
  ├── GroundingDINO
  ├── SAM
  ├── ROS 2 相机订阅节点
  ├── 检测结果发布器
  └── 标注图像发布器
```

两个容器必须使用相同的 `ROS_DOMAIN_ID` 和 `RMW_IMPLEMENTATION`。

## 2. 不能复用宿主机 GroundingDINO 扩展

宿主机 `vision` 环境使用：

```text
PyTorch 2.5.1+cu121
CUDA 12.1
```

官方镜像使用：

```text
PyTorch 2.7.1+cu118
CUDA 11.8
```

GroundingDINO 的 `groundingdino/_C*.so` 是与 PyTorch 和 CUDA ABI 绑定的动态库。宿主机编译出的文件不能放进官方镜像使用，必须在基于官方镜像构建的客户端镜像中重新编译。

## 3. 修复 Torch 2.7 编译兼容问题

打开：

```text
/media/jiangzhenmin/系统/JZM/GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu
```

只修改第 65、135 行附近的两处：

```cpp
AT_DISPATCH_FLOATING_TYPES(value.type(),
```

改为：

```cpp
AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),
```

不要全局替换 `value.type()`。以下 CUDA 判断不能改成 `value.scalar_type().is_cuda()`：

```cpp
value.type().is_cuda()
```

实测结果：只精确修改两个 `AT_DISPATCH_FLOATING_TYPES` 调度参数后，可以在官方镜像的 Torch 2.7.1、CUDA 11.8 环境中成功编译 `groundingdino._C`。

## 4. 创建派生视觉客户端镜像

创建：

```text
/media/jiangzhenmin/系统/JZM/Vision/Dockerfile.vision
```

内容：

```dockerfile
FROM crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/material_sorting:latest

ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST=8.9
ENV MAX_JOBS=4

RUN python3 -m pip install --no-cache-dir \
    transformers==4.30.2 \
    tokenizers==0.13.3 \
    addict==2.4.0 \
    yapf \
    timm \
    supervision==0.29.1 \
    pycocotools

WORKDIR /media/jiangzhenmin/系统/JZM

COPY GroundingDINO ./GroundingDINO
COPY segment-anything ./segment-anything

RUN rm -rf GroundingDINO/build \
    GroundingDINO/*.egg-info \
    GroundingDINO/groundingdino/_C*.so && \
    python3 -m pip install -e GroundingDINO \
        --no-build-isolation --no-deps && \
    python3 -m pip install -e segment-anything --no-deps
```

注意：

1. 不要在派生镜像中重新安装 Torch 或 torchvision。
2. 必须删除复制进来的宿主机 `_C*.so`。
3. `TORCH_CUDA_ARCH_LIST=8.9` 对应 RTX 4090。
4. Transformers 固定为 4.30.2。5.x 已移除 GroundingDINO 使用的 BERT 接口。
5. Dockerfile 没有覆盖官方镜像的 ENTRYPOINT，ROS 环境仍由官方入口加载。

构建镜像：

```bash
cd /media/jiangzhenmin/系统/JZM

docker build \
  -f Vision/Dockerfile.vision \
  -t challengecup-vision:latest \
  .
```

由于构建上下文包含模型和权重，镜像会比较大。初次验证建议先使用完整复制方式，确认流程后再通过 `.dockerignore` 排除大权重并改为运行时只读挂载。

## 5. 修改 grounded_sam.py 的输入模块

原来的本地图片导入：

```python
from groundingdino.util.inference import load_image, load_model, predict
```

替换为：

```python
import rclpy
from cv_bridge import CvBridge
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict
```

增加相机帧预处理。它代替 `load_image(image_path)`：

```python
DINO_TRANSFORM = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


def preprocess_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_pil = PILImage.fromarray(frame_rgb)
    image_tensor, _ = DINO_TRANSFORM(image_pil, None)
    return image_tensor
```

把原来接收 `image_path` 的函数替换为：

```python
def run_grounding_dino(
    model,
    frame_bgr,
    text_prompt,
    box_threshold,
    text_threshold,
    device,
):
    image_tensor = preprocess_frame(frame_bgr)

    return predict(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )
```

以下逻辑可以继续复用：

- `load_models()`
- `convert_boxes_to_xyxy()`
- `run_sam()`
- `create_visualization()`

本地图片专用的 `read_image()`、单次 `save_results()` 和原 `main()` 不再作为实时节点入口。

## 6. ROS 2 节点设计

建议单独创建：

```text
/home/jiangzhenmin/Desktop/挑战杯/robot_test_client/grounded_sam_camera_node.py
```

不要在相机回调中直接运行模型。正确职责划分：

```text
on_image()
  └── 只把最新 ROS Image 转成 BGR numpy 图像

定时器 on_inference()
  ├── 复制最新帧
  ├── GroundingDINO 检测
  ├── 坐标从 cxcywh 转成 xyxy
  ├── SAM 分割
  ├── 发布 JSON
  └── 发布标注图像
```

相机订阅必须使用传感器 QoS：

```python
self.create_subscription(
    Image,
    "/head_camera/color/image_raw",
    self.on_image,
    qos_profile_sensor_data,
)
```

相机回调：

```python
def on_image(self, message):
    self.latest_frame = self.bridge.imgmsg_to_cv2(
        message,
        desired_encoding="bgr8",
    ).copy()
    self.latest_header = message.header
```

定时器建议从每秒一次开始：

```python
self.create_timer(1.0, self.on_inference)
```

发布两个结果话题：

```python
self.result_pub = self.create_publisher(
    String,
    "/grounded_sam/detections",
    10,
)

self.annotated_pub = self.create_publisher(
    Image,
    "/grounded_sam/annotated",
    10,
)
```

模型必须在节点初始化时加载一次，不能每收到一帧重新加载。

## 7. 启动前检查相机

先启动官方服务端容器。服务端启动命令保持官方教程原样，但确认设置：

```bash
-e ROS_DOMAIN_ID=99
-e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

检查相机话题：

```bash
ros2 topic list -t
ros2 topic info /head_camera/color/image_raw -v
ros2 topic hz /head_camera/color/image_raw
```

预期消息类型：

```text
sensor_msgs/msg/Image
```

如果看不到话题，首先检查：

1. 官方服务端容器是否仍在运行。
2. 服务端和客户端的 `ROS_DOMAIN_ID` 是否相同。
3. 两端是否都使用 `rmw_cyclonedds_cpp`。
4. 是否都使用 `--network host`。
5. 相机话题名是否被比赛镜像版本修改。

## 8. 启动视觉客户端

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v /home/jiangzhenmin/Desktop/挑战杯/robot_test_client:/client:ro \
  challengecup-vision:latest \
  python3 /client/grounded_sam_camera_node.py \
  --image-topic /head_camera/color/image_raw \
  --infer-period 1.0 \
  --no-display
```

建议直接让官方 ENTRYPOINT 执行 `python3`。如果使用 `bash -lc`，登录 Shell 可能丢失 ROS 的 `PYTHONPATH`，必须先执行：

```bash
source /opt/ros/humble/setup.bash
```

## 9. 检查输出

查看检测 JSON：

```bash
ros2 topic echo /grounded_sam/detections
```

查看标注图像：

```bash
rqt_image_view /grounded_sam/annotated
```

检测 JSON 建议至少包含：

```json
{
  "label": "pink box",
  "score": 0.86,
  "box_xyxy": [120.0, 80.0, 340.0, 290.0],
  "mask_area": 18324,
  "centroid_uv": [228.4, 184.7]
}
```

`centroid_uv` 是 RGB 图像中的二维像素坐标，不是机器人坐标。若要用于机械臂抓取，还需要同步深度图、相机内参和相机到机器人基座的外参。

## 10. 实时性能注意事项

SAM ViT-H 与 GroundingDINO 都是较大的模型，不能等同于相机原始帧率实时运行。

建议顺序：

1. 先只订阅并显示相机，验证 ROS 通信。
2. 暂时关闭 SAM，只运行 GroundingDINO。
3. 确认检测框正确后启用 SAM。
4. 初始 `infer_period` 使用 1.0 秒。
5. 只处理最新帧，不建立无限图像队列。
6. 使用 `torch.no_grad()`。
7. 不要每帧调用 `torch.cuda.empty_cache()`。
8. 不要在每帧中重新加载权重。
9. 容器无显示器时使用 `--no-display`，通过 ROS 图像话题查看结果。
10. 后续需要更高帧率时考虑 SAM ViT-B、MobileSAM 或只对抓取候选框运行 SAM。

## 11. 常见问题

### ModuleNotFoundError: rclpy

说明 ROS 环境没有加载。执行：

```bash
source /opt/ros/humble/setup.bash
```

或者不要通过会清除环境的登录 Shell 启动，直接使用镜像 ENTRYPOINT 执行 Python。

### ModuleNotFoundError: groundingdino

派生镜像没有成功安装 GroundingDINO，或运行的仍是原始官方镜像。检查：

```bash
python3 -c "import groundingdino; print(groundingdino.__file__)"
```

### ImportError 或 undefined symbol，指向 groundingdino._C

说明加载了宿主机 CUDA 12.1 / Torch 2.5 编译出的扩展。删除复制进镜像的 `_C*.so`，在官方镜像的 CUDA 11.8 / Torch 2.7.1 环境重新编译。

### 编译出现 DeprecatedTypeProperties 转 ScalarType 错误

确认只将两个：

```cpp
AT_DISPATCH_FLOATING_TYPES(value.type(),
```

改成：

```cpp
AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),
```

### 有节点但收不到图像

优先检查 QoS。相机订阅使用 `qos_profile_sensor_data`，并确认 ROS Domain ID、RMW 和 host 网络一致。

### 容器中 OpenCV 窗口报错

容器没有 X11 显示环境。使用 `--no-display`，将标注结果发布到 `/grounded_sam/annotated`。

## 12. 比赛提交注意事项

本地开发使用派生视觉镜像是可行的，但如果比赛最终只允许提交一个 `client.py`，还需要向主办方确认：

1. 是否允许提交自定义 Docker 镜像。
2. 是否允许提交 GroundingDINO、SAM 和 BERT 权重。
3. 是否限制镜像大小。
4. 评测环境是否允许联网下载模型。
5. 评测 GPU、CUDA 和 PyTorch 版本是否与当前官方镜像一致。
6. 是否只能使用官方镜像预装依赖。

官方镜像当前没有 GroundingDINO、SAM 和 Transformers，因此一个不附带依赖和权重的 `client.py` 无法独立完成该视觉方案。
