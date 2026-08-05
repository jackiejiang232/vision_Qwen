# GroundingDINO + SAM

本项目将 **GroundingDINO** 与 **Segment Anything Model (SAM)** 结合，实现：

- 文本提示目标检测（Open-Vocabulary Detection）
- 自动目标实例分割（Instance Segmentation）
- 检测结果可视化
- Mask 单独保存
- 检测信息保存为 JSON

整个流程如下：

```
输入图片
      │
      ▼
GroundingDINO
（文本目标检测）
      │
      ▼
检测框(Box)
      │
      ▼
SAM
（根据检测框分割）
      │
      ▼
Mask
      │
      ▼
结果可视化
```

---

# 项目目录

```
Vision/
│
├── grounded_sam.py                # 主程序
│
├── output/                        # 输出目录
│   ├── grounded_sam_result.png
│   ├── detections.json
│   ├── mask_000.png
│   ├── mask_001.png
│   └── ...
│
├── GroundingDINO/
│
└── segment-anything/
```

---

# 模型路径配置

程序开始位置定义了三个模型路径。

```python
DINO_ROOT = Path("/media/jiangzhenmin/系统/JZM/GroundingDINO")

SAM_ROOT = Path("/media/jiangzhenmin/系统/JZM/segment-anything")
```

GroundingDINO配置文件：

```python
DINO_CONFIG
```

GroundingDINO权重：

```python
DINO_CHECKPOINT
```

SAM权重：

```python
SAM_CHECKPOINT
```

如果模型放在其它目录，只需要修改这里即可。

---

# 测试图片

默认图片：

```python
DEFAULT_IMAGE = DINO_ROOT / "images/box.png"
```

如果想测试自己的图片，可以修改为

```python
DEFAULT_IMAGE = Path(
"/你的图片路径/test.jpg"
)
```

更推荐的方法是在运行时指定：

```
python grounded_sam.py \
--image /你的图片路径/test.jpg
```

无需修改代码。

---

# 文本提示词（Prompt）

程序使用 GroundingDINO 的文本提示检测目标。

默认：

```python
default="object . box ."
```

可以改为：

检测纸箱

```text
box .
```

检测瓶子

```text
bottle .
```

检测苹果

```text
apple .
```

检测桌子

```text
table .
```

检测人

```text
person .
```

多个类别

```text
box . bottle . cup .
```

注意：

GroundingDINO 官方要求每个类别后面都有一个英文句点：

```
box .
```

而不是

```
box
```

也可以运行时指定：

```
python grounded_sam.py \
--text "box . bottle ."
```

---

# GroundingDINO 参数

GroundingDINO 有两个重要阈值。

## box_threshold

```python
default=0.35
```

控制检测框保留阈值。

建议：

```
0.20
```

目标较少时。

```
0.30
```

默认推荐。

```
0.45
```

减少误检。

---

## text_threshold

```python
default=0.25
```

控制文本匹配程度。

通常保持默认即可。

推荐：

```
0.20~0.30
```

---

# 输出目录

默认：

```python
DEFAULT_OUTPUT
```

所有结果都会保存到：

```
output/
```

包括：

```
grounded_sam_result.png
```

最终检测结果。

```
mask_000.png
```

每个目标对应一个Mask。

```
detections.json
```

保存检测信息。

---

# GroundingDINO模型

当前模型：

```
groundingdino_swint_ogc.pth
```

如果更换其它模型：

例如：

```
groundingdino_swinb_cogcoor.pth
```

修改：

```python
DINO_CHECKPOINT
```

同时修改配置：

```python
DINO_CONFIG
```

---

# SAM模型

当前：

```
sam_vit_h_4b8939.pth
```

对应：

```python
sam_model_registry["vit_h"]
```

如果改成 ViT-B：

```
sam_vit_b_01ec64.pth
```

则需要修改：

```python
sam_model_registry["vit_b"]
```

---

# GPU设置

程序自动判断：

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

无需修改。

如果需要强制CPU：

```python
device="cpu"
```

---

# 程序流程说明

整个程序可以分为以下几个模块。

---

## 1. parse_args()

读取命令行参数。

包括：

- 图片路径
- Prompt
- 输出目录
- 阈值

---

## 2. load_models()

加载两个模型。

```
GroundingDINO
```

负责目标检测。

```
SAM
```

负责实例分割。

---

## 3. run_grounding_dino()

输入：

```
图片
+
Prompt
```

输出：

```
boxes

logits

phrases
```

其中：

boxes：

```
(cx,cy,w,h)
```

logits：

检测分数。

phrases：

对应类别名称。

---

## 4. convert_boxes_to_xyxy()

GroundingDINO输出：

```
cxcywh
```

SAM需要：

```
xyxy
```

因此这里负责坐标转换。

同时：

归一化坐标

↓

转换成像素坐标。

---

## 5. run_sam()

SAM根据GroundingDINO输出的检测框：

```
Box
```

预测：

```
Mask
```

输出：

```
masks

mask_scores
```

---

## 6. create_visualization()

负责：

画框

↓

画Mask

↓

写类别名称

↓

写检测分数

最终生成：

```
grounded_sam_result.png
```

---

## 7. save_results()

保存：

```
检测图片
```

```
Mask图片
```

```
JSON信息
```

JSON内容包括：

```
类别

GroundingDINO得分

SAM得分

Bounding Box

Mask路径
```

---

## 8. main()

整个程序入口。

执行流程：

```
读取参数
      │
      ▼
检查模型
      │
      ▼
加载模型
      │
      ▼
读取图片
      │
      ▼
GroundingDINO检测
      │
      ▼
检测框
      │
      ▼
SAM分割
      │
      ▼
保存结果
```

---

# 如何测试自己的图片

推荐方法：

```
python grounded_sam.py \
--image /你的图片/test.jpg \
--text "box ."
```

例如：

```
python grounded_sam.py \
--image images/test.jpg \
--text "apple . bottle ."
```

程序会自动完成：

```
GroundingDINO检测

↓

SAM分割

↓

保存结果
```

无需修改代码。

---

# 输出结果说明

程序最终会生成：

```
output/
```

目录。

包括：

```
grounded_sam_result.png
```

带检测框和Mask的可视化图片。

```
mask_000.png
```

目标1的Mask。

```
mask_001.png
```

目标2的Mask。

```
detections.json
```

保存检测信息。

---

# 后续扩展建议

本项目目前支持：

- 单张图片推理

后续可扩展：

- 批量图片推理
- 摄像头实时检测
- ROS 图像订阅
- 机器人视觉系统集成
- GroundingDINO + SAM2
- GroundingDINO + MobileSAM
- GroundingDINO + EfficientSAM

适用于机器人抓取、开放词汇目标检测及视觉感知等任务。
