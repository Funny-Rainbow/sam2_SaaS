# SAM 2 视频分割 API 使用说明

服务默认在 `http://localhost:8000` 启动

推荐使用 `sam2_segment_api.sh` 启动（带 supervisor，每 12h 自动滚动重启一次，避免显存长期增长导致 OOM）。


## API接口

### POST /segment - 视频分割

**请求参数：**

| 参数               | 类型     | 必填 | 说明                                           |
|------------------|--------|----|----------------------------------------------|
| frame_folder     | string | 是  | 视频帧文件夹路径，图片帧格式必须为宽度为5的整数的jpg格式，例如00003.jpg   |
| annotations      | array  | 是  | 物体标注列表                                       |
| output_dir       | string | 否  | 输出目录（默认与frame_folder同级）                      |
| background_color | array  | 否  | 分割图背景颜色RGB值（默认黑色[0,0,0]），**仅影响分割图，掩码图始终为黑白** |

**标注参数 (annotations)：**

| 参数          | 类型      | 必填 | 说明                                 |
|-------------|---------|----|------------------------------------|
| object_name | string  | 是  | 物体名称                               |
| frame_idx   | integer | 是  | 标注所在帧索引                            |
| points      | array   | 否  | 点标注列表                              |
| box         | object  | 否  | 边界框标注                              |
| auto_mask   | int     | 是  | 是否自动获取帧的前景mask并作为sam输入，1为是，0为否 |

**点标注 (points)：**
- `x`: X坐标
- `y`: Y坐标
- `label`: 1=前景点，0=背景点

**边界框 (box)：**
- `x1`, `y1`: 左上角坐标
- `x2`, `y2`: 右下角坐标

---

## 使用示例

### 视频分割 - Python调用

```python
import requests

# API地址
API_URL = "http://localhost:8000/segment"

# 请求数据
data = {
    "frame_folder": "/datadisk/sam2/notebooks/videos/bedroom",
    "annotations": [
        {
            "object_name": "person1",
            "frame_idx": 0,
            "points": [
                {"x": 210, "y": 350, "label": 1},
                {"x": 250, "y": 220, "label": 1}
            ],
            "auto_mask": 0,
        },
        {
            "object_name": "person2",
            "frame_idx": 0,
            "auto_mask": 1,

        }
    ],
    "output_dir": "/path/to/output",  # 可选，指定输出路径
    "background_color": [255, 255, 255]  # 可选，白色背景
}

# 发送请求
response = requests.post(API_URL, json=data)
result = response.json()

print(f"分割成功: {result['success']}")
print(f"处理帧数: {result['frame_count']}")
print(f"物体列表: {result['objects']}")
print(f"结果路径: {result['result_paths']}")
```
---

## 响应格式

### 视频分割响应

**成功响应：**

```json
{
  "success": true,
  "message": "分割完成",
  "frame_count": 200,
  "objects": ["person1", "person2"],
  "result_paths": {
    "person1": {
      "object_folder": "/path/to/person1",
      "mask_folder": "/path/to/person1/person1_mask",
      "segframe_folder": "/path/to/person1/person1_segframe"
    },
    "person2": {
      "object_folder": "/path/to/person2",
      "mask_folder": "/path/to/person2/person2_mask",
      "segframe_folder": "/path/to/person2/person2_segframe"
    }
  }
}
```

**错误响应：**

```json
{
  "detail": "错误信息"
}
```


## 输出文件结构

### 视频分割输出

```
<输出目录>/
├── <物体名称1>/
│   ├── <物体名称1>_mask/         # 黑白掩码图
│   │   ├── 00000_mask.jpg
│   │   ├── 00001_mask.jpg
│   │   └── ...
│   └── <物体名称1>_segframe/     # 分割后的彩色图
│       ├── 00000_segframe.jpg
│       ├── 00001_segframe.jpg
│       └── ...
└── <物体名称2>/
    ├── <物体名称2>_mask/
    └── <物体名称2>_segframe/
```

**文件说明：**
- `_mask`: **黑白掩码图**，白色(255)=物体，黑色(0)=背景，**不受background_color参数影响**
- `_segframe`: **分割图**，前景=物体原色，背景=background_color指定的颜色
