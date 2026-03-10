import requests


def video_seg_test():
    # API地址
    API_URL = "http://localhost:8000/segment"

    # 请求数据
    data = {
        "frame_folder": "/datadisk/sam2/notebooks/videos/bedroom",
        "annotations": [
            {
                "object_name": "boy",
                "frame_idx": 0,
                "points": [
                    {"x": 210, "y": 350, "label": 1},
                    {"x": 250, "y": 220, "label": 1},
                ],
                "auto_mask": 0,
            },
            {
                "object_name": "rmbg",
                "frame_idx": 0,
                "auto_mask": 1,
            }
        ],
        "output_dir": "/datadisk/sam2/notebooks/videos/test/",  # 可选，指定输出路径
        "background_color": [0, 0, 0]  # 可选，黑色背景
    }

    # 发送请求
    response = requests.post(API_URL, json=data)
    result = response.json()

    print(f"分割成功: {result['success']}")
    print(f"处理帧数: {result['frame_count']}")
    print(f"物体列表: {result['objects']}")
    print(f"结果路径: {result['result_paths']}")


if __name__ == '__main__':
    video_seg_test()