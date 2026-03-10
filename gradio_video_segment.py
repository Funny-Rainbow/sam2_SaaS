"""
SAM2 视频分割 Gradio 界面
支持文件夹输入，交互式标注（前景点、背景点、边界框）
"""
import os
import requests
import numpy as np
from PIL import Image, ImageDraw
import gradio as gr

# 默认配置
DEFAULT_API_URL = "http://localhost:8000"


def get_frame_list(frame_folder: str):
    """获取帧列表"""
    if not frame_folder or not os.path.exists(frame_folder):
        return []
    exts = (".jpg", ".jpeg", ".png")
    frames = sorted(
        [f for f in os.listdir(frame_folder) if f.lower().endswith(exts)],
        key=lambda p: int(os.path.splitext(p)[0]) if os.path.splitext(p)[0].isdigit() else p,
    )
    return frames


def draw_annotations(image, fg_points, bg_points, box_points):
    """在图像上绘制标注"""
    if image is None:
        return None

    img = image.copy()
    draw = ImageDraw.Draw(img)

    # 绘制前景点（绿色）
    for x, y in fg_points:
        r = 30
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(0, 255, 0), outline=(0, 200, 0), width=2)

    # 绘制背景点（红色）
    for x, y in bg_points:
        r = 30
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0), outline=(200, 0, 0), width=2)

    # 绘制边界框
    if len(box_points) == 1:
        x, y = box_points[0]
        size = 10
        draw.line([x - size, y, x + size, y], fill=(0, 0, 255), width=10)
        draw.line([x, y - size, x, y + size], fill=(0, 0, 255), width=10)
    elif len(box_points) == 2:
        x1, y1 = box_points[0]
        x2, y2 = box_points[1]
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        draw.rectangle([left, top, right, bottom], outline=(0, 0, 255), width=15)

    return img


def load_frame_folder(frame_folder: str):
    """加载帧文件夹，返回第一帧用于标注"""
    if not frame_folder or not os.path.exists(frame_folder):
        return None, None, [], [], [], "请输入有效的帧文件夹路径", gr.update(maximum=0, value=0)

    frames = get_frame_list(frame_folder)
    if not frames:
        return None, None, [], [], [], "文件夹中没有找到图片帧", gr.update(maximum=0, value=0)

    first_frame_path = os.path.join(frame_folder, frames[0])
    first_frame = Image.open(first_frame_path).convert("RGB")

    return (
        first_frame,
        first_frame.copy(),
        [],
        [],
        [],
        f"已加载 {len(frames)} 帧，显示第一帧: {frames[0]}",
        gr.update(maximum=len(frames) - 1, value=0),
    )


def switch_frame(frame_folder: str, frame_idx: int, fg_points, bg_points, box_points):
    """切换当前预览的帧"""
    frames = get_frame_list(frame_folder)
    if not frames or frame_idx >= len(frames):
        return None, None

    frame_path = os.path.join(frame_folder, frames[frame_idx])
    frame = Image.open(frame_path).convert("RGB")

    # 如果是第一帧，绘制标注
    if frame_idx == 0:
        annotated = draw_annotations(frame, fg_points, bg_points, box_points)
        return annotated, frame
    else:
        return frame, frame


def handle_click(original_image, fg_points, bg_points, box_points, mode, evt: gr.SelectData):
    """处理图像点击，添加标注"""
    if original_image is None:
        return None, fg_points, bg_points, box_points

    x, y = evt.index[0], evt.index[1]

    if mode == "前景点":
        fg_points = fg_points + [[x, y]]
    elif mode == "背景点":
        bg_points = bg_points + [[x, y]]
    elif mode == "边界框":
        if len(box_points) >= 2:
            box_points = [[x, y]]
        else:
            box_points = box_points + [[x, y]]

    annotated_img = draw_annotations(original_image, fg_points, bg_points, box_points)
    return annotated_img, fg_points, bg_points, box_points


def clear_annotations(original_image):
    """清除所有标注"""
    if original_image is None:
        return None, [], [], []
    return original_image.copy(), [], [], []


def run_segmentation(
    api_url: str,
    frame_folder: str,
    output_dir: str,
    object_name: str,
    fg_points,
    bg_points,
    box_points,
    frame_idx: int,
    bg_color_r: int,
    bg_color_g: int,
    bg_color_b: int,
):
    """调用 SAM2 API 进行视频分割"""
    if not frame_folder or not os.path.exists(frame_folder):
        raise gr.Error("请先加载有效的帧文件夹")

    if not output_dir:
        raise gr.Error("请指定输出目录")

    has_visual = fg_points or bg_points or len(box_points) == 2
    if not has_visual:
        raise gr.Error("请至少添加一个前景点、背景点或完整边界框")

    # 构建标注
    annotation = {"object_name": object_name or "object", "frame_idx": frame_idx}

    all_points = []
    for x, y in fg_points:
        all_points.append({"x": int(x), "y": int(y), "label": 1})
    for x, y in bg_points:
        all_points.append({"x": int(x), "y": int(y), "label": 0})
    if all_points:
        annotation["points"] = all_points

    if len(box_points) == 2:
        x1, y1 = box_points[0]
        x2, y2 = box_points[1]
        annotation["box"] = {
            "x1": int(min(x1, x2)),
            "y1": int(min(y1, y2)),
            "x2": int(max(x1, x2)),
            "y2": int(max(y1, y2)),
        }

    # 调用 API
    api_endpoint = f"{api_url.rstrip('/')}/segment"
    data = {
        "frame_folder": frame_folder,
        "annotations": [annotation],
        "output_dir": output_dir,
        "background_color": [bg_color_r, bg_color_g, bg_color_b],
    }

    response = requests.post(api_endpoint, json=data, timeout=600)
    result = response.json()

    if not result.get("success", False):
        raise gr.Error(f"分割失败: {result.get('message', '未知错误')}")

    # 获取结果路径
    objects = result.get("objects", [])
    result_paths = result.get("result_paths", {})
    frame_count = result.get("frame_count", 0)

    # 加载第一个物体的第一帧分割结果预览
    preview_img = None
    if objects and result_paths:
        first_obj = objects[0]
        segframe_folder = result_paths[first_obj].get("segframe_folder", "")
        if segframe_folder and os.path.exists(segframe_folder):
            seg_files = sorted(os.listdir(segframe_folder))
            if seg_files:
                preview_img = Image.open(os.path.join(segframe_folder, seg_files[0]))

    status = f"分割完成！处理了 {frame_count} 帧，检测到 {len(objects)} 个物体: {objects}"
    return preview_img, status, result_paths


def create_demo():
    """创建 Gradio 界面"""
    with gr.Blocks(title="SAM2 视频分割") as demo:
        gr.Markdown("# SAM2 视频分割")
        gr.Markdown("输入帧文件夹 → 在第一帧标注物体 → 分割整个视频")

        # 状态变量
        original_image = gr.State(None)
        fg_points = gr.State([])
        bg_points = gr.State([])
        box_points = gr.State([])
        result_paths = gr.State({})

        # API 配置
        with gr.Row():
            api_url = gr.Textbox(
                label="SAM2 API 地址",
                value=DEFAULT_API_URL,
                placeholder="http://ip:port",
            )

        # 输入输出路径
        with gr.Row():
            frame_folder = gr.Textbox(
                label="帧文件夹路径",
                placeholder="/path/to/frames",
            )
            output_dir = gr.Textbox(
                label="输出目录",
                placeholder="/path/to/output",
            )
            load_btn = gr.Button("加载帧文件夹", variant="primary")

        # 主布局
        with gr.Row():
            # 左列：标注
            with gr.Column(scale=1):
                gr.Markdown("### 标注区域（在第一帧上标注）")
                annotated_image = gr.Image(
                    label="点击添加标注",
                    type="pil",
                    interactive=True,
                    sources=[],  # 禁用上传，只能通过加载文件夹
                )

                frame_slider = gr.Slider(
                    minimum=0,
                    maximum=0,
                    step=1,
                    value=0,
                    label="预览帧（标注仅在第一帧生效）",
                    interactive=True,
                )

                annotation_mode = gr.Radio(
                    choices=["前景点", "背景点", "边界框"],
                    value="前景点",
                    label="标注模式",
                )

                with gr.Row():
                    object_name = gr.Textbox(
                        label="物体名称",
                        value="object",
                        placeholder="object",
                    )

                with gr.Row():
                    bg_color_r = gr.Number(label="背景R", value=0, minimum=0, maximum=255, precision=0)
                    bg_color_g = gr.Number(label="背景G", value=0, minimum=0, maximum=255, precision=0)
                    bg_color_b = gr.Number(label="背景B", value=0, minimum=0, maximum=255, precision=0)

                with gr.Row():
                    clear_btn = gr.Button("清除标注", variant="secondary")
                    segment_btn = gr.Button("运行分割", variant="primary")

                status_text = gr.Textbox(label="状态", interactive=False)

            # 右列：结果
            with gr.Column(scale=1):
                gr.Markdown("### 分割结果预览")
                segmentation_result = gr.Image(
                    label="分割结果（第一帧）",
                    type="pil",
                    interactive=False,
                )

                gr.Markdown("### 输出路径")
                output_info = gr.JSON(label="输出路径信息")

        # 事件绑定
        load_btn.click(
            fn=load_frame_folder,
            inputs=[frame_folder],
            outputs=[annotated_image, original_image, fg_points, bg_points, box_points, status_text, frame_slider],
        )

        frame_slider.change(
            fn=switch_frame,
            inputs=[frame_folder, frame_slider, fg_points, bg_points, box_points],
            outputs=[annotated_image, original_image],
        )

        annotated_image.select(
            fn=handle_click,
            inputs=[original_image, fg_points, bg_points, box_points, annotation_mode],
            outputs=[annotated_image, fg_points, bg_points, box_points],
        )

        clear_btn.click(
            fn=clear_annotations,
            inputs=[original_image],
            outputs=[annotated_image, fg_points, bg_points, box_points],
        )

        segment_btn.click(
            fn=run_segmentation,
            inputs=[
                api_url,
                frame_folder,
                output_dir,
                object_name,
                fg_points,
                bg_points,
                box_points,
                frame_slider,
                bg_color_r,
                bg_color_g,
                bg_color_b,
            ],
            outputs=[segmentation_result, status_text, output_info],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
