import os
import gc
from typing import List, Optional, Dict
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from modelscope import AutoModelForImageSegmentation

# ==================== 配置 ====================
SAM2_CHECKPOINT = os.environ.get("SAM2_CHECKPOINT", "checkpoints/sam2.1_hiera_large.pt")
MODEL_CFG = os.environ.get("SAM2_MODEL_CFG", "configs/sam2.1/sam2.1_hiera_l.yaml")
BIREFNET_MODEL_DIR = os.environ.get("BIREFNET_MODEL_DIR", "./BiRefNet")

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# ==================== 数据模型 ====================
class Point(BaseModel):
    """标注点"""
    x: float = Field(..., description="X坐标")
    y: float = Field(..., description="Y坐标")
    label: int = Field(..., description="标签：1=前景点，0=背景点")


class BoundingBox(BaseModel):
    """边界框"""
    x1: float = Field(..., description="左上角X坐标")
    y1: float = Field(..., description="左上角Y坐标")
    x2: float = Field(..., description="右下角X坐标")
    y2: float = Field(..., description="右下角Y坐标")


class ObjectAnnotation(BaseModel):
    """物体标注"""
    object_name: str = Field(..., description="物体名称，用于标识和保存结果")
    frame_idx: int = Field(..., description="标注所在的帧索引")
    points: Optional[List[Point]] = Field(None, description="标注点列表")
    box: Optional[BoundingBox] = Field(None, description="边界框")
    auto_mask: int = Field(0, description="是否自动获取帧的前景mask并作为sam输入，1为是，0为否")


class SegmentRequest(BaseModel):
    """视频分割请求"""
    frame_folder: str = Field(..., description="视频帧文件夹的绝对路径")
    annotations: List[ObjectAnnotation] = Field(..., description="物体标注列表")
    output_dir: Optional[str] = Field(None, description="结果输出目录（可选，默认与frame_folder同级）")
    background_color: Optional[List[int]] = Field([0, 0, 0], description="背景颜色RGB值（可选，默认黑色[0,0,0]）")

    class Config:
        json_schema_extra = {
            "example": {
                "frame_folder": "/home/caokun/projects/sam2/notebooks/videos/bedroom",
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
                        "auto_mask": 1
                    }
                ],
                "output_dir": "/path/to/output",
                "background_color": [0, 0, 0],
            }
        }


class SegmentResponse(BaseModel):
    """视频分割响应"""
    success: bool = Field(..., description="分割是否成功")
    message: str = Field(..., description="响应消息")
    frame_count: int = Field(..., description="处理的帧数")
    objects: List[str] = Field(..., description="分割的物体名称列表")
    result_paths: Dict[str, Dict[str, str]] = Field(..., description="每个物体的结果路径")


# ==================== FastAPI 应用 ====================
app = FastAPI(
    title="SAM 2 视频分割 API",
    description="SAM 2 视频分割 API",
    version="2.0.0",
    docs_url="/docs"
)

# 添加CORS中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 显存释放工具函数 ====================
def deep_clean_inference_state(inference_state):
    """
    深度清理 inference_state 中的所有 CUDA 张量
    """
    if inference_state is None:
        return
    
    try:
        # 递归清理所有包含 Tensor 的属性
        for key in list(vars(inference_state).keys()):
            attr = getattr(inference_state, key, None)
            if attr is None:
                continue
                
            # 清理 Tensor
            if isinstance(attr, torch.Tensor):
                if attr.is_cuda:
                    attr.cpu()
                delattr(inference_state, key)
            
            # 清理字典（如 output_dict）
            elif isinstance(attr, dict):
                for k in list(attr.keys()):
                    if isinstance(attr[k], torch.Tensor) and attr[k].is_cuda:
                        attr[k] = attr[k].cpu()
                    del attr[k]
                attr.clear()
            
            # 清理列表
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, torch.Tensor) and item.is_cuda:
                        item.cpu()
                attr.clear()
    except Exception as e:
        print(f"清理 inference_state 时出错: {e}")


def cleanup_gpu_memory():
    """
    彻底清理 GPU 显存
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # 强制垃圾回收 CUDA 缓存的分配器
        torch.cuda.reset_peak_memory_stats()


# ==================== API 端点 ====================

@app.get("/")
async def root():
    """API根路径"""
    return {
        "name": "SAM 2 视频分割 API - 修复版",
        "version": "2.0.0",
        "status": "running",
        "device": str(device),
        "docs": "/docs"
    }


@app.post("/segment", response_model=SegmentResponse)
async def segment_video(request: SegmentRequest):
    video_predictor = None
    birefnet = None
    inference_state = None
    video_segments = None
    
    try:
        # ==================== 1. 输入校验 ====================
        assert os.path.exists(request.frame_folder), f"视频帧文件夹不存在: {request.frame_folder}"
        
        frame_names = [p for p in os.listdir(request.frame_folder)
                       if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]]
        assert len(frame_names) > 0, "文件夹中没有找到图片文件"

        def sort_key(filename):
            basename = os.path.splitext(filename)[0]
            return (0, int(basename)) if basename.isdigit() else (1, basename)
        frame_names.sort(key=sort_key)

        # ==================== 2. 动态加载 SAM2 ====================
        print("正在加载 SAM2 视频预测器...")
        video_predictor = build_sam2_video_predictor(
            MODEL_CFG, SAM2_CHECKPOINT, device=device
        )
        
        # 初始化推理状态
        inference_state = video_predictor.init_state(video_path=request.frame_folder)
        video_predictor.reset_state(inference_state)

        # ==================== 3. 添加标注 ====================
        obj_id_map = {}
        for idx, annotation in enumerate(request.annotations):
            obj_id = idx + 1
            obj_id_map[annotation.object_name] = obj_id

            if annotation.auto_mask:
                # 动态加载 BiRefNet
                if birefnet is None:
                    print("正在加载 BiRefNet...")
                    birefnet = AutoModelForImageSegmentation.from_pretrained(
                        BIREFNET_MODEL_DIR, trust_remote_code=True
                    ).to(device).eval()
                    torch.set_float32_matmul_precision('high')

                actual_frame_name = frame_names[annotation.frame_idx]
                binary_mask = get_frame_mask_with_model(
                    birefnet, request.frame_folder, actual_frame_name
                )
                video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=annotation.frame_idx,
                    obj_id=obj_id,
                    mask=binary_mask
                )
            else:
                points = np.array([[p.x, p.y] for p in (annotation.points or [])], dtype=np.float32) if annotation.points else None
                labels = np.array([p.label for p in (annotation.points or [])], dtype=np.int32) if annotation.points else None
                box = np.array([annotation.box.x1, annotation.box.y1,
                               annotation.box.x2, annotation.box.y2], dtype=np.float32) if annotation.box else None

                assert points is not None or box is not None, f"物体 '{annotation.object_name}' 需要点或框标注"

                video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=annotation.frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                    box=box,
                )

        # ==================== 4. 传播分割 ====================
        print("开始全视频传播...")
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()  # 立即转到 CPU
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # ==================== 5. 保存结果 ====================
        output_dir = request.output_dir or os.path.dirname(os.path.abspath(request.frame_folder))
        bg_color = request.background_color or [0, 0, 0]

        result_paths = save_segmentation_results(
            frame_folder=request.frame_folder,
            frame_names=frame_names,
            video_segments=video_segments,
            output_dir=output_dir,
            obj_id_map=obj_id_map,
            background_color=bg_color
        )

        return SegmentResponse(
            success=True,
            message="分割完成",
            frame_count=len(frame_names),
            objects=list(obj_id_map.keys()),
            result_paths=result_paths
        )

    except Exception as e:
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        return SegmentResponse(
            success=False,
            message=f"分割失败: {str(e)}",
            frame_count=0,
            objects=[],
            result_paths={}
        )

    finally:
        # ==================== 6. 彻底释放显存（关键优化）====================
        print("开始清理显存...")
        
        # Step 1: 深度清理 inference_state（最重要！）
        if inference_state is not None:
            deep_clean_inference_state(inference_state)
            del inference_state
            inference_state = None
        
        # Step 2: 清理 video_segments
        if video_segments is not None:
            for frame_data in video_segments.values():
                frame_data.clear()
            video_segments.clear()
            del video_segments
            video_segments = None
        
        # Step 3: 清理模型（按相反顺序删除）
        if birefnet is not None:
            birefnet.cpu()  # 先移到 CPU
            del birefnet
            birefnet = None
        
        if video_predictor is not None:
            # 清理 predictor 内部的模型
            if hasattr(video_predictor, 'model'):
                video_predictor.model.cpu()
            del video_predictor
            video_predictor = None
        
        # Step 4: 强制垃圾回收（多次执行）
        for _ in range(3):  # 多次 GC 确保彻底清理
            cleanup_gpu_memory()
        
        # 打印显存使用情况（调试用）
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"清理后显存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")


# ==================== 辅助函数 ====================

def save_segmentation_results(
    frame_folder: str,
    frame_names: List[str],
    video_segments: Dict[int, Dict[int, np.ndarray]],
    output_dir: str,
    obj_id_map: Dict[str, int],
    background_color: List[int]
) -> Dict[str, Dict[str, str]]:
    """
    保存分割结果到文件
    """
    result_paths = {}
    
    for obj_name, obj_id in obj_id_map.items():
        # 创建物体文件夹
        obj_folder = os.path.join(output_dir, obj_name)
        mask_folder = os.path.join(obj_folder, f"{obj_name}_mask")
        segframe_folder = os.path.join(obj_folder, f"{obj_name}_segframe")
        
        os.makedirs(mask_folder, exist_ok=True)
        os.makedirs(segframe_folder, exist_ok=True)
        
        # 保存每一帧
        for frame_idx, frame_name in tqdm(enumerate(frame_names), desc=f"正在保存 {obj_name}", total=len(frame_names)):
            # 读取原始帧
            frame_path = os.path.join(frame_folder, frame_name)
            original_frame = np.array(Image.open(frame_path))
            
            # 检查该帧是否有该物体的mask
            if frame_idx in video_segments and obj_id in video_segments[frame_idx]:
                mask = video_segments[frame_idx][obj_id].squeeze()
            else:
                if len(original_frame.shape) == 3:
                    mask = np.zeros((original_frame.shape[0], original_frame.shape[1]), dtype=bool)
                else:
                    mask = np.zeros(original_frame.shape, dtype=bool)
            
            # 保存掩码图
            mask_image = (mask * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_image, mode="L")
            mask_filename = f"{frame_name}.png"
            mask_pil.save(os.path.join(mask_folder, mask_filename))
            
            # 保存分割帧图
            if len(original_frame.shape) == 2:
                original_frame = np.stack([original_frame] * 3, axis=-1)
            
            segmented_frame = original_frame.copy()
            segmented_frame[~mask] = background_color
            
            segframe_pil = Image.fromarray(segmented_frame.astype(np.uint8))
            segframe_pil.save(os.path.join(segframe_folder, frame_name))
        
        result_paths[obj_name] = {
            "object_folder": obj_folder,
            "mask_folder": mask_folder,
            "segframe_folder": segframe_folder
        }
    
    return result_paths


def get_frame_mask_with_model(birefnet_model, frame_folder: str, frame_name: str):
    """
    使用 BiRefNet 获取前景掩码
    """
    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    file_path = os.path.join(frame_folder, frame_name)
    image = Image.open(file_path).convert("RGB")
    input_images = transform_image(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = birefnet_model(input_images)[-1].sigmoid()

    # 转换为 float32 并移到 CPU
    pred = preds[0, 0].float().cpu()
    mask = (pred > 0.5).numpy()

    # resize 回原图尺寸
    mask_pil = Image.fromarray(mask)
    mask_pil = mask_pil.resize(image.size, Image.NEAREST)
    
    # 清理中间变量
    del input_images, preds, pred
    
    return np.asarray(mask_pil)


# ==================== 启动信息 ====================
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("SAM 2 视频分割 API - 修复显存泄漏")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"模型: {SAM2_CHECKPOINT}")
    print(f"BiRefNet: {BIREFNET_MODEL_DIR}")
    print("=" * 60)
    print("启动服务...")
    print("API端点: POST /segment")
    host = os.environ.get("SAM2_API_HOST", "0.0.0.0")
    port = int(os.environ.get("SAM2_API_PORT", "8000"))
    print(f"API文档: http://{host}:{port}/docs")
    print("=" * 60)
    
    uvicorn.run(
        "sam2_segment_api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
        workers=1
    )
