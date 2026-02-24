import os
import torch
import numpy as np
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms

# Import từ các module đã tách
import config
from models import Generator

# ==========================================
# 1. LOAD MODEL
# ==========================================
def load_model(checkpoint_path, device):
    print(f"🔄 Đang tải trọng số mô hình từ {checkpoint_path}...")
    model = Generator().to(device)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file trọng số: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Xử lý key state_dict (tương thích với file của bạn)
    state_dict = checkpoint.get('generator', checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    print("✅ Load model thành công!")
    return model

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def convert_sketch_to_rect_mask(mask_array):
    """
    Biến đổi nét vẽ tự do thành hình chữ nhật bao quanh (Bounding Box)
    """
    rows, cols = np.where(mask_array > 0)
    
    if len(rows) == 0:
        return mask_array
        
    y_min, y_max = np.min(rows), np.max(rows)
    x_min, x_max = np.min(cols), np.max(cols)
    
    rect_mask = np.zeros_like(mask_array)
    rect_mask[y_min:y_max+1, x_min:x_max+1] = 255
    return rect_mask

def tensor_to_pil(t):
    t = (t + 1) / 2
    t = t.clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((t * 255).astype(np.uint8))

# ==========================================
# 3. CORE INFERENCE PIPELINE
# ==========================================
def predict_inpainting(input_dict):
    """
    Xử lý dữ liệu từ Gradio UI, chạy model và trả về kết quả
    """
    if input_dict is None or input_dict["background"] is None:
        return None, None
    
    # 1. Lấy dữ liệu gốc
    orig_image = input_dict["background"]
    
    # Xử lý số kênh màu để tránh lỗi
    if orig_image.ndim == 2: # Ảnh đen trắng
        orig_image = np.stack([orig_image]*3, axis=-1)
    if orig_image.ndim == 3 and orig_image.shape[2] == 4: # Ảnh PNG có Alpha
        orig_image = orig_image[:, :, :3]
        
    orig_h, orig_w = orig_image.shape[:2]
    
    # Lấy Mask từ Gradio ImageEditor
    if len(input_dict["layers"]) > 0 and input_dict["layers"][0] is not None:
        mask_layer = input_dict["layers"][0]
        orig_mask = mask_layer[:, :, 3]  # Lấy kênh Alpha của nét vẽ
    else:
        orig_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        
    # 2. Chuẩn bị dữ liệu cho AI (Resize xuống 256x256)
    pil_orig_image = Image.fromarray(orig_image).convert("RGB")
    pil_orig_mask = Image.fromarray(orig_mask).convert("L")
    
    image_256 = pil_orig_image.resize((config.IMG_SIZE, config.IMG_SIZE), Image.BICUBIC)
    mask_256 = pil_orig_mask.resize((config.IMG_SIZE, config.IMG_SIZE), Image.NEAREST)
    
    # Xử lý Mask thành hình chữ nhật
    mask_256_np = np.array(mask_256)
    rect_mask_256_np = convert_sketch_to_rect_mask(mask_256_np)
    
    # Tạo Tensor
    mask_tensor = transforms.ToTensor()(Image.fromarray(rect_mask_256_np))
    mask_tensor = (mask_tensor > 0.5).float()
    img_tensor = transform(image_256)
    
    # Đục lỗ ảnh 256
    corrupted_img_256 = img_tensor * (1 - mask_tensor) + mask_tensor * (-1.0)
    
    # 3. Chạy Model
    img_input = corrupted_img_256.unsqueeze(0).to(config.device)
    with torch.no_grad():
        fake_output_256 = model(img_input)
    
    fake_output_256_cpu = fake_output_256[0].cpu()
    
    # 4. Hậu xử lý & Blending độ phân giải cao
    ai_result_pil = tensor_to_pil(fake_output_256_cpu)
    ai_result_upscaled = ai_result_pil.resize((orig_w, orig_h), Image.BICUBIC)
    
    rect_mask_pil = Image.fromarray(rect_mask_256_np).resize((orig_w, orig_h), Image.NEAREST)
    rect_mask_final = np.array(rect_mask_pil) / 255.0 
    rect_mask_final = np.expand_dims(rect_mask_final, axis=-1)
    
    orig_image_float = orig_image.astype(float)
    ai_result_float = np.array(ai_result_upscaled).astype(float)
    
    final_image = orig_image_float * (1 - rect_mask_final) + ai_result_float * rect_mask_final
    final_image = final_image.astype(np.uint8)
    
    vis_corrupted = tensor_to_pil(corrupted_img_256) 
    return np.array(vis_corrupted), final_image

# ==========================================
# 4. KHỞI TẠO VÀ CHẠY APP
# ==========================================
if __name__ == '__main__':
    # Nạp mô hình toàn cục
    MODEL_PATH = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
    try:
        model = load_model(MODEL_PATH, config.device)
    except Exception as e:
        print(f"Lỗi: {e}. Vui lòng kiểm tra lại đường dẫn trong config.py")
        exit(1)
        
    # Xây dựng giao diện Gradio
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 AI Face Inpainting - High Resolution")
        gr.Markdown("Tải ảnh lên, dùng cọ vẽ bôi đen vùng khuôn mặt bị lỗi, AI sẽ tự động tạo khung chữ nhật và khôi phục nó.")
        
        with gr.Row():
            with gr.Column():
                input_img = gr.ImageEditor(
                    label="Vẽ vùng cần xóa",
                    type="numpy",
                    brush=gr.Brush(colors=["#000000"], default_size=15),
                    eraser=gr.Eraser(),
                    transforms=[] 
                )
                btn = gr.Button("🚀 Phục hồi khuôn mặt", variant="primary")
            
            with gr.Column():
                output_corrupted = gr.Image(label="Vùng AI nhìn thấy (Hình chữ nhật)")
                output_final = gr.Image(label="Kết quả phục hồi (High-Res Blending)")
        
        btn.click(fn=predict_inpainting, inputs=input_img, outputs=[output_corrupted, output_final])
        
    # Chạy Web UI
    demo.launch(share=False, debug=True)