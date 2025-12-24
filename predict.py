import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from engine.core import YAMLConfig
from PIL import Image, ImageDraw
import os

def load_model(config_path, pth_path, device='cuda'):
    """加载模型"""
    # 加载配置
    cfg = YAMLConfig(config_path, resume=pth_path)
    
    # 如果使用 HGNetv2 backbone，禁用自动下载预训练权重
    if 'HGNetv2' in cfg.yaml_cfg: 
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    
    # 加载权重
    checkpoint = torch.load(pth_path, map_location='cpu')
    if 'ema' in checkpoint: 
        state = checkpoint['ema']['module']
    else: 
        state = checkpoint['model']
    
    cfg.model.load_state_dict(state)
    
    # 封装为推理模型
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()  # 转换为推理模式
            self.postprocessor = cfg. postprocessor.deploy()
        
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = Model().to(device).eval()
    return model, cfg. yaml_cfg

def draw(images, labels, boxes, scores, image_path, thrh=0.45):
    name = image_path.split('/')[-1].split('.')[0]
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline='red')
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(), 2)}", fill='blue', )

        im.save(f'{name}_results.jpg')

def predict(model, image_path, device='cuda', size=(640, 640), vit_backbone=False):
    """进行预测"""
    # 加载图片
    im_pil = Image.open(image_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)
    
    # 图像预处理
    transforms = T.Compose([
        T. Resize(size),
        T.ToTensor(),
        # ViT backbone 需要归一化，HGNetv2 不需要
        T. Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
            if vit_backbone else T.Lambda(lambda x: x)
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        labels, boxes, scores = model(im_data, orig_size)
        draw([im_pil], labels, boxes, scores, image_path)
    
    return labels, boxes, scores

# 使用示例
if __name__ == '__main__':
    device = 'cuda:0'
    config_path = './configs/deimv2_dinov3_x_custom.yml'
    pth_path = './weight/best_stg2_slim.pth'
    image_path = './20250625190246.jpeg'
    
    # 加载模型
    model, yaml_cfg = load_model(config_path, pth_path, device)
    
    # 判断是否是 ViT backbone
    vit_backbone = 'DINOv3STAs' in yaml_cfg
    size = tuple(yaml_cfg. get('eval_spatial_size', [640, 640]))
    
    # 预测
    labels, boxes, scores = predict(model, image_path, device, size, vit_backbone)
    
    # 处理结果（过滤低置信度）
    # threshold = 0.5
    # for label, box, score in zip(labels[0], boxes[0], scores[0]):
    #     # if score > threshold:
    #     print(f"Class:  {label. item()}, Score: {score.item():.2f}, Box: {box. tolist()}")