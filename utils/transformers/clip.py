import torch
import open_clip
import timm
import math
import torch.nn.functional as F

def clip_vitb16(img_size):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k',jit=False,force_image_size=img_size) # laion2b_s34b_b88k,laion400m_e31
    return model.visual

def main():
   
   model = clip_vitb16(img_size =1024)
   img = torch.randn((10,3,1024,1024))
   enc = model(img)
   print(enc.shape)
  

if __name__ == '__main__':
    main()
