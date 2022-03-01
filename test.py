'''
I add the test.py in this repo , it can test other images .
But it only test one image in a run.
I believe you can add a Pytorch DataSet class to make it test more images every time.
Author:Shubin Huang
'''
from __future__ import absolute_import, division, print_function
import numpy as np
import torch
from models.modeling import VisionTransformer, CONFIGS
from PIL import Image
from torchvision import transforms

#Load the model
config = CONFIGS["ViT-B_16"]
model = VisionTransformer(config, 384, zero_head=True, num_classes=1000)
model.to("cuda")
ckpt = torch.load('./ViT_TRAR_384_012.ckpt', map_location=torch.device("cpu"))
model.load_state_dict(ckpt['state_dict'])
model.eval()

#You can change the test image file path
pic_dir='./pic/1.jpg'

#change the image format to tensor and do some transform like valid setting
image = Image.open(pic_dir)
transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
image = transform(image)
#Add batch dimension , the image shape can be like 1*C*W*H
image = image.unsqueeze(0).cuda()

output=model(image)
#print label index
print(torch.argmax(output[0]))

