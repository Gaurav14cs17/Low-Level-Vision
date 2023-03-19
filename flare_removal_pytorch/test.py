import torch
from loss.model_loss import PerceptualLoss
from model.model import AB
from dataloader.flare_dataloader import Flare_dataloader
import torchvision.transforms as transforms
from dataloader.image_utils import *
import os

weight_path = "./weights/trash/ryolov4_with_gostnet.pth"
dir_path = "D:/labs/LOW_LEVEL_IMAGEING/Low-Level-Vision/flare_removal_pytorch/Data/"
images_dir_path = os.path.join(dir_path , "testing")
output_dir_path = os.path.join(dir_path ,"output")
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)

def preprocess():
    pass



model = AB(3)
pretrained_dict = torch.load(weight_path, map_location=torch.device('cpu'))
model.load_state_dict(pretrained_dict)
#load model


for file_name in os.listdir(images_dir_path):
    print(file_name)
    image_path = os.path.join(images_dir_path , file_name)
    image = Image.open(image_path)
    input_image = transforms.ToTensor()(image)
    input_image = input_image.unsqueeze(0)
    output = model(input_image)
    pred_scene = torch.clip(output ,0 , 1)
    pred_flare = remove_flare(input_image, pred_scene)
    pred_blend = blend_light_source(input_image[0].detach(), pred_scene[0].detach())
    plt.imsave(os.path.join(output_dir_path , file_name.split('.')[0] + '_pred_blend.png'), pred_blend.detach().permute(1, 2, 0).cpu().numpy())
    plt.imsave(os.path.join(output_dir_path, file_name.split('.')[0] + '_pred_flare.png'), pred_flare[0].detach().permute(1, 2, 0).cpu().numpy())
    #exit()



