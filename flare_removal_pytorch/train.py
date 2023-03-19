import torch
from loss.model_loss import PerceptualLoss
from model.model import AB
from dataloader.flare_dataloader import Flare_dataloader
import torchvision.transforms as transforms
from dataloader.image_utils import *




shuffle = True
data_path = "D:/labs/LOW_LEVEL_IMAGEING/data/Flare7K_dataset/"
output_path = "D:/labs/LOW_LEVEL_IMAGEING/data/output/"
from matplotlib import pyplot as plt
transform_base = transforms.Compose([transforms.RandomCrop((256, 256), pad_if_needed=True, padding_mode='reflect'),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip()
                                     ])
transform_flare = transforms.Compose([transforms.RandomAffine(degrees=(0, 360), scale=(0.8, 1.5),
                                                              translate=(300 / 1440, 300 / 1440), shear=(-20, 20)),
                                      transforms.CenterCrop((256,256)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip()
                                      ])
flare_image_loader = Flare_dataloader(data_path + 'Flickr24K', transform_base, transform_flare, "luminance")
flare_image_loader.load_scattering_flare('Flare7K', "D:/labs/LOW_LEVEL_IMAGEING/data/Flare_Removal_DataSet/lens-flare/")



import time
import random
import numpy as np
import torch
import os
import json
from terminaltables import AsciiTable
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.logger import *
from utils.options import TrainOptions


def weights_init_normal(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def init():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Train:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = os.path.join("weights", self.args.model_name)
        self.model = None
        self.logger = None

    def check_model_path(self):
        try:
            os.makedirs(self.model_path)
            os.makedirs(os.path.join(self.model_path, "logs"))
        except Exception as e:
            print(e)

    def load_model(self):
        self.model = AB(3)
        self.model = self.model.to(self.device)
        self.loss_fn = PerceptualLoss()
        if self.args.weights_path != "":
            pretrained_dict = torch.load(self.args.weights_path)
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for i, (k, v) in enumerate(pretrained_dict.items()) if i < 552}
            model_dict.update(pretrained_dict)
            self.model.apply(weights_init_normal)
            self.model.load_state_dict(model_dict)
            print("load model")

    def save_model(self):
        save_folder = os.path.join(self.model_path, "ryolov4_with_gostnet.pth")
        torch.save(self.model.state_dict(), save_folder)

    def save_opts(self):
        to_save = self.args.__dict__.copy()
        with open(os.path.join(self.model_path, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def log(self, total_loss, epoch, global_step, total_step, start_time):
        log = "\n---- [Epoch %d/%d] ----\n" % (epoch + 1, self.args.epochs)
        tensorboard_log = {}
        #loss_table_name = ["Step: %d/%d" % (global_step, total_step), "loss", "reg_loss", "conf_loss", "cls_loss"]
        #loss_table = [loss_table_name]
        tensorboard_log["total_loss"] = total_loss
        self.logger.list_of_scalars_summary(tensorboard_log, global_step)
        #log += AsciiTable(loss_table).table
        log += "\nTotal Loss: %f, Runtime: %f\n" % (total_loss, time.time() - start_time)
        print(log)


    def get_loss(self , pred_scene , pred_flare ,flare_mask , scene , flare, flare_loss_weight = 0.0):
        masked_scene = pred_scene * (1 - flare_mask) + scene * flare_mask
        # plt.imsave("masked_scene.png", masked_scene[0].detach().permute(1, 2, 0).cpu().numpy())
        loss_value = self.loss_fn(scene, masked_scene)
        if flare_loss_weight > 0:
            masked_flare = pred_flare * (1 - flare_mask) + flare * flare_mask
            loss_value += flare_loss_weight * self.loss_fn(flare, masked_flare)
        return loss_value

    def train(self):
        init()
        self.check_model_path()
        self.load_model()
        self.save_opts()
        self.logger = Logger(os.path.join(self.model_path, "logs"))
        augment = False if self.args.no_augmentation else True
        mosaic = False if self.args.no_mosaic else True
        multiscale = False if self.args.no_multiscale else True
        train_dataloader = torch.utils.data.DataLoader(flare_image_loader, batch_size=self.args.batch_size, shuffle=shuffle,pin_memory=True, )
        num_iters_per_epoch = len(train_dataloader)
        scheduler_iters = round(self.args.epochs * len(train_dataloader) / self.args.subdivisions)
        total_step = num_iters_per_epoch * self.args.epochs
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                  first_cycle_steps=round(scheduler_iters),
                                                  max_lr=self.args.lr,
                                                  min_lr=1e-5,
                                                  warmup_steps=round(scheduler_iters * 0.1),
                                                  cycle_mult=1,
                                                  gamma=1)

        start_time = time.time()
        self.model.train()
        for epoch in range(self.args.epochs):
            for i, (base_img, flare_img, merge_img, flare_mask_img , gamma) in enumerate(train_dataloader):
                global_step = i * epoch + self.args.batch_size + 1
                merge_img = merge_img.to(self.device)
                #-----------------------------------
                flare_img = flare_img.to(self.device)
                base_img = base_img.to(self.device)
                flare_mask_img = flare_mask_img.to(self.device)
                #------------------------------------------------
                pred_scene = self.model(merge_img)
                pred_flare = remove_flare(merge_img, pred_scene, gamma.detach().cpu().numpy()) # plt.imsave("pred_flare.png", pred_flare[0].detach().permute(1, 2, 0).cpu().numpy())

                loss = self.get_loss(pred_scene, pred_flare, flare_mask_img, base_img, flare_img)

                loss.backward()
                total_loss = loss.detach().item()
                print("total loss , global_step" , total_loss , global_step)
                if global_step % self.args.subdivisions == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                self.log(total_loss, epoch, global_step, total_step, start_time)

            self.save_model()
            print("Model is saved!")

        print("Done!")


if __name__ == "__main__":
    parser = TrainOptions()
    args = parser.parse()
    print(args)

    t = Train(args)
    t.train()

