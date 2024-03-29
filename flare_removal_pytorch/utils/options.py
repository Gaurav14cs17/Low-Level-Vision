import argparse

class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--data_folder", type=str, default="./DATA/DOTA_DATASET/", help="path to train dataset")
        self.parser.add_argument("--weights_path", type=str, default="", help="path to pretrained weights file")
        self.parser.add_argument("--model_name", type=str, default="trash", help="new model name")
        self.parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
        self.parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
        self.parser.add_argument("--batch_size", type=int, default=6, help="size of batches")
        self.parser.add_argument("--subdivisions", type=int, default=1, help="size of mini batches")
        self.parser.add_argument("--img_size", type=int, default=448, help="size of each image dimension")
        self.parser.add_argument("--sample_size", type=int, default=1000, help="size of cropped area when doing mosaic augmentation")
        self.parser.add_argument("--number_of_classes", type=int, default=16, help="number of your output classes")
        self.parser.add_argument("--no_augmentation", action="store_true", help="if set, disable data augmentation in training")
        self.parser.add_argument("--no_mosaic", action="store_true", help="if set, disable mosaic data augmentation in training")
        self.parser.add_argument("--no_multiscale", action="store_true", help="if set, disable multiscale data in training")
        self.parser.add_argument("--dataset", type=str, default="DOTA", choices=["UCAS_AOD", "DOTA", "custom"], help="specify dataset to use for training")

    def parse(self):
        return self.parser.parse_args()

class TestOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--data_folder", type=str, default="/home/nripendra/Multiple_treatment&Diagnosis/Experiment/YOLO-V4/DATA/DOTA_DATASET/", help="path to dataset")
        self.parser.add_argument("--model_name", type=str, default="ryolov4.pth", help="model name")
        self.parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
        self.parser.add_argument("--nms_thres", type=float, default=0.1, help="iou thresshold for non-maximum suppression")
        self.parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold for evaluation")
        self.parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
        self.parser.add_argument("--img_size", type=int, default=448, help="size of each image dimension")
        self.parser.add_argument("--number_of_classes", type=int, default=16, help="number of your output classes")
        self.parser.add_argument("--dataset", type=str, default="UCAS_AOD", choices=["UCAS_AOD", "DOTA", "custom"], help="specify dataset to use for testing")


    def parse(self):
        return self.parser.parse_args()

class DetectOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--data_folder", type=str, default="/home/nripendra/Multiple_treatment&Diagnosis/Experiment/YOLO-V4/DATA/DOTA_DATASET/", help="path to dataset")
        self.parser.add_argument("--model_name", type=str, default="ryolov4.pth", help="model name")
        self.parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
        self.parser.add_argument("--nms_thres", type=float, default=0.1, help="iou thresshold for non-maximum suppression")
        self.parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
        self.parser.add_argument("--img_size", type=int, default=448, help="size of each image dimension")
        self.parser.add_argument("--number_of_classes", type=int, default=16, help="number of your output classes")
        self.parser.add_argument("--ext", type=str, default="png", choices=["png", "jpg"], help="Image file format")

    def parse(self):
        return self.parser.parse_args()
