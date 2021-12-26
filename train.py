import argparse
import os

import racklay

from racklay.dataloader import Loader
from racklay import VideoLayout
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import GPUtil  
import tqdm

from utils import mean_IU, mean_precision

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def get_args():
    parser = argparse.ArgumentParser(description="racklay options")

    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to the root data directory")

    parser.add_argument("--save_path", type=str, default="./models/",
                        help="Path to save models")

    parser.add_argument(
        "--load_weights_folder",
        type=str,
        default="",
        help="Path to a pretrained model used for initialization")

    parser.add_argument("--model_name", type=str, default="racklay",
                        help="Model Name with specifications")

    parser.add_argument(
        "--split",
        type=str,
        choices=[
            "argo",
            "3Dobject",
            "odometry",
            "raw",
            "warehouse"],
        help="Data split for training/validation")
    parser.add_argument("--ext", type=str, default="png",
                        help="File extension of the images")
    parser.add_argument("--height", type=int, default=512,
                        help="Image height")
    parser.add_argument("--width", type=int, default=512,
                        help="Image width")
    parser.add_argument(
        "--type",
        type=str,
        choices=[
            "both",
            "topview",
            "frontview"
            ],
        help="Type of model being trained")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Mini-Batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="learning rate")
    parser.add_argument("--lr_D", type=float, default=1e-5,
                        help="discriminator learning rate")
    parser.add_argument("--scheduler_step_size", type=int, default=5,
                        help="step size for the both schedulers")
    parser.add_argument("--static_weight", type=float, default=5.,
                        help="static weight for calculating loss")
    parser.add_argument("--dynamic_weight", type=float, default=15.,
                        help="dynamic weight for calculating loss")
    parser.add_argument("--occ_map_size", type=int, default=128,
                        help="size of topview occupancy map")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Max number of training epochs")
    parser.add_argument("--num_racks", type=int, default=1,
                        help="Max number of racks")
    parser.add_argument("--log_frequency", type=int, default=5,
                        help="Log files every x epochs")
    parser.add_argument("--num_workers", type=int, default=12,
                        help="Number of cpu workers for dataloaders")
    parser.add_argument("--lambda_D", type=float, default=0.01,
                        help="tradeoff weight for discriminator loss")
    parser.add_argument("--discr_train_epoch", type=int, default=5,
                        help="epoch to start training discriminator")
    parser.add_argument("--osm_path", type=str, default="./data/osm",
                        help="OSM path")
    parser.add_argument("--seq_len", type=int, default=8,
                        help="number of frames in an input")                    
    return parser.parse_args()

class Trainer:
    def __init__(self):
        self.opt = get_args()
        self.models = {}
        self.weight = {}
        self.weight["static"] = self.opt.static_weight
        self.weight["dynamic"] = self.opt.dynamic_weight
        self.device = "cuda"
        self.criterion_d = nn.BCEWithLogitsLoss()
        self.parameters_to_train = []
        self.parameters_to_train_D = []

        # Initializing models
        self.model = VideoLayout(self.opt).cuda()
        
        # Data Loaders
        dataset_dict = {
                        "warehouse": Loader,
                        "3Dobject": racklay.KITTIObject,
                        "odometry": racklay.KITTIOdometry,
                        "argo": racklay.Argoverse,
                        "raw": racklay.KITTIRAW}

        self.dataset = dataset_dict[self.opt.split]
        fpath = os.path.join(
            os.path.dirname(__file__),
            "splits",
            self.opt.split,
            "{}_files.txt")
        print("THE FPATH IS")
        print(fpath)

        if self.opt.model_name == "videolayout":
            readlines_fn = self.temporal_readlines
            train_file = "train_temporal"
            val_file = "val_temporal"
        else:
            readlines_fn = self.readlines
            train_file = "train"
            val_file = "val"

        train_filenames = readlines_fn(fpath.format(train_file))
        val_filenames = readlines_fn(fpath.format(val_file))
        self.val_filenames = val_filenames
        self.train_filenames = train_filenames

        train_dataset = self.dataset(self.opt, train_filenames)
        val_dataset = self.dataset(self.opt, val_filenames, is_train=False)

        print(train_dataset.data_path)

        self.train_loader = DataLoader(
            train_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)
        
        self.val_loader = DataLoader(
            val_dataset,
            1,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=False)

        if self.opt.load_weights_folder != "":
            self.load_model()

        print("Using split:\n  ", self.opt.split)
        print(
            "There are {:d} training items and {:d} validation items\n".format(
                len(train_dataset),
                len(val_dataset)))

    def train(self):
        iters = []
        val_loss = []
        train_loss = []
        for self.epoch in range(self.opt.num_epochs):
        
            loss = self.run_epoch()
            if(self.opt.type == "both"):
                print("Epoch: %d | Top Loss: %.4f | Top Discriminator Loss: %.4f | Front Loss: %.4f | Front Discriminator Loss: %.4f"%
                      (self.epoch, loss["top_loss"], loss["top_loss_discr"], loss["front_loss"], loss["front_loss_discr"]))

            elif(self.opt.type == "topview"):
                print("Epoch: %d | Top Loss: %.4f | Top Discriminator Loss: %.4f"%
                      (self.epoch, loss["top_loss"], loss["top_loss_discr"]))

            elif(self.opt.type == "frontview"):
                print("Epoch: %d | Front Loss: %.4f | Front Discriminator Loss: %.4f"%
                      (self.epoch, loss["front_loss"], loss["front_loss_discr"]))

            if self.epoch % self.opt.log_frequency == 0:
                self.save_model()
                
            if self.epoch % 2 == 0:
                loss_val = self.validation()
                iters.append(self.epoch)
                val_loss.append(loss_val["loss"])
                train_loss.append(loss["loss"])
        plt.figure(figsize=(10,4))
        plt.title("Training Curve:Loss")
        plt.plot(iters, train_loss, label="Train")
        plt.plot(iters, val_loss, label="Validation")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend(loc='best')
        plt.savefig('Curves.png')


    def readlines(filename):
        """Read all the lines in a text file and return as a list
        """
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
        return lines

    def temporal_readlines(self, filename):
        f = open(filename, "r")
        files = [k.split("\n")[:-1] for k in f.read().split(",")[:-1]]
        temporal_files = []
        for seq_files in files:
            seq_files = [seq_files[0]]*self.opt.seq_len + seq_files
            for i in range(self.opt.seq_len, len(seq_files)):
                temporal_files.append(seq_files[i-self.opt.seq_len:i])
        return temporal_files

    def process_batch(self, inputs, validation=False):
        outputs = {}
        for key, inpt in inputs.items():
            inputs[key] = inpt.to(self.device)

        outputs = self.model(inputs["color"])

        if validation:
            return outputs
        
        #print("PRINTING THE INPUT AND OUTPUT OCCUPANCY MAP SIZES")
        # print(inputs["topview"].size,outputs["topview"].size)
        losses = self.compute_losses(inputs, outputs)
        losses["loss_discr"] = torch.zeros(1)

        return outputs, losses

    def run_epoch(self):
        loss = {}
        loss["top_loss"], loss["front_loss"], loss["top_loss_discr"], loss["front_loss_discr"] = 0.0, 0.0, 0.0, 0.0
        loss["loss"] = 0.0
        for batch_idx, inputs in tqdm.tqdm(enumerate(self.train_loader)):

            outputs, losses = self.process_batch(inputs)
            lossess = self.model.step(inputs, outputs, losses, self.epoch)
        
            if(self.opt.type == "both" or self.opt.type == "topview"):
                loss["top_loss"] += losses["top_loss"].item()
                loss["loss"] += losses["top_loss"].item()
                loss["top_loss_discr"] += lossess["loss_D_top"].item() 
                # loss["top_loss_discr"] += 0
            if(self.opt.type == "both" or self.opt.type == "frontview"):
                loss["front_loss"] += losses["front_loss"].item()
                loss["loss"] += losses["front_loss"].item()
                loss["front_loss_discr"] += lossess["loss_D_front"].item()
                # loss["front_loss_discr"] += 0
        
        return loss

    def validation(self):
        #iou_rack, mAP_rack = np.array([0., 0.]), np.array([0., 0.])
        loss = {}
        loss["top_loss"], loss["front_loss"], loss["top_loss_discr"], loss["front_loss_discr"] = 0.0, 0.0, 0.0, 0.0
        loss["loss"] = 0.0
        for batch_idx, inputs in tqdm.tqdm(enumerate(self.val_loader)):
            outputs, losses = self.process_batch(inputs)

            if(self.opt.type == "both" or self.opt.type == "topview"):
                loss["top_loss"] += losses["top_loss"].item()
            if(self.opt.type == "both" or self.opt.type == "frontview"):
                loss["front_loss"] += losses["front_loss"].item()
            
        print(" Top Loss: %.4f  | Front Loss: %.4f "%( loss["top_loss"], loss["front_loss"])) 

        if(self.opt.type == "both" or self.opt.type == "topview"):
            loss["loss"] += losses["top_loss"].item() 
        if(self.opt.type == "both" or self.opt.type == "frontview"):
            loss["loss"] += losses["front_loss"].item() 
        return loss

    def compute_losses(self, inputs, outputs):
        losses = {}
        if self.opt.type == "both":
            losses["top_loss"] = self.compute_topview_loss(
                                            outputs["topview"],
                                            inputs["topview"])
            losses["front_loss"] = self.compute_topview_loss(
                                            outputs["frontview"],
                                            inputs["frontview"])
        elif self.opt.type == "topview":
            losses["top_loss"] = self.compute_topview_loss(
                                            outputs[self.opt.type],
                                            inputs[self.opt.type])

        elif self.opt.type == "frontview":
            losses["front_loss"] = self.compute_topview_loss(
                                            outputs[self.opt.type],
                                            inputs[self.opt.type])
        return losses

    def compute_topview_loss(self, outputs, true_top_view):
        generated_top_view = outputs;       
        true_top_view = true_top_view.long()#.reshape(self.opt.batch_size, self.opt.occ_map_size, self.opt.occ_map_size)

        loss = nn.CrossEntropyLoss(weight=torch.Tensor([1., 5., 5.]).cuda())
        loss_list = []
        for i in range(self.opt.num_racks):
            gen_temp = generated_top_view[:,3*i:3*i+3,:,:]
            true_temp = true_top_view[:,i,:,:]
            loss_temp = loss(gen_temp, true_temp) 
            # print(loss_temp.mean().data)
            # loss_list.append(loss_temp.mean().cpu().detach().numpy())
            loss_list.append(loss_temp.mean())
        return np.mean(np.array(loss_list))

    def save_model(self):
        save_path = os.path.join(
            self.opt.save_path,
            self.opt.model_name,
            self.opt.split,
            "weights_{}".format(
                self.epoch))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for model_name, model in self.models.items():
            model_path = os.path.join(save_path, "{}.pth".format(model_name))
            state_dict = model.state_dict()
            if model_name == "encoder":
                state_dict["height"] = self.opt.height
                state_dict["width"] = self.opt.width

            torch.save(state_dict, model_path)
        optim_path = os.path.join(save_path, "{}.pth".format("adam"))
        torch.save(self.model.model_optimizer.state_dict(), optim_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(
            self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print(
            "loading model from folder {}".format(
                self.opt.load_weights_folder))

        for key in self.models.keys():
            print("Loading {} weights...".format(key))
            path = os.path.join(
                self.opt.load_weights_folder,
                "{}.pth".format(key))
            model_dict = self.models[key].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[key].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(
            self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()