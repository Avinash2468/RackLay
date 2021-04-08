
import argparse
import os

import racklay

import numpy as np

import torch
from torch.utils.data import DataLoader
from racklay.dataloader import Loader
import PIL.Image as pil
import cv2
import matplotlib.pyplot as plt

import tqdm

from utils import mean_IU, mean_precision


def get_args():
    parser = argparse.ArgumentParser(description="Evaluation options")
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to the root data directory")
    parser.add_argument("--pretrained_path", type=str, default="./models/",
                        help="Path to the pretrained model")
    parser.add_argument("--osm_path", type=str, default="./data/osm",
                        help="OSM path")
    parser.add_argument(
        "--split",
        type=str,
        choices=[
            "warehouse",
            "argo",
            "3Dobject",
            "odometry",
            "raw"],
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
            "frontview"],
        help="Type of model being trained")
    parser.add_argument("--occ_map_size", type=int, default=256,
                        help="size of topview occupancy map")
    parser.add_argument("--num_workers", type=int, default=12,
                        help="Number of cpu workers for dataloaders")
    parser.add_argument("--num_racks", type=int, default=4,
                        help="Max number of racks")

    return parser.parse_args()


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def load_model(models, model_path):
    """Load model(s) from disk
    """
    model_path = os.path.expanduser(model_path)

    assert os.path.isdir(model_path), \
        "Cannot find folder {}".format(model_path)
    print("loading model from folder {}".format(model_path))

    for key in models.keys():
        print("Loading {} weights...".format(key))
        path = os.path.join(model_path, "{}.pth".format(key))
        model_dict = models[key].state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {
            k: v for k,
            v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        models[key].load_state_dict(model_dict)
    return models


def evaluate():
    opt = get_args()

    # Loading Pretarined Model
    models = {}
    models["encoder"] = racklay.Encoder(18, opt.height, opt.width, True)
    if opt.type == "both":
        models["top_decoder"] = racklay.Decoder(
            models["encoder"].resnet_encoder.num_ch_enc, 3*opt.num_racks,opt.occ_map_size)
        models["front_decoder"] = racklay.Decoder(
            models["encoder"].resnet_encoder.num_ch_enc, 3*opt.num_racks,opt.occ_map_size)
    elif opt.type == "topview":
        models["top_decoder"] = racklay.Decoder(
            models["encoder"].resnet_encoder.num_ch_enc)
    elif opt.type == "frontview":
        models["front_decoder"] = racklay.Decoder(
            models["encoder"].resnet_encoder.num_ch_enc)

    for key in models.keys():
        models[key].to("cuda")

    models = load_model(models, opt.pretrained_path)

    # Loading Validation/Testing Dataset

    # Data Loaders
    dataset_dict = {"warehouse": Loader,
                    "3Dobject": racklay.KITTIObject,
                    "odometry": racklay.KITTIOdometry,
                    "argo": racklay.Argoverse,
                    "raw": racklay.KITTIRAW}

    dataset = dataset_dict[opt.split]
    fpath = os.path.join(
        os.path.dirname(__file__),
        "splits",
        opt.split,
        "{}_files.txt")
    test_filenames = readlines(fpath.format("val"))
    test_dataset = dataset(opt, test_filenames, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        1,
        True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True)

    iou_box_top, mAP_box_top = np.array([0., 0.]), np.array([0., 0.])
    iou_rack_top, mAP_rack_top = np.array([0., 0.]), np.array([0., 0.])
    iou_box_front, mAP_box_front = np.array([0., 0.]), np.array([0., 0.])
    iou_rack_front, mAP_rack_front = np.array([0., 0.]), np.array([0., 0.])
    for batch_idx, inputs in tqdm.tqdm(enumerate(test_loader)):
        with torch.no_grad():
            outputs = process_batch(opt, models, inputs)
        #top view
        if(opt.type == "both" or opt.type == "topview"):
            for i in range(opt.num_racks):  # For the Rack Case
                input_temp =   inputs["topview"][:,i,:,:].detach().cpu().numpy()
                input_onlyrack = np.zeros_like(input_temp)
                input_onlyrack[input_temp==1] = 1

                input_temp = np.squeeze(input_onlyrack)
                input_temp = cv2.resize(input_temp, dsize=(opt.occ_map_size, opt.occ_map_size), interpolation=cv2.INTER_NEAREST)

                pred = np.squeeze(
                    torch.argmax(
                        outputs["top"][:,3*i:3*i+3,:,:].detach(),
                        1).cpu().numpy())
                pred_temp = np.zeros_like(pred)
                pred_temp[pred==1] = 1

                true = np.squeeze(input_temp)
                iou_rack_top += mean_IU(pred_temp, true)
                mAP_rack_top += mean_precision(pred_temp, true)

            for i in range(opt.num_racks):  
                input_temp =   inputs["topview"][:,i,:,:].detach().cpu().numpy()
                input_onlybox = np.zeros_like(input_temp)
                input_onlybox[input_temp==2] = 1
                input_temp = np.squeeze(input_onlybox)
                input_temp = cv2.resize(input_temp, dsize=(opt.occ_map_size, opt.occ_map_size), interpolation=cv2.INTER_NEAREST)

                pred = np.squeeze(
                    torch.argmax(
                        outputs["top"][:,3*i:3*i+3,:,:].detach(),
                        1).cpu().numpy())
                pred_temp = np.zeros_like(pred)
                pred_temp[pred==2] = 1

                true = np.squeeze(input_temp)
                iou_box_top += mean_IU(pred_temp, true)
                mAP_box_top += mean_precision(pred_temp, true)
            
        #front view
        if(opt.type == "both" or opt.type == "frontview"):
            for i in range(opt.num_racks):  # For the Rack Case
                input_temp =   inputs["frontview"][:,i,:,:].detach().cpu().numpy()
                input_onlyrack = np.zeros_like(input_temp)
                input_onlyrack[input_temp==1] = 1

                input_temp = np.squeeze(input_onlyrack)
                input_temp = cv2.resize(input_temp, dsize=(opt.occ_map_size, opt.occ_map_size), interpolation=cv2.INTER_NEAREST)

                pred = np.squeeze(
                    torch.argmax(
                        outputs["front"][:,3*i:3*i+3,:,:].detach(),
                        1).cpu().numpy())
                pred_temp = np.zeros_like(pred)
                pred_temp[pred==1] = 1

                true = np.squeeze(input_temp)
                iou_rack_front += mean_IU(pred_temp, true)
                mAP_rack_front += mean_precision(pred_temp, true)

            for i in range(opt.num_racks):  
                input_temp =   inputs["frontview"][:,i,:,:].detach().cpu().numpy()
                input_onlybox = np.zeros_like(input_temp)
                input_onlybox[input_temp==2] = 1
                input_temp = np.squeeze(input_onlybox)
                input_temp = cv2.resize(input_temp, dsize=(opt.occ_map_size, opt.occ_map_size), interpolation=cv2.INTER_NEAREST)

                pred = np.squeeze(
                    torch.argmax(
                        outputs["front"][:,3*i:3*i+3,:,:].detach(),
                        1).cpu().numpy())
                pred_temp = np.zeros_like(pred)
                pred_temp[pred==2] = 1

                true = np.squeeze(input_temp)
                iou_box_front += mean_IU(pred_temp, true)
                mAP_box_front += mean_precision(pred_temp, true)


    if(opt.type == "both" or opt.type == "topview"):        
        iou_rack_top /= (len(test_loader)*opt.num_racks)
        mAP_rack_top /= (len(test_loader)*opt.num_racks)
        iou_box_top /= (len(test_loader)*opt.num_racks)
        mAP_box_top /= (len(test_loader)*opt.num_racks)
        print("Evaluation Results for Rack Top: mIOU: %.4f mAP: %.4f" % (iou_rack_top[1], mAP_rack_top[1]))
        print("Evaluation Results for Box Top: mIOU: %.4f mAP: %.4f" % (iou_box_top[1], mAP_box_top[1]))
    
    if(opt.type == "both" or opt.type == "frontview"):
        iou_rack_front /= (len(test_loader)*opt.num_racks)
        mAP_rack_front /= (len(test_loader)*opt.num_racks)
        iou_box_front /= (len(test_loader)*opt.num_racks)
        mAP_box_front /= (len(test_loader)*opt.num_racks)
        print("Evaluation Results for Rack Front: mIOU: %.4f mAP: %.4f" % (iou_rack_front[1], mAP_rack_front[1]))
        print("Evaluation Results for Box Front: mIOU: %.4f mAP: %.4f" % (iou_box_front[1], mAP_box_front[1]))


def process_batch(opt, models, inputs):
    outputs = {}
    for key, input_ in inputs.items():
        inputs[key] = input_.to("cuda")

    features = models["encoder"](inputs["color"])

    if opt.type == "both":
        outputs["front"] = models["front_decoder"](features, is_training=False)
        outputs["top"] = models["top_decoder"](features, is_training=False)
    elif opt.type == "topview":
        outputs["top"] = models["top_decoder"](features)
    elif opt.type == "frontview":
        outputs["front"] = models["front_decoder"](features)

    return outputs


if __name__ == "__main__":
    evaluate()
