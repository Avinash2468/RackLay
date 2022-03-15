import argparse
import os

import racklay

import numpy as np

import torch
from torch.utils.data import DataLoader
from racklay import videolayout
from racklay.dataloader import Loader
import PIL.Image as pil
import cv2
import matplotlib.pyplot as plt

from torchvision import transforms

import tqdm

from utils import mean_IU, mean_precision


def get_args():
    parser = argparse.ArgumentParser(description="Evaluation options")
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to the root data directory")
    parser.add_argument("--pretrained_path", type=str, default="./models/",
                        help="Path to the pretrained model")
    
    parser.add_argument("--model_name", type=str, default="videolayout",
                        help="Name of model")

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

    parser.add_argument("--seq_len", type=int, default=8,
                        help="number of frames in an input")   
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
    parser.add_argument("--occ_map_size", type=int, default=512,
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

def sequence_readlines(filename , seq_len):
    f = open(filename, "r")
    files = [k.split("\n")[:-1] for k in f.read().split(",")[:-1]]
    sequence_files = []
    temporal_files = []
    for seq_files in files:
        temporal_files[:] = []
        seq_files = [seq_files[0]]*seq_len + seq_files
        for i in range(seq_len, len(seq_files)):
            temporal_files.append(seq_files[i-seq_len:i])
        
        sequence_files.append(temporal_files)
    # print(sequence_files)
    return sequence_files

def temporal_readlines(filename , seq_len):
    f = open(filename, "r")
    files = [k.split("\n")[:-1] for k in f.read().split(",")[:-1]]
    print("NUM SEQUENCES - " , len(files))
    temporal_files = []
    for seq_files in files:
        seq_files = [seq_files[0]]*seq_len + seq_files
        for i in range(seq_len, len(seq_files)):
            temporal_files.append(seq_files[i-seq_len:i])
    # print(temporal_files)
    # for tf in temporal_files:
    #     print(tf)
    return temporal_files


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
    args = get_args()

    # Loading Pretarined Model
    # models = {}
    # models["encoder"] = videolayout.Encoder(18, opt.height, opt.width, False)
    # models["convlstm"] = videolayout.ConvLSTM((16, 16), 512, 512, (3, 3), 1)
    # if opt.type == "both":
    #     models["top_decoder"] = racklay.Decoder(
    #         models["encoder"].resnet_encoder.num_ch_enc, 3*opt.num_racks,opt.occ_map_size)
    #     models["front_decoder"] = racklay.Decoder(
    #         models["encoder"].resnet_encoder.num_ch_enc, 3*opt.num_racks,opt.occ_map_size)
    # elif opt.type == "topview":
    #     models["top_decoder"] = racklay.Decoder(
    #         models["encoder"].resnet_encoder.num_ch_enc)
    # elif opt.type == "frontview":
    #     models["front_decoder"] = racklay.Decoder(
    #         models["encoder"].resnet_encoder.num_ch_enc)

    # for key in models.keys():
    #     models[key].to("cuda")

    # models = load_model(models, opt.pretrained_path)

    models = {}
    device = torch.device("cuda")
    encoder_path = os.path.join(args.pretrained_path, "encoder.pth")
    encoder_dict = torch.load(encoder_path, map_location=device)
    feed_height = encoder_dict["height"]
    feed_width = encoder_dict["width"]
    seq_len = args.seq_len
    to_tensor = transforms.ToTensor()

    models["encoder"] = videolayout.Encoder(18, feed_height, feed_width, False)
    filtered_dict_enc = {
        k: v for k,
        v in encoder_dict.items() if k in models["encoder"].state_dict()}
    models["encoder"].load_state_dict(filtered_dict_enc)

    models["convlstm"] = videolayout.ConvLSTM((16, 16), 512, 512, (3, 3), 1)
    convlstm_path = os.path.join(args.pretrained_path, "convlstm.pth")
    models["convlstm"].load_state_dict(torch.load(convlstm_path, map_location=device))
    
    if args.type == "both":
        top_decoder_path = os.path.join(
            args.pretrained_path, "top_decoder.pth")
        front_decoder_path = os.path.join(
            args.pretrained_path, "front_decoder.pth")
        models["top_decoder"] = videolayout.Decoder(
            models["encoder"].resnet_encoder.num_ch_enc, 3*args.num_racks,args.occ_map_size)
        models["top_decoder"].load_state_dict(
            torch.load(top_decoder_path, map_location=device))
        models["front_decoder"] = videolayout.Decoder(
            models["encoder"].resnet_encoder.num_ch_enc, 3*args.num_racks,args.occ_map_size)
        models["front_decoder"].load_state_dict(
            torch.load(front_decoder_path, map_location=device))
    elif args.type == "topview":
        decoder_path = os.path.join(args.pretrained_path, "top_decoder.pth")
        models["top_decoder"] = videolayout.Decoder(
            models["encoder"].resnet_encoder.num_ch_enc, 3*args.num_racks,args.occ_map_size)
        models["top_decoder"].load_state_dict(
            torch.load(decoder_path, map_location=device))
    elif args.type == "frontview":
        decoder_path = os.path.join(args.pretrained_path, "front_decoder.pth")
        models["front_decoder"] = videolayout.Decoder(
            models["encoder"].resnet_encoder.num_ch_enc, 3*args.num_racks,args.occ_map_size)
        models["front_decoder"].load_state_dict(
            torch.load(decoder_path, map_location=device))

    for key in models.keys():
        models[key].to(device)
        models[key].eval()
    
    print("ALL MODEL WEIGHTS LOADED")


    # Loading Validation/Testing Dataset

    # Data Loaders
    dataset_dict = {"warehouse": Loader,
                    "3Dobject": racklay.KITTIObject,
                    "odometry": racklay.KITTIOdometry,
                    "argo": racklay.Argoverse,
                    "raw": racklay.KITTIRAW}

    dataset = dataset_dict[args.split]
    fpath = os.path.join(
        os.path.dirname(__file__),
        "splits",
        args.split,
        "{}_files.txt")
    print(fpath)
    test_filenames = temporal_readlines(fpath.format("val_temporal") , args.seq_len)
    test_dataset = dataset(args, test_filenames, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        1,
        True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True)
    
    print("LOADERS READY")
    print(
        "There are {:d} testing items\n".format(
            len(test_dataset)))

    iou_box_top, mAP_box_top = np.array([0., 0.]), np.array([0., 0.])
    iou_rack_top, mAP_rack_top = np.array([0., 0.]), np.array([0., 0.])
    iou_box_front, mAP_box_front = np.array([0., 0.]), np.array([0., 0.])
    iou_rack_front, mAP_rack_front = np.array([0., 0.]), np.array([0., 0.])
    for batch_idx, inputs in tqdm.tqdm(enumerate(test_loader)):
        with torch.no_grad():
            outputs = process_batch(args, models, inputs)
            # print(outputs.keys())
    #     #top view
        if(args.type == "both" or args.type == "topview"):
            for i in range(args.num_racks):  # For the Rack Case
                input_temp =   inputs["topview"][:,i,:,:].detach().cpu().numpy()
                input_onlyrack = np.zeros_like(input_temp)
                input_onlyrack[input_temp==1] = 1
            
                input_temp = np.squeeze(input_onlyrack)
                input_temp = cv2.resize(input_temp, dsize=(args.occ_map_size, args.occ_map_size), interpolation=cv2.INTER_NEAREST)


                pred = np.squeeze(
                    torch.argmax(
                        outputs["top"][:,3*i:3*i+3,:,:].detach(),
                        1).cpu().numpy())
                pred_temp = np.zeros_like(pred)
                pred_temp[pred==1] = 1

                true = np.squeeze(input_temp)
                iou_rack_top += mean_IU(pred_temp, true)
                mAP_rack_top += mean_precision(pred_temp, true)

            for i in range(args.num_racks):  
                input_temp =   inputs["topview"][:,i,:,:].detach().cpu().numpy()
                input_onlybox = np.zeros_like(input_temp)
                input_onlybox[input_temp==2] = 1
                input_temp = np.squeeze(input_onlybox)
                input_temp = cv2.resize(input_temp, dsize=(args.occ_map_size, args.occ_map_size), interpolation=cv2.INTER_NEAREST)

                pred = np.squeeze(
                    torch.argmax(
                        outputs["top"][:,3*i:3*i+3,:,:].detach(),
                        1).cpu().numpy())
                pred_temp = np.zeros_like(pred)
                pred_temp[pred==2] = 1

                true = np.squeeze(input_temp)
                iou_box_top += mean_IU(pred_temp, true)
                mAP_box_top += mean_precision(pred_temp, true)
            
    #     #front view
        if(args.type == "both" or args.type == "frontview"):
            for i in range(args.num_racks):  # For the Rack Case
                input_temp =   inputs["frontview"][:,i,:,:].detach().cpu().numpy()
                input_onlyrack = np.zeros_like(input_temp)
                input_onlyrack[input_temp==1] = 1

                input_temp = np.squeeze(input_onlyrack)
                input_temp = cv2.resize(input_temp, dsize=(args.occ_map_size, args.occ_map_size), interpolation=cv2.INTER_NEAREST)

                pred = np.squeeze(
                    torch.argmax(
                        outputs["front"][:,3*i:3*i+3,:,:].detach(),
                        1).cpu().numpy())
                pred_temp = np.zeros_like(pred)
                pred_temp[pred==1] = 1

                true = np.squeeze(input_temp)
                iou_rack_front += mean_IU(pred_temp, true)
                mAP_rack_front += mean_precision(pred_temp, true)

            for i in range(args.num_racks):  
                input_temp =   inputs["frontview"][:,i,:,:].detach().cpu().numpy()
                input_onlybox = np.zeros_like(input_temp)
                input_onlybox[input_temp==2] = 1
                input_temp = np.squeeze(input_onlybox)
                input_temp = cv2.resize(input_temp, dsize=(args.occ_map_size, args.occ_map_size), interpolation=cv2.INTER_NEAREST)

                pred = np.squeeze(
                    torch.argmax(
                        outputs["front"][:,3*i:3*i+3,:,:].detach(),
                        1).cpu().numpy())
                pred_temp = np.zeros_like(pred)
                pred_temp[pred==2] = 1

                true = np.squeeze(input_temp)
                iou_box_front += mean_IU(pred_temp, true)
                mAP_box_front += mean_precision(pred_temp, true)


    if(args.type == "both" or args.type == "topview"):        
        iou_rack_top /= (len(test_loader)*args.num_racks)
        mAP_rack_top /= (len(test_loader)*args.num_racks)
        iou_box_top /= (len(test_loader)*args.num_racks)
        mAP_box_top /= (len(test_loader)*args.num_racks)
        print("Evaluation Results for Rack Top: mIOU: %.4f mAP: %.4f" % (iou_rack_top[1], mAP_rack_top[1]))
        print("Evaluation Results for Box Top: mIOU: %.4f mAP: %.4f" % (iou_box_top[1], mAP_box_top[1]))
    
    if(args.type == "both" or args.type == "frontview"):
        iou_rack_front /= (len(test_loader)*args.num_racks)
        mAP_rack_front /= (len(test_loader)*args.num_racks)
        iou_box_front /= (len(test_loader)*args.num_racks)
        mAP_box_front /= (len(test_loader)*args.num_racks)
        print("Evaluation Results for Rack Front: mIOU: %.4f mAP: %.4f" % (iou_rack_front[1], mAP_rack_front[1]))
        print("Evaluation Results for Box Front: mIOU: %.4f mAP: %.4f" % (iou_box_front[1], mAP_box_front[1]))


def process_batch(opt, models, inputs):
    outputs = {}
    for key, input_ in inputs.items():
        inputs[key] = input_.to("cuda")
    
    mu = models["encoder"](inputs['color'])
    z = mu
    z = models["convlstm"](z)[0][0][:,-1]

    if opt.type == "both":
        outputs["front"] = models["front_decoder"](z, is_training=False)
        outputs["top"] = models["top_decoder"](z, is_training=False)
    elif opt.type == "topview":
        outputs["top"] = models["top_decoder"](z , is_training=False)
    elif opt.type == "frontview":
        outputs["front"] = models["front_decoder"](z , is_training=False)

    return outputs


if __name__ == "__main__":
    evaluate()
