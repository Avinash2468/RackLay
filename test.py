import argparse
import glob
import os

import PIL.Image as pil
from PIL import Image
import matplotlib.pyplot as plt

import cv2

from racklay import videolayout
# from .train import temporal_readlines

import numpy as np

import torch

from torchvision import transforms

def get_args():
    parser = argparse.ArgumentParser(
        description="Testing arguments for Racklay")
    parser.add_argument("--image_paths", type=str,
                        help="path of file containing temporal image paths", required=True)
    parser.add_argument("--model_path", type=str,
                        help="path to Racklay model", required=True)
    parser.add_argument( 
        "--ext",
        type=str,
        default="png",
        help="extension of images in the folder")
    parser.add_argument("--out_dir", type=str,
                        default="output directory to save topviews")
    parser.add_argument("--type", type=str,
                        default="both/topview/frontview")
    parser.add_argument("--num_racks", type=int, default=1,
                        help="Max number of racks")
    parser.add_argument("--occ_map_size", type=int, default=128,
                        help="size of topview occupancy map")
    parser.add_argument("--seq_len", type=int, default=8,
                        help="number of frames in an input")                    
    return parser.parse_args()

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

def save_topview(idx, tv_temp, name_dest_im):
    # print("PRINTING THE TEST OUTPUT SHAPE")
    # print(tv_temp.shape)
    for i in range(args.num_racks):
        tv = tv_temp[:,3*i:3*i+3,:,:]
        tv_np = tv.squeeze()
        tv = torch.argmax(tv_np, 0)
        #for i in range(len(tv)):
        #    for j in range(len(tv[i])):
        #        if(tv[i][j]==1):
        #            print("Here",end=" ")
        tv[tv==1] = 115
        tv[tv==2] = 255
        dir_name = os.path.dirname(name_dest_im)
        if not os.path.exists(dir_name):
            print(dir_name)
            os.makedirs(dir_name)
        cv2.imwrite(name_dest_im + "rackno_" +str(i) + ".png", tv.cpu().numpy())

    # print("Saved prediction to {}".format(name_dest_im))

def npy_loader(path):
    return np.load(path,allow_pickle=True)

def pil_loader(path):
    with open(path, 'rb') as f:
        with pil.open(f) as img:
            return img.convert('RGB')

def test(args):
    models = {}
    device = torch.device("cuda")
    encoder_path = os.path.join(args.model_path, "encoder.pth")
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
    convlstm_path = os.path.join(args.model_path, "convlstm.pth")
    models["convlstm"].load_state_dict(torch.load(convlstm_path, map_location=device))
    
    if args.type == "both":
        top_decoder_path = os.path.join(
            args.model_path, "top_decoder.pth")
        front_decoder_path = os.path.join(
            args.model_path, "front_decoder.pth")
        models["top_decoder"] = videolayout.Decoder(
            models["encoder"].resnet_encoder.num_ch_enc, 3*args.num_racks,args.occ_map_size)
        models["top_decoder"].load_state_dict(
            torch.load(top_decoder_path, map_location=device))
        models["front_decoder"] = videolayout.Decoder(
            models["encoder"].resnet_encoder.num_ch_enc, 3*args.num_racks,args.occ_map_size)
        models["front_decoder"].load_state_dict(
            torch.load(front_decoder_path, map_location=device))
    elif args.type == "topview":
        decoder_path = os.path.join(args.model_path, "top_decoder.pth")
        models["top_decoder"] = videolayout.Decoder(
            models["encoder"].resnet_encoder.num_ch_enc, 3*args.num_racks,args.occ_map_size)
        models["top_decoder"].load_state_dict(
            torch.load(decoder_path, map_location=device))
    elif args.type == "frontview":
        decoder_path = os.path.join(args.model_path, "front_decoder.pth")
        models["front_decoder"] = videolayout.Decoder(
            models["encoder"].resnet_encoder.num_ch_enc, 3*args.num_racks,args.occ_map_size)
        models["front_decoder"].load_state_dict(
            torch.load(decoder_path, map_location=device))

    for key in models.keys():
        models[key].to(device)
        models[key].eval()

    sequences = []
    if os.path.isfile(args.image_paths):
        sequences[:] = sequence_readlines(args.image_paths , args.seq_len)
    else:
        raise Exception(
            "Can not find args.image_paths: {}".format(
                args.image_paths))

    print("-> Predicting on {:d} test sequences".format(len(sequences)))
    print("Each sequence has {} {}-framed mini-sequences".format(len(sequences[0]) , len(sequences[0][0])))
    # for i in range(len(sequences)):
    #     print()
    #     print("SEQUENCE ",i)
    #     for mini_seq in sequences[i]:
    #         print(mini_seq)
        
    # PREDICTING ON EACH SEQUENCE
    with torch.no_grad():
        for idx, seq in enumerate(sequences):
            print("ON SEQ" , idx)
            #print(seq.shape)
            output_layouts = [] # append all outputs for this sequence here
            # predicting on each mini 8-sized sequence in this sequence
            for mini_idx, mini_seq in enumerate(seq):
                print(mini_idx , end='-')
                output_name = os.path.splitext(mini_seq[-1])[0]
                inputs = torch.empty(seq_len, 3, feed_width, feed_height)
                for mini_frame_idx, mini_frame_seq in enumerate(mini_seq):
                    #print(mini_frame_seq.shape)
                    img_path = os.path.join(mini_frame_seq)
                    color = pil_loader(img_path)
                    color = color.resize((feed_width, feed_height), pil.LANCZOS)
                    inputs[mini_frame_idx, :]  = to_tensor(color)
                inputs = inputs.unsqueeze(0)

                input_seq = inputs.to(device)
                mu = models["encoder"](input_seq)
                z = mu
                z = models["convlstm"](z)[0][0][:,-1]

                if args.type == "both":
                    top_tv = models["top_decoder"](z , is_training=False)
                    front_tv = models["front_decoder"](z, is_training=False)
                    output_name_top = output_name.replace("img/", "Results/topview/")
                    save_topview(
                        idx,
                        top_tv,
                        os.path.join("{}".format(output_name_top)))

                    output_name_front = output_name.replace("img/", "Results/frontview/")
                    save_topview(
                        idx,
                        front_tv,
                        os.path.join("{}".format(output_name_front)))

                elif args.type == "topview":
                    tv = models["top_decoder"](z, is_training=False)
                    output_name_top = output_name.replace("img/", "Results/topview/")
                    save_topview(
                        idx,
                        tv,
                        os.path.join("{}".format(output_name_top)))

                elif args.type == "frontview":
                    tv = models["front_decoder"](z, is_training=False) 
                    output_name_front = output_name.replace("img/", "Results/frontview/")
                    save_topview(
                        idx,
                        tv,
                        os.path.join("{}".format(output_name_front)))
            
            print("SEQ DONE")

        print('-> Done!')    

if __name__ == "__main__":
    args = get_args()
    test(args)
