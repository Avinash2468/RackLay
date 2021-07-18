import argparse
import glob
import os

import PIL.Image as pil
from PIL import Image
import matplotlib.pyplot as plt

import cv2

from racklay import model

import numpy as np

import torch

from torchvision import transforms


def get_args():
    parser = argparse.ArgumentParser(
        description="Testing arguments for Racklay")
    parser.add_argument("--image_path", type=str,
                        help="path to folder of images", required=True)
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
    return parser.parse_args()


def save_topview(idx, tv_temp, name_dest_im):
    print("PRINTING THE TEST OUTPUT SHAPE")
    print(tv_temp.shape)
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
            os.makedirs(dir_name)
        cv2.imwrite(name_dest_im + "rackno_" +str(i) + ".png", tv.cpu().numpy())

    print("Saved prediction to {}".format(name_dest_im))


def npy_loader(path):
    return np.load(path,allow_pickle=True)
    
    
def test(args):
    models = {}
    device = torch.device("cuda")
    encoder_path = os.path.join(args.model_path, "encoder.pth")
    encoder_dict = torch.load(encoder_path, map_location=device)
    feed_height = encoder_dict["height"]
    feed_width = encoder_dict["width"]
    models["encoder"] = model.Encoder(18, feed_width, feed_height, False)
    filtered_dict_enc = {
        k: v for k,
        v in encoder_dict.items() if k in models["encoder"].state_dict()}
    models["encoder"].load_state_dict(filtered_dict_enc)

    if args.type == "both":
        top_decoder_path = os.path.join(
            args.model_path, "top_decoder.pth")
        front_decoder_path = os.path.join(
            args.model_path, "front_decoder.pth")
        models["top_decoder"] = model.Decoder(
            models["encoder"].resnet_encoder.num_ch_enc, 3*args.num_racks,args.occ_map_size)
        models["top_decoder"].load_state_dict(
            torch.load(top_decoder_path, map_location=device))
        models["front_decoder"] = model.Decoder(
            models["encoder"].resnet_encoder.num_ch_enc, 3*args.num_racks,args.occ_map_size)
        models["front_decoder"].load_state_dict(
            torch.load(front_decoder_path, map_location=device))
    elif args.type == "topview":
        decoder_path = os.path.join(args.model_path, "top_decoder.pth")
        models["top_decoder"] = model.Decoder(
            models["encoder"].resnet_encoder.num_ch_enc, 3*args.num_racks,args.occ_map_size)
        models["top_decoder"].load_state_dict(
            torch.load(decoder_path, map_location=device))
    elif args.type == "frontview":
        decoder_path = os.path.join(args.model_path, "front_decoder.pth")
        models["front_decoder"] = model.Decoder(
            models["encoder"].resnet_encoder.num_ch_enc, 3*args.num_racks,args.occ_map_size)
        models["front_decoder"].load_state_dict(
            torch.load(decoder_path, map_location=device))

    for key in models.keys():
        models[key].to(device)
        models[key].eval()

    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(
            args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.out_dir
        try:
            os.mkdir(output_directory)
        except BaseException:
            pass
    else:
        raise Exception(
            "Can not find args.image_path: {}".format(
                args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            #img = npy_loader(image_path)
            #input_image = Image.fromarray(img.astype('uint8'), 'RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize(
                (feed_width, feed_height), pil.LANCZOS)
            print("INPUT IMAGE SHAPE")
            print(input_image.size)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = models["encoder"](input_image)
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            print(
                "Processing {:d} of {:d} images- ".format(idx + 1, len(paths)))
            if args.type == "both":
                top_tv = models["top_decoder"](
                    features, is_training=False)
                front_tv = models["front_decoder"](
                    features, is_training=False)
                save_topview(
                    idx,
                    top_tv,
                    os.path.join(
                        args.out_dir,
                        "top",
                        "{}".format(output_name)))
                save_topview(
                    idx,
                    front_tv,
                    os.path.join(
                        args.out_dir,
                        "front",
                        "{}".format(output_name)))
            elif args.type == "topview":
                tv = models["top_decoder"](features, is_training=False)
                save_topview(
                    idx,
                    tv,
                    os.path.join(
                        args.out_dir,
                        args.type,
                        "{}".format(output_name)))
            elif args.type == "frontview":
                tv = models["front_decoder"](features, is_training=False)
                save_topview(
                    idx,
                    tv,
                    os.path.join(
                        args.out_dir,
                        args.type,
                        "{}".format(output_name)))

    print('-> Done!')


if __name__ == "__main__":
    args = get_args()
    test(args)

