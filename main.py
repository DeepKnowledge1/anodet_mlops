import os
import anodet
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import argparse


# DATASET_PATH = os.path.realpath(r"D:\01-DATA\bottle")
# MODEL_DATA_PATH = os.path.realpath("./distributions/")
# os.makedirs(MODEL_DATA_PATH,exist_ok=True)


# dataset = anodet.AnodetDataset(os.path.join(DATASET_PATH, "train/good"))
# dataloader = DataLoader(dataset, batch_size=2)
# print("Number of images in dataset:", len(dataloader.dataset))


# padim = anodet.Padim(backbone='resnet18')

# padim.fit(dataloader)

# padim.export_onnx("padim_model.onnx")



def parse_args():
    parser = argparse.ArgumentParser(description="Train a PaDiM model for anomaly detection.")

    parser.add_argument('--dataset_path',default=r"D:\01-DATA\bottle", type=str, required=False,
                        help='Path to the dataset folder containing "train/good" images.')

    parser.add_argument('--model_data_path', type=str, default='./distributions/',
                        help='Directory to save model distributions and ONNX file.')

    parser.add_argument('--backbone', type=str, choices=['resnet18', 'wide_resnet50'], default='resnet18',
                        help='Backbone network to use for feature extraction.')

    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size used during training and inference.')

    parser.add_argument('--output_model', type=str, default='padim_model.onnx',
                        help='Filename to save the exported ONNX model.')

    parser.add_argument('--layer_indices', nargs='+', type=int, default=[0],
                        help='List of layer indices to extract features from. Default: [0].')

    parser.add_argument('--feat_dim', type=int, default=50,
                        help='Number of random feature dimensions to keep.')

    return parser.parse_args()


def main(args):
    # Set up paths
    DATASET_PATH = os.path.realpath(args.dataset_path)
    MODEL_DATA_PATH = os.path.realpath(args.model_data_path)
    os.makedirs(MODEL_DATA_PATH, exist_ok=True)

    # Load dataset
    dataset = anodet.AnodetDataset(os.path.join(DATASET_PATH, "train/good"))
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    print("Number of images in dataset:", len(dataloader.dataset))

    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model with args
    padim = anodet.Padim(
        backbone=args.backbone,
        device=device,
        layer_indices=args.layer_indices,
        feat_dim=args.feat_dim
    )

    # Train model
    padim.fit(dataloader)

    # Export model to ONNX
    onnx_path = os.path.join(MODEL_DATA_PATH, args.output_model)
    padim.export_onnx(onnx_path)
    print(f"Model saved to {onnx_path}")
    
if __name__ == "__main__":
        args = parse_args()
        main(args)
