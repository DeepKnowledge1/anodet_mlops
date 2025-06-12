import os
import anodet
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


DATASET_PATH = os.path.realpath(r"D:\01-DATA\bottle")
MODEL_DATA_PATH = os.path.realpath("./distributions/")
os.makedirs(MODEL_DATA_PATH,exist_ok=True)


dataset = anodet.AnodetDataset(os.path.join(DATASET_PATH, "train/good"))
dataloader = DataLoader(dataset, batch_size=2)
print("Number of images in dataset:", len(dataloader.dataset))


padim = anodet.Padim(backbone='resnet18')

padim.fit(dataloader)

padim.export_onnx("padim_model.onnx")




# import os
# from pathlib import Path

# # Define the directory structure
# structure = {
#     "industrial_anodet_mlops": [
#         "data",
#         "src/model",
#         "src",
#         "deployment",
#         "pipelines",
#         "docker",
#         "devops",
#         "monitoring",
#         "keyvault",
#         "load_testing"
#     ]
# }

# # Files to create inside directories
# files_to_create = {
#     "src/model": ["padim.py", "utils.py"],
#     "src": ["train.py", "inference.py", "fastapi_app.py", "logger.py", "monitor_drift.py", "metrics_logger.py"],
#     "deployment": ["score.py", "environment.yml", "deployment.yml", "aks_compute.yml"],
#     "pipelines": ["train_pipeline.yml", "deploy_pipeline.yml", "retrain_pipeline.yml"],
#     "docker": ["Dockerfile.train", "Dockerfile.inference"],
#     "devops": ["azure-pipelines.yml", "env-variables-template.yml"],
#     "monitoring": ["app_insights_setup.py", "log_config.yaml", "drift_alert_setup.py"],
#     "keyvault": ["setup_keyvault.py", "secrets_template.json"],
#     "load_testing": ["locustfile.py"],
#     "industrial_anodet_mlops": ["README.md"]
# }

# # Create directories and files
# for base, dirs in structure.items():
#     base_path = Path(base)
#     for sub in dirs:
#         dir_path = base_path / sub
#         dir_path.mkdir(parents=True, exist_ok=True)

# # Create files
# for dir_path, file_list in files_to_create.items():
#     for file in file_list:
#         file_path = Path("industrial_anodet_mlops") / dir_path / file
#         file_path.touch(exist_ok=True)

# print("Project structure created successfully.")
