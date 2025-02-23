from ultralytics import YOLO
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
import os

from quantization.quantization_tools import QuantizeActivation, test_quantization, get_qstat
from approximation.act_approximation_tools import SiluApproximation, test_silu_approximation
from QSiLUApprox.QSiLUApprox import QSiLUApprox
from utils.module_replacer import replace_module



# Load model and set to evaluation mode
model = YOLO('yolov5n.pt').eval().to('cuda' if torch.cuda.is_available() else 'cpu')


replace_module(model.model, nn.SiLU, QSiLUApprox, ["act"])
        
        


results = model.val(
    data='coco.yaml',
    batch=32,
    imgsz=640,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    half=True, 
    workers=14,
)













        
