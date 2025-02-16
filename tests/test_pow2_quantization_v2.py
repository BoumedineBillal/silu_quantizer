from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
import os

from tools.quantization_tools import QuantizeActivation, test_quantization


#########################
# sigmoid aproximation
#########################

def float_to_fixed(x):
    """Convert a float tensor to 4.12 fixed-point format."""
    return torch.round(x * 4096).to(torch.int32)

def approx_sigmoid_value(z_fixed):
    """Approximate the sigmoid function in fixed-point using PyTorch."""
    z = z_fixed >> 2  # Divide by 4
    z = z - 0x1000    # Subtract 1.0 in 4.12 format (4096)
    tmp = z * z       # Square the result
    tmp = tmp >> 12   # Divide by 4096
    tmp = tmp >> 1    # Divide by 2
    return tmp

def sigmoid_approximation_single_torch(x):
    """Compute the approximated sigmoid for a single torch.Tensor input."""
    # Initialize y as x for cases where x > 4
    y = torch.where(x > 4, x, torch.tensor(0.0, device=x.device))

    # Clip the input tensor to the range [-4, 4] for approximation
    x_clipped = torch.clamp(x, min=-4.0, max=4.0)

    # Convert to fixed-point format
    x_fixed = float_to_fixed(x_clipped)

    # Compute sigmoid approximation based on sign of input
    z_fixed = torch.where(
        x_clipped <= 0,
        -x_fixed,  # Use positive input for approximation
        x_fixed
    )

    y_approx_fixed = approx_sigmoid_value(z_fixed)
    y_fixed = torch.where(
        x_clipped <= 0,
        y_approx_fixed,
        0x1000 - y_approx_fixed  # 1.0 - approx_value in fixed-point
    )

    # Convert back to floating-point and scale by input
    y_approx = y_fixed.to(torch.float32) / 4096.0
    y += torch.where(x > 4, torch.tensor(0.0, device=x.device), y_approx * x_clipped)  # Add approximation result for x <= 4
    #y = torch.where(y < 20, y, torch.tensor(20.0, device=x.device)) # clip >20
    return y

# Input tensor
x_tensor = torch.tensor([-10, -4, -1.25, 0.0, 1.0, 4.0, 5.0, 8.0, 40.0, 60.0])

# Apply the PyTorch sigmoid approximation function
y_approx = sigmoid_approximation_single_torch(x_tensor)
print(f"Input: {x_tensor}")
print(f"Approximated Sigmoid: {y_approx}")


import torch.nn as nn

# Define a custom activation module
class SigmoidApproximationActivation(nn.Module):
    def __init__(self, prev_ref, aftr_ref):
        super().__init__()
        self.prev_ref = prev_ref
        self.aftr_ref = aftr_ref
        
    def forward(self, x):
        return sigmoid_approximation_single_torch(x)
    



#########################
# simple activation quantizer on the fly ( pow of 2 ), optionel zero point
#########################


test_quantization(bit_width=8)
test_quantization(bit_width=4)



#########################
# yolo evaluation
#########################


# Load model and set to evaluation mode
model = YOLO('yolov5n.pt').eval().to('cuda' if torch.cuda.is_available() else 'cpu')



# Replace activation in Conv blocks only
for name, module in model.model.named_modules():
    # Check if it's a Conv module with activation
    if hasattr(module, 'act'):
        # Print original activation
        """
        target_bn_layers = ['model.0', 'model.1']
        if name not in target_bn_layers:
            continue
        """

        print(f"Conv block {name}: Changing activation from {module.act} to new")

        # Replace activation
        #module.act = SigmoidApproximationActivation()# negative value clip impact:nn.Sequential(module.act, nn.ReLU(inplace=True)) #nn.ReLU(inplace=True)
        #module.act = nn.Sequential(QuantizeActivation(bit_width=8), module.act)
        #module.act = nn.Sequential(QuantizeActivation(bit_width=8), SigmoidApproximationActivation())
        #module.act = nn.Sequential(QuantizeActivation(bit_width=8), SigmoidApproximationActivation(), QuantizeActivation(bit_width=8))
        #module.act = nn.Sequential(QuantizeActivation(bit_width=8), module.act)
        
        a = QuantizeActivation(bit_width=8, use_zero_point=False)
        c = QuantizeActivation(bit_width=8)
        b = SigmoidApproximationActivation(a, c)
        
        module.act = nn.Sequential(a, b, c)
        
        


#if __name__ == '__main__':
# Evaluate on COCO val (requires dataset setup)
"""
results = model.val(
    data='coco.yaml',
    batch=32,
    imgsz=640,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    half=True, 
    workers=14,
)
"""

        
