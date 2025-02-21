from ultralytics import YOLO
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
import os

from tools.quantization_tools import QuantizeActivation, test_quantization
from tools.act_approximation_tools import SiluApproximation, test_silu_approximation


#########################
# sigmoid aproximation
#########################

test_silu_approximation()




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



class NewAct(nn.Module):
    def __init__(self):
        super().__init__()
        # Use descriptive names for clarity.
        self.pre_activation_quantizer = QuantizeActivation(
            bit_width=8, clamp_values=(-4.0, None), use_zero_point=False
        )
        self.post_activation_quantizer = QuantizeActivation(
            bit_width=8, clamp_values=(None, None)
        )
        self.silu_approx = SiluApproximation(
            pre_activation_quantizer=self.pre_activation_quantizer,
            post_activation_quantizer=self.post_activation_quantizer
        )
        
    def forward(self, x):
        # --- Fake Quantization Pass (for calibration/monitoring) ---
        # Use no_grad to avoid storing intermediate states.
        with torch.no_grad():
            fake_out = self.pre_activation_quantizer(x)
            fake_out = self.silu_approx(fake_out)
            fake_out = self.post_activation_quantizer(fake_out)
            
        
        # --- Real Quantization Pass ---
        # Perform clamping and quantization
        x = self.pre_activation_quantizer._clamp_input(x)
        xq = self.pre_activation_quantizer.quantizer.quantize(x)
        
        # Use the core quantized activation from SiluApproximation.
        xq = self.silu_approx.quantized_activation(xq)
        
        x = self.post_activation_quantizer.quantizer.dequantize(xq)
        
        return x


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
        
        
        module.act = NewAct()
        
        


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

        
