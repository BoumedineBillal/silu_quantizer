from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
import os


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
    def forward(self, x):
        return sigmoid_approximation_single_torch(x)
    

#########################
# simple activation quantizer on the fly
#########################

"""
import torch
import torch.nn as nn

class QuantizeActivation(nn.Module):
    def __init__(self, bit_width=8):
        super().__init__()
        self.bit_width = bit_width
        self.scale = None
        self.zero_point = None

    def forward(self, x):

        x = torch.where(x > -4, x, torch.tensor(-4.0, device=x.device)) #x = torch.where(x > -4, x, torch.tensor(0.0, device=x.device)) # hack -4 to 0 to be 0 after act similer to silu approximation
        #x = torch.where(x < 10, x, torch.tensor(10.0, device=x.device))

        # Calculate quantization range
        quant_max = (2 ** self.bit_width / 2) - 1
        quant_min = -(2 ** self.bit_width / 2)

        # Calculate 98% dynamic range
        q_min = torch.min(x.detach())
        q_max = torch.max(x.detach()) #torch.sort(x.detach().flatten())[0][int((1-0.0005) * (x.numel() - 1))] #torch.quantile(x.detach(), 0.99)

        #print(q_min, x.detach().min())
        #print(q_max, x.detach().max())

        # Handle case where all values are identical
        if q_min >= q_max:
            q_max = q_min + 1e-5

        # Calculate scale and zero-point
        self.scale = (q_max - q_min) / (quant_max - quant_min)
        self.zero_point = torch.round(quant_min - q_min / self.scale)

        # Clamp zero-point to valid range
        #self.zero_point = torch.clamp(self.zero_point, quant_min, quant_max).int()

        # Quantization process
        x_clamped = torch.clamp(x, q_min, q_max)
        x_quant = torch.round(x_clamped / self.scale + self.zero_point.float())
        x_quant = torch.clamp(x_quant, quant_min, quant_max)

        # Dequantization
        x_dequant = (x_quant - self.zero_point.float()) * self.scale

        self.x_quant = x_quant
        return x_dequant

# Test with 8-bit
bit_width=8
quantizer = QuantizeActivation(bit_width=bit_width)
x = torch.randn(2, 3, 32, 32) * 5  # Test data with large variance
x_dequant = quantizer(x)
x_quant = quantizer.x_quant

# Check quantization bounds
assert x_quant.min() >= -(2 ** bit_width / 2) , f"Min value {x_quant.min().item()} < 0"
assert x_quant.max() <= (2 ** bit_width / 2) - 1, f"Max value {x_quant.max().item()} > 255"

# Verify scale/zero-point calculations
print(f"Scale: {quantizer.scale.item():.4f}")
print(f"Zero point: {quantizer.zero_point.item()}")
print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
print(f"Quantized range: [{x_quant.min().item():.4f}, {x_quant.max().item():.4f}]")
print("Test passed!")
"""


"""
#########################
# simple activation quantizer on the fly ( pow of 2 ) 
#########################


import torch
import torch.nn as nn

exponents = set()
z = set()

class QuantizeActivation(nn.Module):
    def __init__(self, bit_width=8):
        super().__init__()
        self.bit_width = bit_width
        self.scale = None
        self.zero_point = None

    def forward(self, x):
        x = torch.where(x > -4, x, torch.tensor(-4.0, device=x.device))  # Clamp values below -4 to -4
        #x = torch.where(x < 10, x, torch.tensor(10.0, device=x.device))  # Clamp values above 10 to 10 (optional)

        # Calculate quantization range
        quant_max = (2 ** self.bit_width / 2) - 1
        quant_min = -(2 ** self.bit_width / 2)

        # Calculate 98% dynamic range
        q_min = torch.min(x.detach())
        q_max = torch.max(x.detach())

        # Handle case where all values are identical
        if q_min >= q_max:
            q_max = q_min + 1e-5

        # Calculate scale as a power of 2
        dynamic_range = q_max - q_min
        log2_scale = torch.log2(dynamic_range / (quant_max - quant_min))
        n = torch.ceil(log2_scale)
        if torch.isinf(n):
            print("Warning: log2_scale resulted in infinity!")
            n = 0  # Or another default value to prevent crashes
        exponents.add(int(n))
        self.scale = 2 ** n  # Ensure scale is a power of 2
        

        # Calculate zero-point
        self.zero_point = torch.round(quant_min - q_min / self.scale)
        z.add((int(n), int(self.zero_point)))

        # Quantization process
        x_clamped = torch.clamp(x, q_min, q_max)
        x_quant = torch.round(x_clamped / self.scale + self.zero_point.float())
        x_quant = torch.clamp(x_quant, quant_min, quant_max)

        # Dequantization
        x_dequant = (x_quant - self.zero_point.float()) * self.scale

        self.x_quant = x_quant
        return x_dequant

# Test with 8-bit
bit_width = 8
quantizer = QuantizeActivation(bit_width=bit_width)
x = torch.randn(2, 3, 32, 32) * 5  # Test data with large variance
x_dequant = quantizer(x)
x_quant = quantizer.x_quant

# Check quantization bounds
assert x_quant.min() >= -(2 ** bit_width / 2), f"Min value {x_quant.min().item()} < 0"
assert x_quant.max() <= (2 ** bit_width / 2) - 1, f"Max value {x_quant.max().item()} > 255"

# Verify scale/zero-point calculations
print(f"Scale (power of 2): {quantizer.scale.item():.4f}")
print(f"Zero point: {quantizer.zero_point.item()}")
print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
print(f"Quantized range: [{x_quant.min().item():.4f}, {x_quant.max().item():.4f}]")
print("Test passed!")
"""

#########################
# simple activation quantizer on the fly ( pow of 2 ), optionel zero point
#########################

import torch
import torch.nn as nn

exponents = set()
z = set()

class QuantizeActivation(nn.Module):
    def __init__(self, bit_width=8, use_zero_point=True):
        super().__init__()
        self.bit_width = bit_width
        self.use_zero_point = use_zero_point  # Enable/disable zero-point
        self.scale = None
        self.zero_point = None

    def asymmetric_quantize(self, x):
        # Calculate quantization range
        quant_max = (2 ** self.bit_width / 2) - 1
        quant_min = -(2 ** self.bit_width / 2)

        # Calculate dynamic range
        q_min = torch.min(x.detach())
        q_max = torch.max(x.detach())

        # Handle case where all values are identical
        if q_min >= q_max:
            q_max = q_min + 1e-5

        # Calculate scale as a power of 2
        dynamic_range = q_max - q_min
        log2_scale = torch.log2(dynamic_range / (quant_max - quant_min))
        n = torch.ceil(log2_scale)
        if torch.isinf(n):
            print("Warning: log2_scale resulted in infinity!")
            n = 0  # Default value to prevent crashes
        exponents.add(int(n))
        self.scale = 2 ** n  # Ensure scale is a power of 2

        # Calculate zero-point
        self.zero_point = torch.round(quant_min - q_min / self.scale)
        z.add((int(n), int(self.zero_point)))

        # Quantization process
        x_clamped = torch.clamp(x, q_min, q_max)
        x_quant = torch.round(x_clamped / self.scale + self.zero_point.float())
        x_quant = torch.clamp(x_quant, quant_min, quant_max)

        # Dequantization
        x_dequant = (x_quant - self.zero_point.float()) * self.scale

        return x_dequant, x_quant

    def symmetric_quantize(self, x):
        # Calculate quantization range
        quant_max = (2 ** self.bit_width / 2) - 1
        quant_min = -(2 ** self.bit_width / 2)

        # Calculate dynamic range (symmetric around 0)
        max_abs = torch.max(torch.abs(x.detach()))
        q_min = -max_abs
        q_max = max_abs

        # Handle case where all values are identical
        if q_min >= q_max:
            q_max = q_min + 1e-5

        # Calculate scale as a power of 2
        dynamic_range = q_max - q_min
        log2_scale = torch.log2(dynamic_range / (quant_max - quant_min))
        n = torch.ceil(log2_scale)
        if torch.isinf(n):
            print("Warning: log2_scale resulted in infinity!")
            n = 0  # Default value to prevent crashes
        exponents.add(int(n))
        self.scale = 2 ** n  # Ensure scale is a power of 2

        # Zero-point is 0 for symmetric quantization
        self.zero_point = torch.tensor(0.0, device=x.device)
        z.add((int(n), int(self.zero_point)))

        # Quantization process
        x_clamped = torch.clamp(x, q_min, q_max)
        x_quant = torch.round(x_clamped / self.scale + self.zero_point.float())
        x_quant = torch.clamp(x_quant, quant_min, quant_max)

        # Dequantization
        x_dequant = (x_quant - self.zero_point.float()) * self.scale

        return x_dequant, x_quant

    def forward(self, x):
        x = torch.where(x > -4, x, torch.tensor(-4.0, device=x.device))  # Clamp values below -4 to -4
        #x = torch.where(x < 10, x, torch.tensor(10.0, device=x.device))  # Clamp values above 10 to 10 (optional)

        if self.use_zero_point:
            x_dequant, x_quant = self.asymmetric_quantize(x)
        else:
            x_dequant, x_quant = self.symmetric_quantize(x)

        self.x_quant = x_quant
        return x_dequant

# Test with 8-bit
bit_width = 8

# Test asymmetric quantization (with zero-point)
print("Testing asymmetric quantization (with zero-point):")
quantizer_asymmetric = QuantizeActivation(bit_width=bit_width, use_zero_point=True)  # Enable zero-point
x = torch.randn(2, 3, 32, 32) * 5  # Test data with large variance
x_dequant_asymmetric = quantizer_asymmetric(x)
x_quant_asymmetric = quantizer_asymmetric.x_quant

# Check quantization bounds
assert x_quant_asymmetric.min() >= -(2 ** bit_width / 2), f"Min value {x_quant_asymmetric.min().item()} < 0"
assert x_quant_asymmetric.max() <= (2 ** bit_width / 2) - 1, f"Max value {x_quant_asymmetric.max().item()} > 255"

# Verify scale/zero-point calculations
print(f"Scale (power of 2): {quantizer_asymmetric.scale.item():.4f}")
print(f"Zero point: {quantizer_asymmetric.zero_point.item()}")
print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
print(f"Quantized range: [{x_quant_asymmetric.min().item():.4f}, {x_quant_asymmetric.max().item():.4f}]")
print("Asymmetric quantization test passed!\n")

# Test symmetric quantization (without zero-point)
print("Testing symmetric quantization (without zero-point):")
quantizer_symmetric = QuantizeActivation(bit_width=bit_width, use_zero_point=False)  # Disable zero-point
x_dequant_symmetric = quantizer_symmetric(x)
x_quant_symmetric = quantizer_symmetric.x_quant

# Check quantization bounds
assert x_quant_symmetric.min() >= -(2 ** bit_width / 2), f"Min value {x_quant_symmetric.min().item()} < 0"
assert x_quant_symmetric.max() <= (2 ** bit_width / 2) - 1, f"Max value {x_quant_symmetric.max().item()} > 255"

# Verify scale/zero-point calculations
print(f"Scale (power of 2): {quantizer_symmetric.scale.item():.4f}")
print(f"Zero point: {quantizer_symmetric.zero_point.item()}")
print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
print(f"Quantized range: [{x_quant_symmetric.min().item():.4f}, {x_quant_symmetric.max().item():.4f}]")
print("Symmetric quantization test passed!")



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
        module.act = nn.Sequential(QuantizeActivation(bit_width=8, use_zero_point=False),
                                   SigmoidApproximationActivation(),
                                   QuantizeActivation(bit_width=8))


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

        
