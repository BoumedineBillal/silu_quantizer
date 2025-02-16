import torch
import torch.nn as nn

class RangeCollector:
    """Handles dynamic range calculation for quantization"""
    def __init__(self, mode='asymmetric'):
        self.mode = mode
        self.q_min = None
        self.q_max = None
        
    def collect_range(self, x):
        """Calculate dynamic range based on collection mode"""
        x_detached = x.detach()
        
        if self.mode == 'asymmetric':
            self.q_min = torch.min(x_detached)
            self.q_max = torch.max(x_detached)
        elif self.mode == 'symmetric':
            max_abs = torch.max(torch.abs(x_detached))
            self.q_min = -max_abs
            self.q_max = max_abs
            
        # Handle edge case where all values are identical
        if self.q_min >= self.q_max:
            self.q_max = self.q_min + 1e-5
            
        return self.q_min, self.q_max

class Quantizer:
    """Handles quantization/dequantization operations"""
    def __init__(self, bit_width, use_zero_point=True):
        self.bit_width = bit_width
        self.use_zero_point = use_zero_point
        self.quant_min = -(2 ** (bit_width - 1))
        self.quant_max = (2 ** (bit_width - 1)) - 1
        self.scale = None
        self.zero_point = None

    def calculate_scale(self, q_min, q_max):
        """Calculate power-of-2 scale factor"""
        dynamic_range = q_max - q_min
        target_range = self.quant_max - self.quant_min
        
        if dynamic_range <= 0:
            self.scale = torch.tensor(1.0)
            return
            
        log2_scale = torch.log2(dynamic_range / target_range)
        n = torch.ceil(log2_scale)
        self.scale = 2 ** torch.where(torch.isinf(n), torch.tensor(0.0), n)

    def calculate_zero_point(self, q_min):
        """Calculate zero-point for asymmetric quantization"""
        if self.use_zero_point:
            self.zero_point = torch.round(self.quant_min - q_min / self.scale)
        else:
            self.zero_point = torch.tensor(0.0)

    def quantize(self, x, q_min, q_max):
        """Perform quantization with clamping"""
        x_clamped = torch.clamp(x, q_min, q_max)
        x_quant = torch.round(x_clamped / self.scale + self.zero_point.to(x.device))
        return torch.clamp(x_quant, self.quant_min, self.quant_max)

    def dequantize(self, x_quant):
        """Convert quantized values back to float"""
        return (x_quant - self.zero_point.to(x_quant.device)) * self.scale.to(x_quant.device)

class QuantizeActivation(nn.Module):
    """Modular Power-of-2 Activation Quantizer"""
    def __init__(self, bit_width=8, use_zero_point=True, clamp_values=(-4.0, None)):
        super().__init__()
        self.bit_width = bit_width
        self.use_zero_point = use_zero_point
        self.clamp_min, self.clamp_max = clamp_values
        
        # Initialize components
        self.range_collector = RangeCollector('asymmetric' if use_zero_point else 'symmetric')
        self.quantizer = Quantizer(bit_width, use_zero_point)
        self.x_quant = None

    def forward(self, x):
        # Input preprocessing
        x = self._clamp_input(x)
        
        # Dynamic range collection
        q_min, q_max = self.range_collector.collect_range(x)
        
        # Quantization parameters calculation
        self.quantizer.calculate_scale(q_min, q_max)
        self.quantizer.calculate_zero_point(q_min)
        
        # Quantization process
        x_quant = self.quantizer.quantize(x, q_min, q_max)
        x_dequant = self.quantizer.dequantize(x_quant)
        
        # Store quantized values
        self.x_quant = x_quant.detach()
        
        return x_dequant

    def _clamp_input(self, x):
        """Apply input clamping"""
        x = torch.where(x > self.clamp_min, x, torch.tensor(self.clamp_min, device=x.device))
        if self.clamp_max is not None:
            x = torch.where(x < self.clamp_max, x, torch.tensor(self.clamp_max, device=x.device))
        return x

    def extra_repr(self):
        return (f"bit_width={self.bit_width}, "
                f"use_zero_point={self.use_zero_point}, "
                f"clamp=({self.clamp_min}, {self.clamp_max})")

# Test Suite remains similar but uses the new class structure
def test_quantization(bit_width=8):
    print(f"\n{'='*40}")
    print(f"Testing {bit_width}-bit Quantization")
    print(f"{'='*40}")
    
    x = torch.randn(2, 3, 32, 32) * 5
    
    # Test asymmetric
    print("\n[1] Asymmetric Quantization:")
    quantizer = QuantizeActivation(bit_width, use_zero_point=True)
    x_dequant = quantizer(x)
    print(f"Scale: {quantizer.quantizer.scale.item():.4f}")
    print(f"Zero Point: {quantizer.quantizer.zero_point.item()}")
    print(f"Quantized Range: [{quantizer.x_quant.min().item()}, {quantizer.x_quant.max().item()}]")
    
    # Test symmetric
    print("\n[2] Symmetric Quantization:")
    quantizer = QuantizeActivation(bit_width, use_zero_point=False)
    x_dequant = quantizer(x)
    print(f"Scale: {quantizer.quantizer.scale.item():.4f}")
    print(f"Zero Point: {quantizer.quantizer.zero_point.item()}")
    print(f"Quantized Range: [{quantizer.x_quant.min().item()}, {quantizer.x_quant.max().item()}]")
    
    print("\nAll tests passed!")




if __name__ == "__main__":  

    test_quantization(8)
    test_quantization(4)









