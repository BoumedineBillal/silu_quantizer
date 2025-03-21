import sys
import os
# Add the 'src' directory to sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(src_dir)

import torch
import torch.nn as nn
from quantization.quantization_tools import QuantizeActivation, test_quantization, get_qstat
from approximation.act_approximation_tools import SiluApproximation, test_silu_approximation

class QSiLUApprox(nn.Module):
    """
    Combined module for approximating SiLU with quantization.

    This module applies:
    1. Pre-activation quantization.
    2. SiLU approximation.
    3. Post-activation quantization.

    Optimized for embedded systems.
    """
    def __init__(self):
        super().__init__()
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
        
        # self.silu = nn.SiLU()

    def forward(self, x):
        # --- Fake Quantization Pass (for calibration/monitoring) ---
        with torch.no_grad():
            fake_out = self.pre_activation_quantizer(x)
            fake_out = self.silu_approx(fake_out)
            fake_out = self.post_activation_quantizer(fake_out)

        # --- Real Quantization Pass ---
        x = self.pre_activation_quantizer._clamp_input(x)
        xq = self.pre_activation_quantizer.quantizer.quantize(x)
        xq = self.silu_approx.quantized_activation(xq)
        x = self.post_activation_quantizer.quantizer.dequantize(xq)

        return x


# --- Test for QSiLUApprox ---

def test_qsilu_approx():
    print(f"\n{'='*40}")
    print("Testing QSiLUApprox Module")
    print(f"{'='*40}")

    x = torch.tensor([-10, -4, -1.25, 0.0, 1.0, 4.0, 5.0, 8.0, 40.0, 60.0], dtype=torch.float32)
    
    qsilu = QSiLUApprox()
    
    y_out = qsilu(x)
    
    print(f"Input:  {x}")
    print(f"Output: {y_out}")
    

def test_qsilu_approx_c():
    print(f"\n{'='*40}")
    print("Testing QSiLUApprox Module")
    print(f"{'='*40}")

    x = torch.tensor(
        [-10, -4, -3.5, -3, -2.5, -2, -1.5, -1.25, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 60.0], 
        dtype=torch.float32
    )

    qsilu = QSiLUApprox()

    # --- Fake Quantization Pass (for calibration/monitoring) ---
    with torch.no_grad():
        fake_out = qsilu.pre_activation_quantizer(x)
        fake_out = qsilu.silu_approx(fake_out)
        fake_out = qsilu.post_activation_quantizer(fake_out)

    # --- Real Quantization Pass ---
    x_clamped = qsilu.pre_activation_quantizer._clamp_input(x)
    xq = qsilu.pre_activation_quantizer.quantizer.quantize(x_clamped)
    yq_approx = qsilu.silu_approx.quantized_activation(xq)
    y_dequant = qsilu.post_activation_quantizer.quantizer.dequantize(yq_approx)

    # --- Quantization Parameters ---
    n1 = int(qsilu.pre_activation_quantizer.quantizer.exponent.item())
    n2 = int(qsilu.post_activation_quantizer.quantizer.exponent.item())
    z1 = int(qsilu.pre_activation_quantizer.quantizer.zero_point.item())
    z2 = int(qsilu.post_activation_quantizer.quantizer.zero_point.item())
    s1 = qsilu.pre_activation_quantizer.quantizer.scale.item()
    s2 = qsilu.post_activation_quantizer.quantizer.scale.item()

    # --- Print Results ---
    print(f"Original Input:          {x}")
    print(f"Clamped Input:           {x_clamped}")

    print("\n[Pre-Activation Quantization]")
    print(f"Quantized Input (xq):    {xq}")
    print(f"Scale (s1):              {s1:.6f}")
    print(f"Zero Point (z1):         {z1}")
    print(f"Exponent (n1):           {n1}")

    print("\n[SiLU Approximation (Quantized)]")
    print(f"Quantized Activation:    {yq_approx}")

    print("\n[Post-Activation Quantization]")
    print(f"Scale (s2):              {s2:.6f}")
    print(f"Zero Point (z2):         {z2}")
    print(f"Exponent (n2):           {n2}")

    print(f"\nFinal Dequantized Output: {y_dequant}") 


if __name__ == "__main__":   
    test_silu_approximation()
    test_quantization(8)
    test_quantization(4)
    test_qsilu_approx()
    
    test_qsilu_approx_c()
