import torch
import torch.nn as nn
from ..quantization.quantization_tools import QuantizeActivation, test_quantization, get_qstat
from ..approximation.act_approximation_tools import SiluApproximation, test_silu_approximation

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

    def forward(self, x):
        # --- Fake Quantization Pass (for calibration/monitoring) ---
        with torch.no_grad():
            fake_out = self.pre_activation_quantizer(x)
            fake_out = self.silu_approx(fake_out)
            fake_out = self.post_activation_quantizer(fake_out)

        # --- Real Quantization Pass ---
        x = self.pre_activation_quantizer._clamp_input(x)
        xq = self.pre_activation_quantizer.quantize(x)
        xq = self.silu_approx.quantized_activation(xq)
        x = self.post_activation_quantizer.quantize(xq)

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


if __name__ == "__main__":  
    test_silu_approximation()
    test_quantization(8)
    test_quantization(4)
    test_qsilu_approx()
