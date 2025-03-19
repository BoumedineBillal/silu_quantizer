import torch
import torch.nn as nn


def arithmetic_right_shift(x, n):
    return torch.floor(x / (2 ** n)) # [3, -3] to tensor([ 1, -2])

class SiluApproximation(nn.Module):
    """
    Approximates the SiLU activation function using fixed-point arithmetic.
    
    Uses a Q{integer_bits}.{fractional_bits} fixed-point format. For example,
    Q4.12 (integer_bits=4, fractional_bits=12) implies a fixed-point scale of 4096.
    """
    def __init__(self, 
                 pre_activation_quantizer = None,
                 post_activation_quantizer = None,
                 integer_bits=4, fractional_bits=12, clamp=None):
        """
        Args:
            integer_bits (int): Number of integer bits.
            fractional_bits (int): Number of fractional bits.
            extra_division (int): Additional division factor required by the approximation.
            clamp (tuple, optional): Clamp range for inputs; defaults to (-4.0, 4.0).
        """
        super().__init__()
        
        self.pre_activation_quantizer = pre_activation_quantizer
        self.post_activation_quantizer = post_activation_quantizer
        
        self.integer_bits = integer_bits
        self.fractional_bits = fractional_bits
        self.fixed_point_scale = 2 ** fractional_bits  # e.g., 4096 for Q4.12

        if clamp is None:
            clamp = (-4.0, 4.0)
        self.clamp_min, self.clamp_max = clamp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Clamp the input within the desired approximation range.
        x_clipped = torch.clamp(x, self.clamp_min, self.clamp_max)
        
        # Convert to fixed-point.
        x_fixed = self.float_to_fixed(x_clipped)
        
        # Use absolute value for the approximation.
        z_fixed = torch.where(x_clipped <= 0, -x_fixed, x_fixed)
        
        # Compute the fixed-point sigmoid approximation.
        y_approx_fixed = self.approx_sigmoid_value(z_fixed)
        
        one_fixed = 1 * self.fixed_point_scale  # Represents 1.0 in fixed-point.
        # Adjust approximation based on the sign.
        y_fixed = torch.where(x_clipped <= 0, y_approx_fixed, one_fixed - y_approx_fixed)
        
        # Convert back to floating point.
        sigmoid_approx = y_fixed.to(torch.float16) / self.fixed_point_scale
        
        # Compute SiLU as x * sigmoid(x).
        silu_approx = x_clipped * sigmoid_approx
        
        # For x > self.clamp_max, return x directly.
        result = torch.where(x > self.clamp_max, x, silu_approx)
        return result

    def float_to_fixed(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a float tensor to fixed-point format."""
        return torch.round(x * self.fixed_point_scale).to(torch.int32)
    
    def approx_sigmoid_value(self, z_fixed: torch.Tensor) -> torch.Tensor:
        """
        Approximate the sigmoid function in fixed-point arithmetic.
        
        g(x) = 0.5*(0.25*x-1)**2
        
        if x<-4: sigmoid(x) = 0
        if -4<=x<=0 : sigmoid(x) = g(-x)
        if 0<=x<=4 : sigmoid(x) = 1 - g(x)
        if 4<x: sigmoid(x) = x
        
        The algorithm:
            1. Shifts right by fractional_bits (dividing by fixed_point_scale).
            2. Subtracts one (in fixed-point, where 1.0 == fixed_point_scale).
            3. Squares the result.
            4. Shifts right by fractional_bits (normalizing after squaring).
            5. Shifts right by log2(extra_division) (for the additional division).
        """
        z = z_fixed >> 2 # Divide by 4 
        
        # Step 2: Subtract 1.0 (in fixed-point representation).
        z = z - 1 * self.fixed_point_scale  
        
        # Step 3: Square the result.
        tmp = z * z  
        
        # Step 4: Normalize by shifting right by fractional_bits.
        tmp = tmp >> self.fractional_bits  
        
        # Step 5: Apply additional division.
        tmp = tmp >> 1 # Divide by 2  
        
        return tmp
    
    def quantized_activation(self, xq):
        """
        Compute the quantized activation using the core snippet.
        
        This method assumes that the input 'xq' is already quantized using the
        pre_activation_quantizer. It does not perform any clamping or dequantization.
        """
        n1 = int(self.pre_activation_quantizer.quantizer.exponent.item())
        n2 = int(self.post_activation_quantizer.quantizer.exponent.item())
        z2 = int(self.post_activation_quantizer.quantizer.zero_point.item())
        
        """
        if n1 <= 2:
            xq2 = torch.round(xq.clone())
            return torch.where(xq <= 0, 0, xq2 * 2**(n2 - n1) + z2)
        """

        xq2 = torch.floor(xq.clone())

        #x_out1 = 2**(n2 - (3 * n1 + 5)) * (xq2 + 2**(2 + n1))**2 * xq2 + z2
        #x_out2 = 2**(n2 - (3 * n1 + 5)) * (2**(2 * n1 + 5) - (xq2 - 2**(2 + n1))**2) * xq2 + z2
        
        #c1 = 2 ** (n2 - (3 * n1 + 5))
        #x_out1 = ( ( (xq2 + 2**(2 + n1))**2 * xq2 ) * c1 ) + z2
        #x_out2 = 2**(n2 - (3 * n1 + 5)) * (2**(2 * n1 + 5) - (xq2 - 2**(2 + n1))**2) * xq2 + z2
        
        c1 = 2**(n2 - (3 * n1 + 5))
        c2 = 2**(2 + n1)
        c3 = 2**(2 * n1 + 5)
        c4 = 2**(n2 - n1)
        c5 = 4 * 2**n1
        
        
        x_out1 = c1 * (xq2 + c2)**2 * xq2 + z2
        x_out2 = c1 * (c3 - (xq2 - c2)**2) * xq2 + z2
        
        
        x_out = torch.where(xq <= 0, x_out1, x_out2)
        x_out = torch.where(xq > c5, xq * c4 + z2, x_out)
        
        return torch.floor(x_out)
    
    def quantized_activation_(self, xq):
        """
        Compute the quantized activation using the core snippet.
        
        This method assumes that the input 'xq' is already quantized using the
        pre_activation_quantizer. It does not perform any clamping or dequantization.
        """
        n1 = self.pre_activation_quantizer.quantizer.exponent
        n2 = self.post_activation_quantizer.quantizer.exponent
        z2 = self.post_activation_quantizer.quantizer.zero_point
        
        """
        if n1 <= 2:
            xq2 = torch.round(xq.clone())
            return torch.where(xq <= 0, 0, xq2 * 2**(n2 - n1) + z2)
        """

        xq2 = torch.round(xq.clone())

        x_out1 = 2**(n2 - (3 * n1 + 5)) * (xq2 + 2**(2 + n1))**2 * xq + z2
        x_out2 = 2**(n2 - (3 * n1 + 5)) * (2**(2 * n1 + 5) - (xq2 - 2**(2 + n1))**2) * xq + z2

        x_out = torch.where(xq <= 0, x_out1, x_out2)
        x_out = torch.where(xq > 4 * 2**n1, xq * 2**(n2 - n1) + z2, x_out)
        return x_out

    def extra_repr(self) -> str:
        return (f"Q{self.integer_bits}.{self.fractional_bits} fixed-point, "
                f"clamp=({self.clamp_min}, {self.clamp_max})")

# --- Test Function ---

def test_silu_approximation():
    print(f"\n{'='*40}")
    print("Testing SiLU Approximation with Q4.12 Fixed-Point Format")
    print(f"{'='*40}")
    
    # Sample input tensor.
    x_tensor = torch.tensor(
        [-10, -4, -1.25, 0.0, 1.0, 4.0, 5.0, 8.0, 40.0, 60.0], dtype=torch.float32
    )
    
    # Instantiate the approximator.
    silu_approximator = SiluApproximation(integer_bits=4, fractional_bits=12)
    
    # Compute the approximated SiLU activation.
    y_approx = silu_approximator(x_tensor)
    
    print(f"Input:             {x_tensor}")
    print(f"Approximated SiLU: {y_approx}")
    


if __name__ == "__main__":
    test_silu_approximation()
