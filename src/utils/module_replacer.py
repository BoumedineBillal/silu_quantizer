import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch.nn as nn
from QSiLUApprox.QSiLUApprox import QSiLUApprox


def replace_module(model, target_module, new_module, attr_list):
    """
    Replace instances of a target module in a PyTorch model if their attribute name starts with an entry from attr_list.

    Args:
        model (nn.Module): The model to modify.
        target_module (type): The module type to replace (e.g., nn.SiLU).
        new_module (type or callable): The replacement module or a callable that returns a new instance.
        attr_list (list of str): List of attribute name prefixes to match (e.g., ["act"]).

    Returns:
        nn.Module: The modified model with replacements.
    """
    for name, module in model.named_modules():
        for attr_name, sub_module in module._modules.items():
            if any(attr_name.startswith(attr) for attr in attr_list) and isinstance(sub_module, target_module):
                print(f"Replacing {name}.{attr_name}: {target_module.__name__} → {new_module().__class__.__name__}")

                # Replace the module
                setattr(module, attr_name, new_module() if callable(new_module) else new_module)


# # Replace activation in Conv blocks only
# for name, module in model.model.named_modules():
#     # Check if it's a Conv module with activation
#     if hasattr(module, 'act'):
#         # Print original activation
#         """
#         target_bn_layers = ['model.0', 'model.1']
#         if name not in target_bn_layers:
#             continue
#         """
#         print(f"Conv block {name}: Changing activation from {module.act} to new")
#         # Replace activation
#         #module.act = SigmoidApproximationActivation()# negative value clip impact:nn.Sequential(module.act, nn.ReLU(inplace=True)) #nn.ReLU(inplace=True)
#         #module.act = nn.Sequential(QuantizeActivation(bit_width=8), module.act)
#         #module.act = nn.Sequential(QuantizeActivation(bit_width=8), SigmoidApproximationActivation())
#         #module.act = nn.Sequential(QuantizeActivation(bit_width=8), SigmoidApproximationActivation(), QuantizeActivation(bit_width=8))
#         #module.act = nn.Sequential(QuantizeActivation(bit_width=8), module.act)
#         module.act = QSiLUApprox()


# --- Test 1: Replace SiLU with QSiLUApprox ---
def test_replace_silu():
    print(f"\n{'='*40}")
    print("Test 1: Replacing SiLU with QSiLUApprox")
    print(f"{'='*40}")

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.act1 = nn.SiLU()
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.act2 = nn.ReLU()
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.act3 = nn.SiLU()

        def forward(self, x):
            x = self.conv1(x)
            x = self.act1(x)
            x = self.conv2(x)
            x = self.act2(x)
            x = self.conv3(x)
            x = self.act3(x)
            return x

    model = TestModel()
    print("\n[Before Replacement]")
    print(model)

    model = replace_module(model, nn.SiLU, QSiLUApprox, ["act"])

    print("\n[After Replacement]")
    print(model)

    assert isinstance(model.act1, QSiLUApprox), "act1 was not replaced!"
    assert isinstance(model.act3, QSiLUApprox), "act3 was not replaced!"
    assert isinstance(model.act2, nn.ReLU), "act2 should NOT be replaced!"

    print("\n✅ Test 1 Passed: SiLU replaced successfully!")


# --- Test 2: Ensure Other Activations Remain Unchanged ---
def test_preserve_other_activations():
    print(f"\n{'='*40}")
    print("Test 2: Ensuring Other Activations Are Not Modified")
    print(f"{'='*40}")

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.act1 = nn.ReLU()
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.act2 = nn.LeakyReLU()
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.act3 = nn.SiLU()  # Only this should be replaced

        def forward(self, x):
            x = self.conv1(x)
            x = self.act1(x)
            x = self.conv2(x)
            x = self.act2(x)
            x = self.conv3(x)
            x = self.act3(x)
            return x

    model = TestModel()
    print("\n[Before Replacement]")
    print(model)

    model = replace_module(model, nn.SiLU, QSiLUApprox, ["act"])

    print("\n[After Replacement]")
    print(model)

    assert isinstance(model.act3, QSiLUApprox), "act3 (SiLU) was not replaced!"
    assert isinstance(model.act1, nn.ReLU), "act1 (ReLU) should NOT be replaced!"
    assert isinstance(model.act2, nn.LeakyReLU), "act2 (LeakyReLU) should NOT be replaced!"

    print("\n✅ Test 2 Passed: Other activations remain unchanged!")


# --- Test 3: Replace SiLU Inside nn.Sequential ---
def test_replace_silu_in_sequential():
    print(f"\n{'='*40}")
    print("Test 3: Replacing SiLU Inside nn.Sequential")
    print(f"{'='*40}")

    class SequentialModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.SiLU()
            )

        def forward(self, x):
            return self.model(x)

    model = SequentialModel()
    print("\n[Before Replacement]")
    print(model)

    model = replace_module(model, nn.SiLU, QSiLUApprox, [""])  # [""] important for number atr

    print("\n[After Replacement]")
    print(model)

    silu_count = sum(1 for module in model.model if isinstance(module, QSiLUApprox))
    relu_count = sum(1 for module in model.model if isinstance(module, nn.ReLU))

    assert silu_count == 2, f"Expected 2 QSiLUApprox, but found {silu_count}!"
    assert relu_count == 1, "ReLU should NOT be replaced!"

    print("\n✅ Test 3 Passed: SiLU replaced inside nn.Sequential!")


# --- Run Tests ---
if __name__ == "__main__":
    test_replace_silu()
    test_preserve_other_activations()
    test_replace_silu_in_sequential()


