import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from test_utils import generate_random_fp16
from fpint_emul import fp16_bit_to_float

# Generate test data
data = generate_random_fp16((2, 4), value_range=(-2.0, 2.0), seed=200)

print("Generated FP16 data (bits):")
print(data)

print("\nConverted to float:")
for i in range(2):
    for j in range(4):
        val = fp16_bit_to_float(data[i, j])
        print(f"[{i},{j}]: 0x{data[i,j]:04x} = {val:.4f}")
