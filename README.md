# SiLU Quantizer for Embedded AI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BoumedineBillal/silu_quantizer/blob/main/notebooks/benchmark_qsilu.ipynb)

![QSiLUApprox](assets/output.png)

This project provides an efficient approximation of the **SiLU activation function** optimized for **quantized inference** on embedded devices, targeting the **ESP32-P4** and **ESP32-S3** platforms.

## 📖 Blog Post  
A detailed explanation of the method and implementation is available here:  
🔗 **[ESP32-P4 Deep Learning Pipeline: Approximating SiLU for Efficient Quantization](https://boumedinebillal.blogspot.com/2025/02/esp32-p4-deep-learning-pipeline-update.html)**

## 📄 Reference Paper  
The sigmoid approximation used in this project is based on:  
🔗 **[Computationally Efficient Approximations of S-Shape Functions](https://www.researchgate.net/publication/311777918_Computationally_Efficient_Methods_of_Approximations_of_the_S-Shape_Functions_for_Image_Processing_and_Computer_Graphics_Tasks#full-text)** (Page 20)

## 🚀 Features  
- **Fast bitwise approximation** of SiLU using a quadratic sigmoid function
- **Optimized for MCUs** with efficient shift-based computation
- **ESP32-S3 SIMD implementation** (available in `QSiLUApprox_espS3_b1` subfile) for enhanced vectorized operations
- **Maintains high accuracy** in the key range **[-4, 4]** while preserving expected SiLU behavior

## 🔧 Installation & Usage  
Clone the repository and follow the usage instructions in the blog post:

```sh
git clone https://github.com/BoumedineBillal/silu_quantizer.git
cd silu_quantizer
```

### ESP32-S3 Implementation
The ESP32-S3 SIMD implementation can be found in the `QSiLUApprox_espS3_b1` directory. To use it:

1. Open the project in VSCode with the ESP-IDF extension
2. Build the project using the ESP-IDF build system
3. Flash the resulting binary to your ESP32-S3 device

This SIMD implementation can be easily integrated into any deep learning inference engine running on ESP32-S3 platforms to accelerate models that use the SiLU activation function.

## 🤝 Contributions  
Contributions are welcome! Feel free to open an issue or submit a pull request.

📩 **Questions?** Reach out via [GitHub Issues](https://github.com/BoumedineBillal/silu_quantizer/issues).