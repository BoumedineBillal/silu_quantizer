#include <stdio.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_mac.h"
#include <math.h>
#include "tools.h"


/*
========================================
       Testing QSiLUApprox Module 
                python
========================================

--- Original Input ---
tensor([-10.0000,  -4.0000,  -3.5000,  -3.0000,  -2.5000,  -2.0000,  -1.5000,  
        -1.2500,  -0.5000,   0.0000,   0.5000,   1.0000,   2.0000,  
         3.0000,   4.0000,  60.0000])

--- Clamped Input (Bounds Applied) ---
tensor([-4.0000,  -4.0000,  -3.5000,  -3.0000,  -2.5000,  -2.0000,  -1.5000,  
        -1.2500,  -0.5000,   0.0000,   0.5000,   1.0000,   2.0000,  
         3.0000,   4.0000,  60.0000])

========================================
      Pre-Activation Quantization
========================================
Quantized Input (x_q):    tensor([ -8.,  -8.,  -7.,  -6.,  -5.,  -4.,  -3.,  -2.,
                                    -1.,   0.,   1.,   2.,   4.,   6.,   8., 120.])
Scale (s1):               0.500000
Zero Point (z1):          0
Exponent (n1):            1

========================================
  SiLU Approximation (Quantized)
========================================
Quantized Activation:     tensor([-126., -126., -127., -127., -127., -127., -128., -128.,
                                   -127., -126., -125., -124., -119., -115., -110.,  114.])

========================================
      Post-Activation Quantization
========================================
Scale (s2):               0.250000
Zero Point (z2):          -126
Exponent (n2):            2

--- Final Dequantized Output ---
tensor([ 0.0000,  0.0000, -0.2500, -0.2500, -0.2500, -0.2500, -0.5000, -0.5000,
        -0.2500,  0.0000,  0.2500,  0.5000,  1.7500,  2.7500,  4.0000, 60.0000])
*/



/**
 * @brief Parameters for quantized SiLU (Swish) activation function
 * 
 * This struct holds all precomputed constants needed for the QSiLU approximation
 * to avoid redundant calculations during inference.
 */
typedef struct {
    int c1;             // Shift for scaling
    int16_t c2;         // Constant for polynomial approximation
    int16_t c3;         // Constant for polynomial approximation
    int16_t c4;         // Shift constant for linear approximation region
    int16_t c5;         // Threshold for linear approximation region
    int16_t z2;         // Zero point for output quantization
    
    // Vectorized constants for SIMD operations
    int16_t c2_vector[8] ALIGNED_16;
    int16_t c3_vector[8] ALIGNED_16;
    int16_t c4_vector[8] ALIGNED_16;
    int16_t c5_vector[8] ALIGNED_16;
    int16_t c5_vector_m1[8] ALIGNED_16;
    int16_t z2_vector[8] ALIGNED_16;
} QSiLUParams;

/**
 * @brief Initialize QSiLU parameters based on quantization parameters
 * 
 * @param params Pointer to QSiLUParams struct to be initialized
 * @param n1 Input quantization exponent
 * @param n2 Output quantization exponent
 * @param z2 Output quantization zero point
 */
void init_qsilu_params(QSiLUParams* params, int n1, int n2, int z2) {
    // Calculate all constants as per QSiLU approximation formulas
    params->c1 = (3 * n1 + 5) - n2;  // shift value
    params->c2 = pow(2, 2 + n1);     // constant
    params->c3 = pow(2, 2 * n1 + 5); // constant
    params->c4 = pow(2, n2 - n1);    // shift value for linear region
    params->c5 = 4 * pow(2, n1);     // threshold constant
    params->z2 = z2;                 // zero point
    
    // Initialize vectorized constants for SIMD processing
    for (int i = 0; i < 8; i++) {
        params->c2_vector[i] = params->c2;
        params->c3_vector[i] = params->c3;
        params->c4_vector[i] = params->c4;
        params->c5_vector[i] = params->c5;
        params->c5_vector_m1[i] = params->c5-1;
        params->z2_vector[i] = params->z2;
    }
}

/**
 * @brief Apply Quantized SiLU activation to input vector
 * 
 * Implements piecewise polynomial approximation of SiLU function for quantized inputs:
 * - For x ≤ 0: c1 * (x + c2)² * x + z2
 * - For 0 < x ≤ c5: c1 * (c3 - (x - c2)²) * x + z2
 * - For x > c5: c4 * x + z2
 * 
 * @param input Pointer to quantized input vector (int8_t)
 * @param params QSiLU parameters struct
 */
void apply_qsilu(int8_t* input, const QSiLUParams* params) {
    // Process first half of vector (expanded to 16-bit)
    LOAD_VECTOR_NO_INC(q3, input);
    
    SET_SAR(8);
    ZERO_VECTOR(q4);
    LOAD_VECTOR_NO_INC(q2, scale_factor_8_int16); // q5 as temp
    EXPAND_INT8_TO_INT16(q4, q3, q2); 

    // Case 1: x ≤ 0 => c1 * (x + c2)² * x + z2
    LOAD_VECTOR_NO_INC(q0, params->c2_vector);
    VECTOR_ADD_SATURATED(q5, q4, q0, 16);
    SET_SAR(0);
    VECTOR_MUL(q5, q5, q5, 16);
    SET_SAR_VAR(params->c1);
    VECTOR_MUL(q5, q5, q4, 16);
    LOAD_VECTOR_NO_INC(q1, params->z2_vector);
    VECTOR_ADD_SATURATED(q5, q5, q1, 16);

    // Case 2: 0 < x ≤ c5 => c1 * (c3 - (x - c2)²) * x + z2
    VECTOR_SUB_SATURATED(q6, q4, q0, 16);
    SET_SAR(0);
    VECTOR_MUL(q6, q6, q6, 16);
    LOAD_VECTOR_NO_INC(q0, params->c3_vector);
    VECTOR_SUB_SATURATED(q6, q0, q6, 16);
    SET_SAR_VAR(params->c1);
    VECTOR_MUL(q6, q6, q4, 16);
    VECTOR_ADD_SATURATED(q6, q6, q1, 16);

    // Apply condition x ≤ 0
    VECTOR_COMPARE(LT, 16, q2, q4, q2);
    VECTOR_LOGIC(AND, q0, q5, q2);

    // Apply condition x > 0
    ZERO_VECTOR(q2);
    VECTOR_COMPARE(GT, 16, q2, q4, q2);

    // Case 3: x > c5 => c4 * x + z2
    LOAD_VECTOR_NO_INC(q7, params->c4_vector);
    SET_SAR(0);
    VECTOR_MUL(q7, q4, q7, 16);
    VECTOR_ADD_SATURATED(q7, q7, q1, 16);

    // Apply conditions and combine results
    LOAD_VECTOR_NO_INC(q5, params->c5_vector);
    VECTOR_COMPARE(LT, 16, q1, q4, q5);
    VECTOR_LOGIC(AND, q1, q2, q1);
    VECTOR_LOGIC(AND, q6, q6, q1);
    
    LOAD_VECTOR_NO_INC(q5, params->c5_vector_m1);
    VECTOR_COMPARE(GT, 16, q1, q4, q5);
    VECTOR_LOGIC(AND, q1, q1, q7);
    
    VECTOR_LOGIC(OR, q0, q0, q6);
    VECTOR_LOGIC(OR, q4, q1, q0);

    // Process second half of vector (similar operations)
    LOAD_VECTOR_NO_INC(q2, scale_factor_8_int16);
    
    // Case 1: x ≤ 0 => c1 * (x + c2)² * x + z2
    LOAD_VECTOR_NO_INC(q0, params->c2_vector);
    VECTOR_ADD_SATURATED(q5, q3, q0, 16);
    SET_SAR(0);
    VECTOR_MUL(q5, q5, q5, 16);
    SET_SAR_VAR(params->c1);
    VECTOR_MUL(q5, q5, q3, 16);
    LOAD_VECTOR_NO_INC(q1, params->z2_vector);
    VECTOR_ADD_SATURATED(q5, q5, q1, 16);

    // Case 2: 0 < x ≤ c5 => c1 * (c3 - (x - c2)²) * x + z2
    VECTOR_SUB_SATURATED(q6, q3, q0, 16);
    SET_SAR(0);
    VECTOR_MUL(q6, q6, q6, 16);
    LOAD_VECTOR_NO_INC(q0, params->c3_vector);
    VECTOR_SUB_SATURATED(q6, q0, q6, 16);
    SET_SAR_VAR(params->c1);
    VECTOR_MUL(q6, q6, q3, 16);
    VECTOR_ADD_SATURATED(q6, q6, q1, 16);

    // Apply condition x ≤ 0
    VECTOR_COMPARE(LT, 16, q2, q3, q2);
    VECTOR_LOGIC(AND, q0, q5, q2);

    // Apply condition x > 0
    ZERO_VECTOR(q2);
    VECTOR_COMPARE(GT, 16, q2, q3, q2);

    // Case 3: x > c5 => c4 * x + z2
    LOAD_VECTOR_NO_INC(q7, params->c4_vector);
    SET_SAR(0);
    VECTOR_MUL(q7, q3, q7, 16);
    VECTOR_ADD_SATURATED(q7, q7, q1, 16);

    // Apply conditions and combine results
    LOAD_VECTOR_NO_INC(q5, params->c5_vector);
    VECTOR_COMPARE(LT, 16, q1, q3, q5);
    VECTOR_LOGIC(AND, q1, q2, q1);
    VECTOR_LOGIC(AND, q6, q6, q1);
    
    LOAD_VECTOR_NO_INC(q5, params->c5_vector_m1);
    VECTOR_COMPARE(GT, 16, q1, q3, q5);
    VECTOR_LOGIC(AND, q1, q1, q7);
    
    VECTOR_LOGIC(OR, q0, q0, q6);
    VECTOR_LOGIC(OR, q3, q1, q0);

    // Compress the result back to int8_t
    COMPRESS_INT16_TO_INT8(q4, q3);
}

/**
 * @brief Main application entry point
 */
void app_main(void)
{
    // Define the input vector
    int8_t q_input[16] ALIGNED_16 = { -8, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 4, 6, 8, 120 };
    
    // Set quantization parameters
    int n1 = 1;
    int n2 = 2;
    int z2 = -126;
    
    // Initialize QSiLU parameters
    QSiLUParams params;
    init_qsilu_params(&params, n1, n2, z2);

    PRINT_ARRAY("Input:\n", q_input, 16);
    
    // Debug: Print input vector after expansion to int16_t
    LOAD_VECTOR_NO_INC(q3, q_input);
    SET_SAR(8);
    ZERO_VECTOR(q4);
    LOAD_VECTOR_NO_INC(q2, scale_factor_8_int16);
    EXPAND_INT8_TO_INT16(q4, q3, q2);
    PRINT_VECTOR("input1 int16 :\n", q4, int16_t);
    PRINT_VECTOR("input2 int16 :\n", q3, int16_t);
    
    // Apply QSiLU and print debug information
    apply_qsilu(q_input, &params);
    
    // Print results
    PRINT_VECTOR("QSiluApprox :\n", q4, int8_t);
}