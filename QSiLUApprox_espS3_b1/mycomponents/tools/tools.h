#ifndef TOOLS_H
#define TOOLS_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include "config.h"

// Function pointer type for printing elements
typedef void (*PrintFunc)(const void *);

// Function prototypes
void print_array(const char *prefix, const void *array, size_t element_size, size_t length, PrintFunc print_func);
void print_int8(const void *elem);
void print_int16(const void *elem);
void print_int(const void *elem);
void print_float(const void *elem);
void print_double(const void *elem);

#ifdef CUSTOM_DEBUG
void check_alignment(void *ptr, size_t alignment);
#endif

// Macro to detect array type and call the appropriate function
#define PRINT_ARRAY(prefix, array, length) \
    _Generic((array),              \
        int8_t *:    print_array(prefix, array, sizeof(int8_t), length, print_int8),  \
        int16_t *:    print_array(prefix, array, sizeof(int16_t), length, print_int16),  \
        int *:       print_array(prefix, array, sizeof(int), length, print_int),      \
        float *:     print_array(prefix, array, sizeof(float), length, print_float),  \
        double *:    print_array(prefix, array, sizeof(double), length, print_double) \
    )

/**
 * @brief Prints the values of a 128-bit vector.
 *
 * This macro dynamically allocates a temporary buffer (`vec_tmp`),
 * stores the vector values, prints them, and then frees the memory.
 *
 * **Important Notes:**
 * - The vector must be **128 bits (16 bytes)**.
 * - The user must specify the **data type** (int8_t, int16_t, int, float, double).
 * - `vec_tmp` must be declared as a pointer **before using this macro**.
 *
 * **Example Usage:**
 * ```c
 * int8_t *vec_tmp; // Declare outside
 * PRINT_VECTOR("Vector values:", q2, int8_t);
 * ```
 *
 * @param prefix  String prefix for the printed output
 * @param q_src   Vector register to print
 * @param type    Data type inside the vector (int8_t, int16_t, int, float, double)
 */
#define PRINT_VECTOR(prefix, q_src, type)          \
    do {                                           \
        type *vec_tmp = (type*)aligned_alloc(16, 16); \
        if (vec_tmp) {                             \
            STORE_VECTOR_NO_INC(q_src, vec_tmp);   \
            PRINT_ARRAY(prefix, vec_tmp, 16 / sizeof(type)); \
            free(vec_tmp);                         \
        } else {                                   \
            printf("Memory allocation failed for PRINT_VECTOR\n"); \
        }                                          \
    } while (0)

/**
 * @brief Sets a vector register with immediate values.
 *
 * @param qd     Destination vector register.
 * @param type   Data type: int8_t, int16_t, or int32_t.
 * @param ...    List of values to set in the register.
 *
 * @note **The number of elements must match the vector type size**:
 *       - `int8_t` → 16 elements
 *       - `int16_t` → 8 elements
 *       - `int32_t` → 4 elements
 * @note Uses a **temporary aligned array** instead of dynamic allocation.
 */
#define SET_VECTOR(qd, type, ...)                           \
    do {                                                   \
        static const type vec_tmp[] ALIGNED_16 = { __VA_ARGS__ }; \
        _Static_assert(                                     \
            sizeof(vec_tmp) == 16,                         \
            "Invalid number of elements for 128-bit vector!" \
        );                                                 \
        LOAD_VECTOR_NO_INC(qd, vec_tmp);                   \
    } while (0)

/**
 * @brief Prints a vector comparison result as "true"/"false".
 *
 * @param prefix  Prefix string for output.
 * @param q_src   Vector register containing comparison results.
 * @param type    Data type: 8, 16, or 32 (bits).
 *
 * @note **Comparison vectors contain 0xFF for true, 0x00 for false.**
 *       - `int8_t` → 16 elements
 *       - `int16_t` → 8 elements
 *       - `int32_t` → 4 elements
 */
#define PRINT_COMPARE_VECTOR(prefix, q_src, type)                 \
    do {                                                          \
        int8_t *vec_tmp = (int8_t*)aligned_alloc(16, 16);         \
        if (vec_tmp) {                                            \
            STORE_VECTOR_NO_INC(q_src, vec_tmp);                  \
            printf("%s [ ", prefix);                              \
            for (size_t i = 0; i < (16 / (type / 8)); i++) {      \
                printf("%s ", (vec_tmp[i * (type / 8)] == -1) ? "true" : "false"); \
            }                                                     \
            printf("]\n");                                        \
            free(vec_tmp);                                        \
        } else {                                                  \
            printf("Memory allocation failed for PRINT_COMPARE_VECTOR\n"); \
        }                                                         \
    } while (0)

/**
 * @brief Sets a vector register with a repeated value.
 *
 * @param qd    Destination vector register.
 * @param type  Data type: int8_t, int16_t, or int32_t.
 * @param val   The value to be repeated across the vector.
 */
#define SET_VECTOR_BROADCAST(qd, type, val)                          \
    do {                                                             \
        type vec_tmp[16 / sizeof(type)] ALIGNED_16;                  \
        for (size_t i = 0; i < (16 / sizeof(type)); i++) {           \
            vec_tmp[i] = (type)(val);                                \
        }                                                            \
        LOAD_VECTOR_NO_INC(qd, vec_tmp);                             \
    } while (0)



#endif // TOOLS_H
