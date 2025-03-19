#ifndef CONFIG_H
#define CONFIG_H

#include <stdint.h>
#include "esp_heap_caps.h"

/**
 * @brief Uncomment to enable debug mode (or define via compiler flag -DCUSTOM_DEBUG)
 */
#define CUSTOM_DEBUG 

/**
 * @brief Defines 16-byte alignment for structures and arrays.
 */
#define ALIGNED_16 __attribute__((aligned(16)))

#ifdef CUSTOM_DEBUG
    #define DEBUG_CHECK_ALIGNMENT(ptr, size) check_alignment(ptr, size)
#else
    #define DEBUG_CHECK_ALIGNMENT(ptr, size) ((void)0)  // No-op in non-debug mode
#endif

/**
 * @brief Declares a register variable that maps to an assembly register.
 * @param reg The register name (e.g., q0, q1, etc.)
 */
#define DECLARE_REGISTER(reg) register int reg __asm__(#reg)

/**
 * @brief Loads a 128-bit vector from memory into a register (no increment).
 *
 * @param qx Output vector register (q0 - q7)
 * @param vec_adr The memory address to load from (must be a pointer or array)
 *
 * @note This macro modifies qx (output), but vec_adr remains unchanged.
 */
#define LOAD_VECTOR_NO_INC(qx, vec_adr) \
    do { \
        _Static_assert( \
            __builtin_types_compatible_p(typeof(vec_adr), typeof((int8_t*)0)) || \
            __builtin_types_compatible_p(typeof(&vec_adr[0]), typeof((int8_t*)0)) || \
            __builtin_types_compatible_p(typeof(vec_adr), typeof((int16_t*)0)) || \
            __builtin_types_compatible_p(typeof(&vec_adr[0]), typeof((int16_t*)0)) || \
            __builtin_types_compatible_p(typeof(vec_adr), typeof(int16_t[8])), /* <-- Added for arrays */ \
            "vec_adr must be a pointer or an array of int8_t or int16_t" \
        ); \
        asm volatile ( \
            "EE.VLD.128.IP " #qx ", %0, 0\n"  /* Load vector without increment */ \
            : /* No output */ \
            : "r"(vec_adr)  /* Input: memory address */ \
            : "memory" /* Clobber memory to prevent unwanted optimizations */ \
        ); \
    } while (0)

/**
 * @brief Loads a 128-bit vector from memory into a register with increment.
 *
 * @param qx Output vector register (q0 - q7)
 * @param vec_adr The memory address to load from (must be a pointer or array)
 * @param inc Memory increment (must be in [-2048, 2032] and a multiple of 16)
 *
 * @note This macro modifies qx (output) and updates vec_adr.
 */
#define LOAD_VECTOR(qx, vec_adr, inc) \
    do { \
        _Static_assert( \
            __builtin_types_compatible_p(typeof(vec_adr), typeof((int8_t*)0)) || \
            __builtin_types_compatible_p(typeof(&vec_adr[0]), typeof((int8_t*)0)) || \
            __builtin_types_compatible_p(typeof(vec_adr), typeof((int16_t*)0)) || \
            __builtin_types_compatible_p(typeof(&vec_adr[0]), typeof((int16_t*)0)), \
            "vec_adr must be a pointer or an array of int8_t or int16_t" \
        ); \
        _Static_assert((inc) >= -2048 && (inc) <= 2032 && ((inc) % 16 == 0), \
                      "Increment must be in [-2048, 2032] and a multiple of 16"); \
        asm volatile ( \
            "EE.VLD.128.IP " #qx ", %0, %1\n"  /* Load vector with increment */ \
            : "+r"(vec_adr) /* Output: updates vec_adr */ \
            : "i"(inc)  /* Input: immediate increment */ \
            : "memory" /* Clobber memory to prevent unwanted optimizations */ \
        ); \
    } while (0)

/**
 * @brief Stores a 128-bit vector register into memory (no increment).
 *
 * @param qx Input vector register (q0 - q7)
 * @param vec_adr The memory address to store to (must be a pointer or array)
 *
 * @note This macro does not modify qx or vec_adr.
 */
#define STORE_VECTOR_NO_INC(qx, vec_adr) \
    do { \
        _Static_assert( \
            __builtin_types_compatible_p(typeof(vec_adr), typeof((int8_t*)0)) || \
            __builtin_types_compatible_p(typeof(&vec_adr[0]), typeof((int8_t*)0)) || \
            __builtin_types_compatible_p(typeof(vec_adr), typeof((int16_t*)0)) || \
            __builtin_types_compatible_p(typeof(&vec_adr[0]), typeof((int16_t*)0)), \
            "vec_adr must be a pointer or an array of int8_t" \
        ); \
        asm volatile ( \
            "EE.VST.128.IP " #qx ", %0, 0\n"  /* Store vector without increment */ \
            : /* No output */ \
            : "r"(vec_adr) /* Input: memory address */ \
            : "memory" /* Clobber memory to prevent unwanted optimizations */ \
        ); \
    } while (0)

/**
 * @brief Stores a 128-bit vector register into memory with increment.
 *
 * @param qx Input vector register (q0 - q7)
 * @param vec_adr The memory address to store to (must be a pointer or array)
 * @param inc Memory increment (must be in [-2048, 2032] and a multiple of 16)
 *
 * @note This macro updates vec_adr.
 */
#define STORE_VECTOR(qx, vec_adr, inc) \
    do { \
        _Static_assert( \
            __builtin_types_compatible_p(typeof(vec_adr), typeof((int8_t*)0)) || \
            __builtin_types_compatible_p(typeof(&vec_adr[0]), typeof((int8_t*)0)) || \
            __builtin_types_compatible_p(typeof(vec_adr), typeof((int16_t*)0)) || \
            __builtin_types_compatible_p(typeof(&vec_adr[0]), typeof((int16_t*)0)), \
            "vec_adr must be a pointer or an array of int8_t" \
        ); \
        _Static_assert((inc) >= -2048 && (inc) <= 2032 && ((inc) % 16 == 0), \
                      "Increment must be in [-2048, 2032] and a multiple of 16"); \
        asm volatile ( \
            "EE.VST.128.IP " #qx ", %0, %1\n"  /* Store vector with increment */ \
            : "+r"(vec_adr) /* Output: updates vec_adr */ \
            : "i"(inc)  /* Input: immediate increment */ \
            : "memory" /* Clobber memory to prevent unwanted optimizations */ \
        ); \
    } while (0)


/**
 * @brief Computes the absolute value of 16 int8 elements in a vector.
 * 
 * @param qa Output vector register (q0 - q7)
 * @param qx Input vector register (q0 - q7)
 * 
 * @note This macro modifies qa (output) but leaves qx unchanged (input-only)
 */
#define ABS_VECTOR_16_INT8(qa, qx) \
    do { \
        asm volatile ( \
            "EE.ZERO.Q " #qa "\n\t"   /* Clear qa to zero (output) */ \
            "EE.VSUBS.S8 " #qa ", " #qa ", " #qx "\n\t"  /* qa = 0 - qx (temp) */ \
            "EE.VMAX.S8 " #qa ", " #qx ", " #qa "\n\t"   /* qa = max(qx, -qx) (output) */ \
            : /* output */ \
            : /* input */ \
            : /* Memory barrier to prevent optimizations */ \
        ); \
    } while (0)


/**
 * @brief Clears a 128-bit vector register (sets all elements to zero).
 *
 * @param qa Output vector register (q0 - q7)
 *
 * @note This macro modifies qa (output), setting it to zero.
 */
#define ZERO_VECTOR(qa) \
    do { \
        asm volatile ( \
            "EE.ZERO.Q " #qa "\n\t"  /* Clear qa register to zero */ \
            : /* No output */ \
            : /* No input */ \
            : /* No clobbering needed, but ensures no unwanted optimizations */ \
        ); \
    } while (0)


/**
 * @brief Sets the SAR (Shift Amount Register) with a 5-bit value.
 *
 * @param value The shift amount (must be in the range [0, 31]).
 *
 * @note Uses compile-time assertions to ensure the value is within range.
 */
#define SET_SAR(value) \
    do { \
        _Static_assert((value) >= 0 && (value) <= 31, "SAR value must be in the range [0, 31]"); \
        asm volatile ( \
            "wsr %0, sar\n"  /* Write value to SAR register */ \
            : /* No output */ \
            : "r"(value) /* Input: value to set */ \
            : "memory" /* Clobber memory to prevent unwanted optimizations */ \
        ); \
    } while (0)


/**
 * @brief Sets the SAR (Shift Amount Register) with a runtime 5-bit value.
 *
 * @param var The shift amount (must be in the range [0, 31]).
 *
 * @note Includes a runtime check to ensure the value is within range.
 */
#define SET_SAR_VAR(var) \
    do { \
        if ((var) < 0 || (var) > 31) { \
            __builtin_trap(); /* Halt execution if out of range */ \
        } \
        asm volatile ( \
            "wsr %0, sar\n"  /* Write var to SAR register */ \
            : /* No output */ \
            : "r"(var) /* Input: runtime value */ \
            : "memory" /* Prevent unwanted optimizations */ \
        ); \
    } while (0)


static const int16_t scale_factor_8_int16[8] ALIGNED_16 = {1, 1, 1, 1, 1, 1, 1, 1};


/**
 * @brief Expand an int8_t vector to int16_t with proper sign extension.
 *
 * This macro takes an int8_t vector (`q_src`) and expands it into two int16_t vectors
 * (`q_dst_high` and `q_dst_low`) using interleaving and arithmetic right shift.
 *
 * **Why is an arithmetic right shift needed?**
 * - `VZIP.8` interleaves elements with zero-padding, effectively shifting values left by 8 bits.
 * - Since the ESP32-P4 lacks an explicit arithmetic right shift instruction for vectors,
 *   we use a multiplication trick to restore sign extension correctly.
 *
 * **Instruction Breakdown:**
 * 1. EE.VZIP.8 q_dst_high, q_src
 *    - Expands q_src into two int16_t vectors:
 *
 *      Before:
 *      q_src  = [ x0, x1, x2, x3, ..., x14, x15 ] (int8_t)
 *
 *      After:
 *      q_dst_high = [ 0, x0, 0, x1, ..., 0, x7 ] (int16_t)
 *      q_dst_low  = [ 0, x8, 0, x9, ..., 0, x15 ] (int16_t)
 *      '0, x0' in int16 is 2**8 * x0 sow must do (2**8 * x0)>>8 
 *      but '>>' must be arith to preserve signe of the negative values
 *      no instraction to do this sow i use the mul inst that have this option
 * 
 * 2. EE.VMUL.S16 q_dst_high, q_dst_high, q_scale // scale is vector 8 of 1 int16
 *    EE.VMUL.S16 q_dst_low, q_dst_low, q_scale
 *    - Simulates an arithmetic right shift by 8 bits, preserving sign extension.
 *
 * @note
 * - `SET_SAR(8);` must be called before using this macro.
 * - `q_scale` should be a preloaded vector containing all `1` values (`scale_factor_8_int16`).
 * - `ZERO_VECTOR(aux);` must be called before to ensure clean zero-padding.
 *
 * @param q_dst_high Output high vector (expanded from q_src)
 * @param q_src_dst_low      Input int8_t vector and Output low vector (expanded from q_src)
 * @param q_scale    Scaling factor vector (should be preloaded with ones)
 */
#define EXPAND_INT8_TO_INT16(q_dst_high, q_src, q_scale) \
    asm volatile ( "EE.VZIP.8 " #q_dst_high ", " #q_src : : );  \
    asm volatile ( "EE.VMUL.S16 " #q_dst_high ", " #q_dst_high ", " #q_scale : : ); \
    asm volatile ( "EE.VMUL.S16 " #q_src ", " #q_src ", " #q_scale : : );



/**
 * @brief Compress two int16_t vectors back to int8_t using EE.VUNZIP.8.
 *
 * This macro reverses the `EXPAND_INT8_TO_INT16` operation by using `EE.VUNZIP.8`
 * to merge `q_src_high` and `q_src_low` back into an `int8_t` vector (q_src_low).
 *
 * **Instruction Breakdown:**
 * 1. EE.VUNZIP.8 q_src_low, q_src_high
 *    - Merges the values back into a packed format.
 *
 * @note
 * - This is the **reverse operation** of `EXPAND_INT8_TO_INT16`.
 * - No saturation handling is done (unlike `VPACK.S16`).
 *
 * @param q_src_low  Input low int16_t vector (overwritten) / Output int8_t.
 * @param q_src_high Input high int16_t vector.
 */
#define COMPRESS_INT16_TO_INT8(q_src_low, q_src_high) \
    asm volatile ( "EE.VUNZIP.8 " #q_src_low ", " #q_src_high : : ); \

    

/**
 * @brief Compares 8-bit, 16-bit, or 32-bit vector elements using EE.VCMP.[OP].S[TYPE].
 *
 * @param op   Comparison operation: EQ (==), LT (<), GT (>)
 * @param type Element size: 8, 16, or 32 bits
 * @param qa   (output) Register to store comparison results (0xFF if condition is met, 0 otherwise)
 * @param qx   (input) First vector operand
 * @param qy   (input) Second vector operand
 *
 * @note The result is stored in `qa` with each element set to either `0xFF` (true) or `0` (false).
 * @note The `type` parameter must be `8`, `16`, or `32`. This macro does **not** check for invalid types at compile-time.
 * @note This instruction modifies `qa` but does **not** modify `qx` or `qy`.
 */
#define VECTOR_COMPARE(op, type, qa, qx, qy)   \
    do {                                       \
        asm volatile (                         \
            "EE.VCMP." #op ".S" #type " "      \
            #qa ", " #qx ", " #qy "\n\t"       \
            : /* output: qa modified */        \
            : /* input: qx, qy */              \
        );                                     \
    } while (0)


/**
 * @brief Performs element-wise saturated addition on signed integers.
 *
 * @param qa    Destination vector register.
 * @param qx    First source vector register.
 * @param qy    Second source vector register.
 * @param type  Data type: 8, 16, or 32 (for int8_t, int16_t, int32_t).
 */
#define VECTOR_ADD_SATURATED(qa, qx, qy, type)                             \
    do {                                                                   \
        asm volatile ("EE.VADDS.S" #type " " #qa ", " #qx ", " #qy "\n\t"  \
                      : /* output */                                       \
                      : /* input */                                        \
                      : /* clobber */);                                    \
    } while (0)

/**
 * @brief Performs element-wise saturated substaction on signed integers.
 *
 * @param qa    Destination vector register.
 * @param qx    First source vector register.
 * @param qy    Second source vector register.
 * @param type  Data type: 8, 16, or 32 (for int8_t, int16_t, int32_t).
 */
#define VECTOR_SUB_SATURATED(qa, qx, qy, type)                             \
    do {                                                                   \
        asm volatile ("EE.VSUBS.S" #type " " #qa ", " #qx ", " #qy "\n\t"  \
                      : /* output */                                       \
                      : /* input */                                        \
                      : /* clobber */);                                    \
    } while (0)


/**
 * @brief Performs element-wise signed vector multiplication with right shift by SAR.
 *
 * @note **IMPORTANT:** Before using this macro, ensure `SAR` is set correctly:  
 *       - Call `SET_SAR(8);` before using with `type = 8`.  
 *       - Call `SET_SAR(16);` before using with `type = 16`.
 *
 * @param qz    Destination vector register.
 * @param qx    First source vector register (multiplicand).
 * @param qy    Second source vector register (multiplier).
 * @param type  Data type: 8 or 16 (for int8_t or int16_t).
 */
#define VECTOR_MUL(qz, qx, qy, type)                                  \
    do {                                                              \
        asm volatile ("EE.VMUL.S" #type " " #qz ", " #qx ", " #qy "\n\t" \
                      : /* output */                                  \
                      : /* input */                                   \
                      : /* clobber */);                               \
    } while (0)


/**
 * @brief Moves the value from source QR register to destination QR register.
 *
 * @param qu    Destination QR register.
 * @param qs    Source QR register.
 */
#define MOVE_QR(qu, qs)                     \
    do {                                     \
        asm volatile ("MV.QR " #qu ", " #qs "\n\t" \
                      : /* output */         \
                      : /* input */          \
                      : /* clobber */);      \
    } while (0)



    /**
 * @brief Performs a bitwise logical operation (AND, OR, XOR) on vector registers.
 *
 * @param op   The operation type: AND, OR, XOR.
 * @param qa   Destination register.
 * @param qx   First source register.
 * @param qy   Second source register.
 *
 * @note Example usage:
 *       VECTOR_LOGIC(AND, q0, q1, q2); // Performs q0 = q1 & q2
 *       VECTOR_LOGIC(OR, q3, q4, q5);  // Performs q3 = q4 | q5
 *       VECTOR_LOGIC(XOR, q6, q7, q8); // Performs q6 = q7 ^ q8
 */
#define VECTOR_LOGIC(op, qa, qx, qy) \
do { \
    asm volatile ( \
        "EE." #op "Q " #qa ", " #qx ", " #qy "\n\t" \
        : /* No output */ \
        : /* No input */ \
        : "memory" /* Prevent compiler optimizations */ \
    ); \
} while (0)




#endif // CONFIG_H


/*
 * @file config.h
 * 
 * @section MACRO_GUIDELINES Macro Design Pattern for Future Extensions
 * 
 * 1️⃣ **General Structure**
 *    - Each macro should be fully **self-contained** with `_Static_assert` checks when applicable.
 *    - Use **do-while(0)** to ensure correct scope behavior.
 * 
 * 2️⃣ **Register Usage Annotations**
 *    - Always **explicitly document which registers are modified**.
 *    - Use the following **keywords**:
 *      - **input** → Register is only read.
 *      - **output** → Register is modified.
 *      - **temp** → Temporary register, modified but not returned.
 *      - **input_temp** → Input register that is modified.
 * 
 * 3️⃣ **Memory & Clobbering**
 *    - Use `"memory"` in the clobber list **to prevent unwanted compiler optimizations**.
 *    - If additional registers are modified inside inline assembly, **explicitly list them**.
 * 
 * 4️⃣ **Type Checking**
 *    - All macro parameters, **except vector registers**, must have at least one **static type check**.
 *    - Example: `_Static_assert(__builtin_types_compatible_p(typeof(vec_adr), typeof((int8_t*)0)), "vec_adr must be a pointer to a valid memory type");`
 *    - This applies to **both `LOAD_VECTOR` and `STORE_VECTOR`**.
 * 
 * 5️⃣ **Naming Conventions**
 *    - Macros that operate on **specific vector element types** must include the **data type and element count**.
 *    - Example:
 *      - **`ABS_VECTOR_16_INT8`** → Operates on **16 int8 elements**.
 *      - **`LOAD_VECTOR`** → **Generic**, as it loads a 128-bit register regardless of data type.
 *      - **`STORE_VECTOR`** → **Generic**, as it stores a 128-bit register regardless of data type.
 *    - This avoids confusion when different data types are introduced in future macros.
 * 
 * 6️⃣ **Future Expansion Example**
 *    - If a new vector operation is needed, follow this format:
 *      ```c
 *      #define VECTOR_ADD_16_INT8(qd, qx, qy) \
 *          do { \
 *              asm volatile ( \
 *                  "EE.VADD.S8 " #qd ", " #qx ", " #qy "\n\t" \
 *                  : * output * \
 *                  : * input * \
 *                  : "memory" * Clobber to prevent optimization * \
 *              ); \
 *          } while (0)
 *      ```
 *    - **qd (output), qx (input), qy (input)** are properly documented.
 * 
 * 7️⃣ **Error Checking**
 *    - Use `_Static_assert` **for compile-time validation** (e.g., memory increments, alignment constraints).
 *    - Always validate that parameters **fall within ISA constraints**.
 */


/*
Constraint	Effect	                        When to Use

"r"(var)	Read-only (unchanged)	        When the assembly only reads the variable 

  --> help the compiler to remove instructions like cloning 
  exemple:
     int a = 4;
     int b = a;
     use b in inline asm with "r" 
     after this b is not changed in all the code, and we use it or not using it in the remaining code 
     ::: the compiler will remove b and replace it with a in the code (no b)

"+r"(var)	Read & Write	                When the assembly modifies the variable

  --> help the compiler to track the updated value 
  exemple:
     int a = 4;
     use a in inline asm with "+r"
     the inline asm increments a by 1
     after this, the compiler knows a is now 5 and uses the new value correctly in the rest of the code

"=r"(var)	Write-only (ignores old value)	When the assembly completely overwrites the variable

  --> help the compiler to ignore the previous value 
  exemple:
     int a = 4;
     use a in inline asm with "=r"
     the inline asm sets a to 10
     after this, the compiler forgets that a was 4 before and treats it as 10

"i"(value)	Immediate (constant)	        When the value must be a constant at compile-time

  --> helps the compiler optimize by directly embedding the value 
  exemple:
     use the number 5 in inline asm with "i"
     the compiler ensures 5 is directly used in the instruction instead of loading it from memory
*/


/*
### Difference Between &vec_adr[0] and vec_adr in Inline Assembly

#### Why vec_adr (array name) cannot use +r
- When vec_adr is declared as an array:
  int8_t vec_adr[16];
  
  - vec_adr is not a pointer, but an array name.
  - It decays into a pointer (int8_t *) in most expressions, but it is not modifiable.
  - The compiler cannot store vec_adr in a register and modify it.

- +r (read & write) means the assembly instruction may change the variable.
  - Since vec_adr is not a modifiable lvalue, it cannot be used with +r.

  Invalid Example:
  asm volatile (
      "EE.VLD.128.IP q0, %0, #16 \n"  
      : "+r"(vec_adr)  // Error! vec_adr (array) is not modifiable
  );

#### Why &vec_adr[0] can use +r
- &vec_adr[0] explicitly takes the address of the first element.
- It evaluates to a pointer (int8_t *), which is modifiable.
- A pointer can be stored in a register and updated.

  Valid Example:
  int8_t vec_adr[16]; 
  int8_t *ptr = &vec_adr[0];  // Get a pointer to the array

  asm volatile (
      "EE.VLD.128.IP q0, %0, #16 \n"
      : "+r"(ptr)  // Works! ptr is a modifiable pointer
  );

#### Conclusion
- If vec_adr is an array, do not use +r on it directly.
- Instead, use &vec_adr[0] or store it in a pointer variable (int8_t *ptr).
- Solution: Always pass a modifiable pointer (+r) instead of an array name.
*/
