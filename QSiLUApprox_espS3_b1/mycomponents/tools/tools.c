#include "tools.h"

void print_array(const char *prefix, const void *array, size_t element_size, size_t length, PrintFunc print_func) {
    if (prefix) {
        printf("%s", prefix);
    }

    const uint8_t *byte_array = (const uint8_t *)array;
    printf("[ ");
    for (size_t i = 0; i < length; i++) {
        print_func(byte_array + (i * element_size));
        if (i < length - 1) printf(", ");
    }
    printf(" ]\n");
}

// Print functions for different types
void print_int8(const void *elem)   { printf("%d", *(const int8_t *)elem); }
void print_int16(const void *elem)   { printf("%d", *(const int16_t *)elem); }
void print_int(const void *elem)    { printf("%d", *(const int *)elem); }
void print_float(const void *elem)  { printf("%.2f", *(const float *)elem); }
void print_double(const void *elem) { printf("%.4lf", *(const double *)elem); }

#ifdef CUSTOM_DEBUG
void check_alignment(void *ptr, size_t alignment) {
    if (((uintptr_t)ptr % alignment) == 0) {
        printf("Vector is correctly aligned to %zu bytes at address %p\n", alignment, ptr);
    } else {
        printf("Vector is NOT correctly aligned! Address: %p\n", ptr);
    }
}
#endif
