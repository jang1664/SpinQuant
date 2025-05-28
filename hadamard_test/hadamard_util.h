#ifndef HADAMARD_UTIL_H
#define HADAMARD_UTIL_H

#include <stdbool.h>
// #include <assert.h>
// #include <stdlib.h>
#include <stdio.h>
#include "hadamard_K.h"

// typedef _Float16 _float16_t;
typedef float _float16_t;
#define MAX_B 16
#define MAX_N 256
#define MAX_K 172
#define MAX_M MAX_N

/**
 * is_pow2(n):
 *   true if n>0 and a power of two.
 */
static int is_pow2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/**
 * get_hadK:
 *   Given total length n and a transpose flag, choose the largest
 *   “block size” K from your predefined list that divides n, assert
 *   that n/K is a power of two, then return that K and a pointer
 *   to the K×K Hadamard (transposed if requested).
 *
 * Params:
 *   n          – total transform length
 *   transpose  – if true, we return the transpose of the chosen hadK
 *   out_hadK   – *out* pointer to the chosen K×K matrix
 *   out_K      – *out* the chosen block size K
 *
 * Notes:
 *   - If K==1, *out_hadK will be NULL (you can just skip the K×K multiply).
 *   - The caller is responsible for free()’ing the transposed buffer
 *     when transpose==true.  When transpose==false, we hand back the
 *     raw pointer from get_hadXX(), which must NOT be freed here.
 */
void get_hadK(int n, int transpose, float **out_hadK, int *out_K) {
    float T[sizeof(float) * MAX_K * MAX_K];
    int K = 1;
    float *base = NULL;

    if (n % 172 == 0) {
        int m = n / 172;      //assert(is_pow2(m));
        K = 172;              get_had172(&base);
    }
    else if (n % 156 == 0) {
        int m = n / 156;      //assert(is_pow2(m));
        K = 156;              get_had156(&base);
    }
    else if (n % 140 == 0) {
        int m = n / 140;      //assert(is_pow2(m));
        K = 140;              get_had140(&base);
    }
    else if (n % 108 == 0) {
        int m = n / 108;      //assert(is_pow2(m));
        K = 108;              get_had108(&base);
    }
    else if (n % 60  == 0) {
        int m = n / 60;       //assert(is_pow2(m));
        K = 60;               get_had60(&base);
    }
    else if (n % 52  == 0) {
        int m = n / 52;       //assert(is_pow2(m));
        K = 52;               get_had52(&base);
    }
    else if (n % 44  == 0) {
        int m = n / 44;       //assert(is_pow2(m));
        K = 44;               get_had44(&base);
    }
    else if (n % 40  == 0) {
        int m = n / 40;       //assert(is_pow2(m));
        K = 40;               get_had40(&base);
    }
    else if (n % 36  == 0) {
        int m = n / 36;       //assert(is_pow2(m));
        K = 36;               get_had36(&base);
    }
    else if (n % 28  == 0) {
        int m = n / 28;       //assert(is_pow2(m));
        K = 28;               get_had28(&base);
    }
    else if (n % 20  == 0) {
        int m = n / 20;       //assert(is_pow2(m));
        K = 20;               get_had20(&base);
    }
    else if (n % 12  == 0) {
        int m = n / 12;       //assert(is_pow2(m));
        K = 12;               get_had12(&base);
    }
    else {
        // No block-size matched → full-Hadamard only valid if n itself is pow2
        //assert(is_pow2(n));
        K = 1;
        base = NULL;
    }

    // If user wants the transpose, and K>1, build a new buffer
    if (transpose && K > 1) {
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                // transpose: T[i,j] = base[j,i]
                T[i*K + j] = base[j*K + i];
            }
        }
        *out_hadK = T;
    } else {
        *out_hadK = base;
    }
    *out_K = K;
}

#endif