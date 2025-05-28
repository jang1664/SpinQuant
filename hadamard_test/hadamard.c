// #include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#include "hadamard_util.h"

/**
 * matmul_hadU_decomposed:
 *   Applies an n-point Hadamard transform on the last dimension of an
 *   up-to-4D tensor X, using the two-stage decomposition:
 *     1) reshape to (..., K, M) and do an M-length Hadamard on each row
 *     2) multiply each column by the K×K hadK matrix
 *   Finally normalizes by sqrt(n).
 *
 * Params:
 *   X      – pointer to input floats, row-major, total size = prod(shape[0..ndims-1])
 *   Y      – pointer to output floats (must be allocated same size)
 *   shape  – int array of length ndims giving sizes of each dim (ndims ≤ 4)
 *   ndims  – number of dims (1 to 4)
 */

static float in_buf[sizeof(float) * MAX_M];
static float tmp_buf[sizeof(float) * MAX_M];
static float X[sizeof(float) * MAX_N * MAX_B]; 
static float Y[sizeof(float) * MAX_N * MAX_B];

void get_hadK(int n, int transpose, float **out_hadK, int *out_K);

void hadamard(
    _float16_t *X_,
    _float16_t *Y_,
    const int shape[4],
    int ndims)
{
  // --- 1) flatten leading dims into batch count, find n and M
  int K;
  int batch = 1;
  float *hadK = NULL;

  for (int d = 0; d < ndims - 1; ++d)
  {
    batch *= shape[d];
  }

  int n = shape[ndims - 1];

  for(int i = 0; i < n * batch; ++i) {
    X[i] = (float)X_[i]; // Convert _float16_t to float
  }

  get_hadK(n, 0, &hadK, &K);

  int M = n / K; // assume n%K==0 and M is a power-of-two
  int merged_batch = batch * K;
  int decomposed_n = M;

  // --- 2) allocate working buffers of length n
  float norm = sqrtf((float)decomposed_n) * sqrtf((float)n);

  // --- 3) for each slice in the batch, do the decomposed transform
  for (int b = 0; b < merged_batch; ++b)
  {
    memcpy(in_buf, X + b * decomposed_n, sizeof(float) * decomposed_n);

    // ------ Stage 1: Hadamard of each length-M row (I_K ⊗ H_M) ------
    int stage = M;
    while (stage > 1)
    {
      int half = stage >> 1;
      // process each block of `stage` elements across the full n-vector
      for (int base = 0; base < decomposed_n; base += stage)
      {
        for (int i = 0; i < half; ++i)
        {
          float a = in_buf[base + i];
          float c = in_buf[base + i + half];
          tmp_buf[base + i] = a + c;
          tmp_buf[base + i + half] = a - c;
        }
      }
      // copy tmp back into in_buf
      memcpy(in_buf, tmp_buf, sizeof(float) * decomposed_n);
      stage = half;
    }
    // normalize after the M-transform
    for (int i = 0; i < decomposed_n; ++i)
    {
      in_buf[i] /= norm;
      X[b * decomposed_n + i] = in_buf[i];
    }
  }

  // ------ Stage 2: Hadamard of each length-K column (H_K ⊗ I_M) ------
  // we have M columns, each of length K, with stride = M
  for (int b = 0; b < batch; ++b)
  {
    for (int row = 0; row < K; row++)
    {
      for (int col = 0; col < M; col++)
      {
        // compute the Hadamard product for each column
        if(K>1) {
          float sum = 0.0f;
          for (int k = 0; k < K; k++)
          {
            sum += hadK[row * K + k] * X[b * n + col + k * M];
          }
          Y[b * n + col + row * M] = sum;
        } else {
          Y[b * n + col + row * M] = X[b * n + col + row * M];
        }
      }
    }
  }

  for(int i = 0; i < n * batch; ++i) {
    Y_[i] = (_float16_t)Y[i]; // Convert float back to _float16_t
  }
}
