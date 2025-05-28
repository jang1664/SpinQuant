#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>

typedef float _float16_t;

void hadamard(
    _float16_t *X,
    _float16_t *Y,
    const int shape[4],
    int ndims);

int main(void)
{
  int n1 = 2;
  int n2 = 2;
  int n3 = 4 * 172;

  _float16_t *ref = (_float16_t *)malloc(sizeof(_float16_t) * n1 * n2 * n3);
  FILE *fp = fopen("hadamard_transform.bin", "rb");
  if (fp == NULL)
  {
    fprintf(stderr, "Error opening file\n");
    return 1;
  }
  size_t read_count = fread(ref, sizeof(_float16_t), n1 * n2 * n3, fp);
  if (read_count != n1 * n2 * n3)
  {
    fprintf(stderr, "Error reading file: expected %d elements, got %zu\n", n1 * n2 * n3, read_count);
    fclose(fp);
    free(ref);
    return 1;
  }
  fclose(fp);

  float *input = (float *)malloc(sizeof(float) * n1 * n2 * n3);
  for (int i = 0; i < n1 * n2 * n3; ++i)
  {
    input[i] = (_float16_t)i / (n1 * n2 * n3);
  }

  _float16_t *output = (_float16_t *)malloc(sizeof(_float16_t) * n1 * n2 * n3);
  int shape[4] = {n1, n2, n3, 1}; // Assuming 4D tensor with last dim size 1
  int ndims = 3;                  // Number of dimensions in the tensor

  printf("start hadamard transform\n");
  hadamard(input, output, shape, ndims);

  // Print output for verification
  _float16_t ref_val;
  _float16_t evl_val;
  int error_count = 0;
  for (int i = 0; i < n1; ++i)
  {
    for (int j = 0; j < n2; ++j)
    {
      for (int k = 0; k < n3; ++k)
      {
        ref_val = ref[i * n2 * n3 + j * n3 + k];
        evl_val = output[i * n2 * n3 + j * n3 + k];
        if (fabs((ref_val - evl_val)) > 1e-5)
        {
          printf("Mismatch at (%d, %d, %d): expected %e, got %e\n",
                 i, j, k, (float)ref_val, (float)evl_val);
          error_count++;
        }
      }
    }
  }

  if(error_count == 0)
  {
    printf("Hadamard transform passed all tests.\n");
  }
  else
  {
    printf("Hadamard transform failed with %d errors.\n", error_count);
  }

  free(input);
  free(output);
}