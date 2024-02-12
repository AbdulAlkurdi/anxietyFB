#include <stdio.h>

void print_matrix(double *v, size_t l, size_t n, double *ecg )
{



   for (size_t k = 0; k < l; k++)
      for (size_t i = k+1; i < n-k-1; i++)
        if ( ecg[i] > ecg[i-k-1] && ecg[i] > ecg[i+k+1] ) v[k*n+i] = 0;



   printf("l=%d, n=%d\n", l, n);
   printf("\n");
}

