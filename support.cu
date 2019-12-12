/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void verify(float *A, float *B, float *X, float *parallel_krp, float *parallel_mttkrp, unsigned m, unsigned n, unsigned c, unsigned d) {

  const float relativeTolerance = 1e-6;
  unsigned int count = 0;
  float *sequential_krp;
  sequential_krp = (float*) malloc( sizeof(float)*(m*n*c) );
  int krp_idx, a_idx, b_idx;
  for(int col = 0; col < c; col++){
    for(int i = 0; i < m; i++){
      for(int j = 0; j < n; j++){
        krp_idx = (i*n+j)%(m*n) + (col*m*n);
        a_idx = i%m + col;
        b_idx = j%n + col;
        sequential_krp[krp_idx] = A[a_idx] * B[b_idx];
        count++;
        float relativeError = (sequential_krp[krp_idx] - parallel_krp[krp_idx]);
        printf("%f/%f ", sequential_krp[krp_idx], parallel_krp[krp_idx]);
        if (relativeError > relativeTolerance
          || relativeError < -relativeTolerance) {
          printf("\nKRP TEST FAILED %u\n\n",count);
          exit(1);
         }
      }
    }
  }
  count = 0;
  int k = m*n;
  for(int row = 0; row < d; ++row) {
    for(int col = 0; col < c; ++col) {
      float sum = 0;
      for(unsigned int i = 0; i < k; ++i) {
        sum += X[row*k + i]*sequential_krp[i*n + col];
      }
      count++;
      float relativeError = (sum - parallel_mttkrp[row*n + col])/sum;
      printf("%f/%f ", sum, parallel_mttkrp[row*n + col]);
      if (relativeError > relativeTolerance
        || relativeError < -relativeTolerance) {
        printf("\n MTTKRP TEST FAILED %u\n\n",count);
        exit(1);
      }
    }
  }
  printf("MTTKRP TEST PASSED %u\n\n", count);

}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

