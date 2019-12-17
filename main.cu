/******************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"

int main (int argc, char *argv[])
{

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *A_h, *B_h, *X_h, *KRP_h, *MTTKRP_h;
    float *A_d, *B_d, *X_d, *KRP_d, *MTTKRP_d;
    size_t A_sz, B_sz, X_sz, KRP_sz, MTTKRP_sz;
    unsigned matArow, matAcol;
    unsigned matBrow, matBcol;
    unsigned matXrow, matXcol;
    dim3 dim_grid, dim_block;

    if (argc == 1) {
        matArow = matBrow = 10;
        matAcol = matBcol = 5;
        matXrow = 10;
        matXcol = matArow*matBrow;
    } else if (argc == 5) {
        matArow = atoi(argv[1]);
        matBrow = atoi(argv[2]);
        matAcol = matBcol = atoi(argv[3]);
        matXrow = atoi(argv[4]);
        matXcol = matArow*matBrow;
    } else {
        printf("\n    Invalid input parameters!"
      "\n    Usage: ./mttkrp                   # A: 1000x5, B:1000x5, X:1000x1000^2"
      "\n    Usage: ./mttkrp <m> <n> <c> <d>   # A: m x c, B: n x c, X: d x mn"
      "\n");
        exit(0);
    }



    A_sz = matArow*matAcol;
    B_sz = matBrow*matBcol;
    X_sz = matXrow*matXcol;
    KRP_sz = matArow*matBrow*matAcol;
    MTTKRP_sz = matXrow*matAcol;

    A_h = (float*) malloc( sizeof(float)*A_sz );
    // for (unsigned int i=0; i < A_sz; i++) { A_h[i] = (rand()%100)/100.00; }
    for (unsigned int i=0; i < A_sz; i++){ A_h[i] = i; }

    B_h = (float*) malloc( sizeof(float)*B_sz );
    // for (unsigned int i=0; i < B_sz; i++) { B_h[i] = (rand()%100)/100.00; }
    for (unsigned int i=0; i < B_sz; i++) { B_h[i] = i; }

    X_h = (float*) malloc( sizeof(float)*X_sz );
    // for (unsigned int i=0; i < X_sz; i++) { X_h[i] = (rand()%100)/100.00; }
    for (unsigned int i=0; i < X_sz; i++) { X_h[i] = 1.0; }

    KRP_h = (float*) malloc( sizeof(float)*KRP_sz );
    MTTKRP_h = (float*) malloc( sizeof(float)*MTTKRP_sz );

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    A: %u x %u\n    B: %u x %u\n    X: %u x %u\n KRP: %u x %u\n MTTKRP: %u x %u\n", matArow, matAcol,matBrow, matBcol, matXrow, matXcol, matArow*matBrow, matAcol, matXrow, matAcol);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    int a_size = matArow*matAcol * sizeof(float);
    int b_size = matBrow*matBcol * sizeof(float);
    int x_size = matXrow*matXcol * sizeof(float);
    int krp_size = matArow*matBrow*matAcol * sizeof(float);
    int mttkrp_size = matXrow*matAcol * sizeof(float);

    cudaMalloc((void **) &A_d, a_size);
    cudaMalloc((void **) &B_d, b_size);
    cudaMalloc((void **) &X_d, x_size);
    cudaMalloc((void **) &KRP_d, krp_size);
    cudaMalloc((void **) &MTTKRP_d, mttkrp_size);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    cudaMemcpy(A_d, A_h, a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, b_size, cudaMemcpyHostToDevice);
    cudaMemcpy(X_d, X_h, x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(KRP_d, KRP_h, krp_size, cudaMemcpyHostToDevice);
    cudaMemcpy(MTTKRP_d, MTTKRP_h, mttkrp_size, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel using standard sgemm interface ---------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    mttkrp(matArow, matBrow, matAcol, matXrow, A_d, B_d, X_d, KRP_d, MTTKRP_d);

    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) {printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(cuda_ret)); FATAL("Unable to launch kernel"); }
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    cudaMemcpy(KRP_h, KRP_d, krp_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(MTTKRP_h, MTTKRP_d, mttkrp_size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(A_h, B_h, X_h, KRP_h, MTTKRP_h, matArow, matBrow, matAcol, matXrow);


    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(X_h);
    free(KRP_h);
    free(MTTKRP_h);

    //INSERT CODE HERE




    return 0;

}
