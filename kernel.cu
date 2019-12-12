/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_SIZE 16

__global__ void matmul_kernel(int m, int n, int k, const float *X, const float *KRP, float* MTTKRP) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    float p_value = 0;

    int width, height;
    __shared__ float tile_X[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_KRP[TILE_SIZE][TILE_SIZE];
    int steps = (k+TILE_SIZE-1) / TILE_SIZE;
    for(int i=0; i<steps; i++){
      width = i*TILE_SIZE+threadIdx.x;
      if(row < m && width < k)
          tile_X[threadIdx.y][threadIdx.x]=X[row*k+width];
      else
          tile_X[threadIdx.y][threadIdx.x]=0.0;

      height = i*TILE_SIZE+threadIdx.y;
      if(col < n && height < k)
          tile_KRP[threadIdx.y][threadIdx.x]=KRP[height*n+col];
      else
          tile_KRP[threadIdx.y][threadIdx.x]=0.0;

      __syncthreads();

      for(int i=0; i<TILE_SIZE; i++)
         p_value += tile_X[threadIdx.y][i]*tile_KRP[i][threadIdx.x];
      __syncthreads();
    }
    if(row<m && col<n)
        MTTKRP[row*n+col] = p_value;
}

void matmul(int d, int c, int k, const float *X, const float *KRP, float *MTTKRP)
{

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;

    //INSERT CODE HERE
    dim3 dim_block(BLOCK_SIZE,BLOCK_SIZE,1);
    dim3 dim_grid;
    dim_grid.x = (c+dim_block.x-1)/dim_block.x;
    dim_grid.y = (d+dim_block.y-1)/dim_block.y;

    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE
    matmul_kernel<<<dim_grid,dim_block>>>(m,n,k,X,KRP,MTTKRP);
}
__global__ void krp_kernel(int m, int n, int c, const float *A, const float *B, float *KRP)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   KRP[idx] = A[] * B[];
}
void krp(int m, int n, int c, const float *A, const float *B, float *KRP)
{
    const unsigned int BLOCK_SIZE = 1024;
    dim3 dim_block(BLOCK_SIZE, 1,1);
    dim3 dim_grid(((m*n*c)-1)/BLOCK_SIZE+1,1,1);

    krp_kernel<<<dim_grid,dim_block>>>(m,n,c,A,B,KRP);
}

void mttkrp(int m, int n, int c, int d, const float *A, const float *B, const float *X, float *KRP, float *MTTKRP)
{
    krp(m,n,c,A,B,KRP);
    matmul(d,c,m*n,X,KRP,MTTKRP);
}
