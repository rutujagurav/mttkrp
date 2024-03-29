/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_SIZE 32
#define KRP_BLOCK_SIZE 8

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
    matmul_kernel<<<dim_grid,dim_block>>>(d,c,k,X,KRP,MTTKRP);
}
__global__ void krp_kernel(int m, int n, int c, const float *A, const float *B, float *KRP)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x; //0 to mnc
   if(idx < m*n*c){
     int col = int(idx/(m*n));
     int start_col = m*n*col;
     int pos_in_col = idx - (start_col);
     // int idx_a = int(pos_in_col/n)%n + (n*col);
     int idx_a = int(pos_in_col/n)+m*col;
     int idx_b = int(pos_in_col%n+n*col);
     // printf("\n bid=%d bdim=%d tid=%d idx=%d col=%d start_col=%d pos_in_col=%d idxA=%d idxB=%d\n",blockIdx.x, blockDim.x, threadIdx.x, idx, col, start_col, pos_in_col, idx_a, idx_b);
     KRP[idx] = A[idx_a] * B[idx_b];
   }

}

__global__ void krp_kernel_sharedmem(int m, int n, int c, const float *A, const float *B, float *KRP)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x; //0 to mnc

   __shared__ float tile_A[KRP_BLOCK_SIZE];
   __shared__ float tile_B[KRP_BLOCK_SIZE];

   int steps_colA = (m+KRP_BLOCK_SIZE-1) / KRP_BLOCK_SIZE;
   int steps_colB = (n+KRP_BLOCK_SIZE-1) / KRP_BLOCK_SIZE;
   int w;

   if(idx < m*n*c){

     for(int i=0; i<steps_colA; i++){
       w = i*KRP_BLOCK_SIZE+threadIdx.x;
       if(w < m*c){
         tile_A[threadIdx.x] = A[idx*m*c+w];
         printf("\n IF A: bid=%d bdim=%d tid=%d idx=%d w=%d step=%d, tile_A=%f\n"
         ,blockIdx.x, blockDim.x, threadIdx.x, idx, w, i,tile_A[threadIdx.x]);

       }

       else{
         tile_A[threadIdx.x] = 0.0;
         // printf("\n ELSE A: bid=%d bdim=%d tid=%d idx=%d col=%d start_col=%d pos_in_col=%d idxA=%d idxB=%d w=%d step=%d tile_A=%f\n"
         // ,blockIdx.x, blockDim.x, threadIdx.x, idx, col, start_col, pos_in_col, idx_a, idx_b, w, i,tile_A[threadIdx.x]);

       }
     }

    __syncthreads();

     for(int i=0; i<steps_colB; i++){
       w = i*KRP_BLOCK_SIZE+threadIdx.x;

       if(w < n*c){
         tile_B[threadIdx.x] = B[idx*n*c+w];
         printf("\n IF B: bid=%d bdim=%d tid=%d idx=%d w=%d step=%d, tile_B=%f\n"
         ,blockIdx.x, blockDim.x, threadIdx.x, idx, w, i,tile_B[threadIdx.x]);

       }

       else{
         tile_B[threadIdx.x] = 0.0;
         // printf("\n ELSE B: bid=%d bdim=%d tid=%d idx=%d col=%d start_col=%d pos_in_col=%d idxA=%d idxB=%d w=%d step=%d tile_B=%f\n"
         // ,blockIdx.x, blockDim.x, threadIdx.x, idx, col, start_col, pos_in_col, idx_a, idx_b, w, i,tile_B[threadIdx.x]);

       }

      }

      __syncthreads();
      int col = int(idx/(m*n));
      int idx_tileA = int(idx/n);
      int idx_tileB = int(idx%n + n*col);
      // if(blockIdx.x == 1)
        printf("\n bid=%d bdim=%d tid=%d idx=%d col=%d tile_A=%f tile_B=%f\n",blockIdx.x, blockDim.x, threadIdx.x, idx, col, tile_A[idx_tileA],tile_B[idx_tileB]);

      KRP[idx] = tile_A[idx_tileA] * tile_B[idx_tileB];
   }
}


void krp(int m, int n, int c, const float *A, const float *B, float *KRP)
{
    const unsigned int BLOCK_SIZE = KRP_BLOCK_SIZE;
    dim3 dim_block(BLOCK_SIZE, 1,1);
    dim3 dim_grid(((m*n*c)-1)/BLOCK_SIZE+1,1,1);

    krp_kernel_sharedmem<<<dim_grid,dim_block>>>(m,n,c,A,B,KRP);
    // krp_kernel<<<dim_grid,dim_block>>>(m,n,c,A,B,KRP);
}

void mttkrp(int m, int n, int c, int d, const float *A, const float *B, const float *X, float *KRP, float *MTTKRP)
{
    krp(m,n,c,A,B,KRP);
    matmul(d,c,m*n,X,KRP,MTTKRP);
}
