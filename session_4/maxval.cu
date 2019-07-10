// MIN and MAX function //
DAT device_MAX=0.0;

#define NB_THREADS     (BLOCK_X*BLOCK_Y)
#define blockId        (blockIdx.x  +  blockIdx.y *gridDim.x)
#define threadId       (threadIdx.x + threadIdx.y*blockDim.x)
#define isBlockMaster  (threadIdx.x==0 && threadIdx.y==0)
// maxval //
#define block_max_init()  DAT __thread_maxval=0.0;
#define __thread_max(A,nx_A,ny_A)  if (iy<ny_A && ix<nx_A){ __thread_maxval = max((__thread_maxval) , (A[ix + iy*nx_A])); } 

__shared__ volatile  DAT __block_maxval;
#define __block_max(A,nx_A,ny_A)  __thread_max(A,nx_A,ny_A);  if (isBlockMaster){ __block_maxval=0; }  __syncthreads(); \
                                  for (int i=0; i < (NB_THREADS); i++){ if (i==threadId){ __block_maxval = max(__block_maxval,__thread_maxval); }  __syncthreads(); }

__global__ void __device_max_d(DAT*A, const int nx_A,const int ny_A, DAT*__device_maxval){
  block_max_init();
  int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
  int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
  // find the maxval for each block
  __block_max(A,nx_A,ny_A);
  __device_maxval[blockId] = __block_maxval;
}

#define __DEVICE_max(A,nx_A,ny_A)  __device_max_d<<<grid, block>>>(A##_d, nx_A,ny_A, __device_maxval_d); \
                                   gather(__device_maxval,grid.x,grid.y); device_MAX=(DAT)0.0;           \
                                   for (int i=0; i < (grid.x*grid.y); i++){                              \
                                      device_MAX = max(device_MAX,__device_maxval_h[i]);                 \
                                   }                                                                     \
                                   A##_MAX = (device_MAX);

// MAX 
DAT P_MAX=0.0;
zeros(__device_maxval  ,grid.x,grid.y);


__MPI_max(P,nx,ny);