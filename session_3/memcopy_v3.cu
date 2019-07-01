// memcopy
// compile: nvcc -arch=sm_70 -O3 memcopy_v3.cu
// run: ./a.out
#include <stdio.h>
#include "sys/time.h"

#define GPU_ID 3

#define USE_SINGLE_PRECISION    /* Comment this line using "!" if you want to use double precision.  */
#ifdef USE_SINGLE_PRECISION
#define DAT     float
#define PRECIS  4
#else
#define DAT     double
#define PRECIS  8
#endif
#define zeros(A,nx,ny,nz)  DAT *A##_d,*A##_h; A##_h = (DAT*)malloc((nx)*(ny)*(nz)*sizeof(DAT)); \
                           for(i=0; i < (nx)*(ny)*(nz); i++){ A##_h[i]=(DAT)0.0; }              \
                           cudaMalloc(&A##_d      ,(nx)*(ny)*(nz)*sizeof(DAT));                 \
                           cudaMemcpy( A##_d,A##_h,(nx)*(ny)*(nz)*sizeof(DAT),cudaMemcpyHostToDevice);
#define  free_all(A)       free(A##_h);cudaFree(A##_d);

#define BLOCK_X   1024
#define BLOCK_Y   1
#define BLOCK_Z   1
#define GRID_X    10*2048*80
#define GRID_Y    1
#define GRID_Z    1
 
const int nx = GRID_X*BLOCK_X;
const int ny = GRID_Y*BLOCK_Y;
const int nz = GRID_Z*BLOCK_Z;
const int nt = 100;

// Timer
double timer_start = 0;
double cpu_sec(){ struct timeval tp; gettimeofday(&tp,NULL); return tp.tv_sec+1e-6*tp.tv_usec; }
void   tic(){ timer_start = cpu_sec(); }
double toc(){ return cpu_sec()-timer_start; }
void   tim(const char *what, double n){ double s=toc(); printf("%s: %8.3f seconds",what,s);if(n>0)printf(", %8.3f GB/s", n/s); printf("\n"); }

void  clean_cuda(){ 
  cudaError_t ce = cudaGetLastError();
  if(ce != cudaSuccess){ printf("ERROR launching GPU C-CUDA program: %s\n", cudaGetErrorString(ce)); cudaDeviceReset();}
}

__global__ void memcopy(DAT*Te, const int nx,const int ny,const int nz){  
  int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
  int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension x
  int iz  = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension x
  if (iz<nz && iy<ny && ix<nx) Te[ix + iy*nx + iz*nx*ny] = Te[ix + iy*nx + iz*nx*ny] + (DAT)1.0;
}
////////// main //////////
int main(){
  size_t i, it, N=nx*ny*nz, mem=N*sizeof(DAT);
  dim3 grid, block;
  block.x = BLOCK_X; block.y = BLOCK_Y; block.z = BLOCK_Z;
  grid.x  = GRID_X;  grid.y  = GRID_Y;  block.z = BLOCK_Z;
  int gpu_id=-1; gpu_id=GPU_ID; cudaSetDevice(gpu_id); cudaGetDevice(&gpu_id);
  cudaDeviceReset(); cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // set L1 to prefered
  printf("Process uses GPU with id %d.\n",gpu_id);
  printf("%dx%dx%d, %1.3f GB, %d iterations.\n", nx,ny,nz, 1*mem/1024./1024./1024., nt);
  printf("launching (%dx%dx%d) grid of (%dx%d) blocks.\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
  // initializations  
  zeros(Te, nx,ny,nz);
  // time loop
  for(it=0; it<nt; it++){ 
    if (it==3){ tic(); }       
    memcopy<<<grid, block>>>(Te_d,nx,ny,nz);  cudaDeviceSynchronize();
  }
  tim("Time (s), Effective MTP (GB/s)", mem*(nt-3)*2/1024./1024./1024.);
  free_all(Te);
  clean_cuda();
}
