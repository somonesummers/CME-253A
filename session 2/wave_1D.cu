// wave 1D GPU
// compile: nvcc -arch=sm_70 -O3 wave_1D.cu
// run: ./a.out
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "cuda.h"

#define DAT double
#define GPU_ID     0
#define BLOCK_X    100
#define GRID_X     1
#define OVERLENGTH 1

#define zeros(A,N)     DAT *A##_d,*A##_h; A##_h = (DAT*)malloc((N)*sizeof(DAT)); \
                       for(i=0; i < (N); i++){ A##_h[i]=(DAT)0.0; }              \
                       cudaMalloc(&A##_d      ,(N)*sizeof(DAT));                 \
                       cudaMemcpy( A##_d,A##_h,(N)*sizeof(DAT),cudaMemcpyHostToDevice);
#define free_all(A)    free(A##_h); cudaFree(A##_d);
#define gather(A,N)    cudaMemcpy( A##_h,A##_d,(N)*sizeof(DAT),cudaMemcpyDeviceToHost);
                       
void save_array(DAT* A, int N, const char A_name[]){
    char* fname; FILE* fid; asprintf(&fname, "%s.dat" , A_name);
    fid=fopen(fname, "wb"); fwrite(A, sizeof(DAT), N, fid); fclose(fid); free(fname);
}
#define SaveArray(A,N,A_name) gather(A,N); save_array(A##_h, N, A_name);

void  clean_cuda(){ 
    cudaError_t ce = cudaGetLastError();
    if(ce != cudaSuccess){ printf("ERROR launching GPU C-CUDA program: %s\n", cudaGetErrorString(ce)); cudaDeviceReset();}
}
// --------------------------------------------------------------------- //
// Physics
const DAT Lx  = 10.0;
const DAT k   = 1.0;
const DAT rho = 1.0;
// Numerics
const int nx  = BLOCK_X*GRID_X-OVERLENGTH;
const int nt  = 200;
const DAT dx  = Lx/((DAT)nx);
const DAT dt  = dx/sqrt(k/rho)/2.1;
// Computing physics kernels
__global__ void init(DAT* x, DAT* P, const DAT Lx, const DAT dx, const int nx){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    if (ix<nx){ x[ix] = (DAT)ix*dx + (-Lx+dx)/2.0; }
    if (ix<nx){ P[ix] = exp(-(x[ix]*x[ix])); }
}
__global__ void compute_V(DAT* V, DAT* P, const DAT dt, const DAT rho, const DAT dx, const int nx){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    if (ix>0 && ix<nx){ V[ix] = V[ix] - dt*(P[ix]-P[ix-1])/dx/rho; }
}
__global__ void compute_P(DAT* V, DAT* P, const DAT dt, const DAT k, const DAT dx, const int nx){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    if (ix<nx){ P[ix] = P[ix] - dt*(V[ix+1]-V[ix])/dx*k; }
}
int main(){
    int i, it;
    // Set up GPU
    int  gpu_id=-1;
    dim3 grid, block;
    block.x = BLOCK_X; grid.x = GRID_X;
    gpu_id = GPU_ID; cudaSetDevice(gpu_id); cudaGetDevice(&gpu_id);
    cudaDeviceReset(); cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // set L1 to prefered
    printf("Process uses GPU with id %d .\n",gpu_id);
    // Initial arrays
    zeros(x,nx  );    
    zeros(P,nx  );
    zeros(V,nx+1);
    // Initial conditions
    init<<<grid,block>>>(x_d, P_d, Lx, dx, nx); cudaDeviceSynchronize();
    // Action
    for (it=0;it<nt;it++){
        compute_V<<<grid,block>>>(V_d, P_d, dt, rho, dx, nx); cudaDeviceSynchronize();
        compute_P<<<grid,block>>>(V_d, P_d, dt, k  , dx, nx); cudaDeviceSynchronize();
    }//it
    SaveArray(P,nx,"P_c");
    free_all(x);
    free_all(P);
    free_all(V);
    clean_cuda();
}
