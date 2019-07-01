// wave 2D GPU
// compile: nvcc -arch=sm_70 -O3 wave_2D_v3.cu
// run: ./a.out
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "cuda.h"

#define USE_SINGLE_PRECISION    /* Comment this line using "//" if you want to use double precision.  */
#ifdef USE_SINGLE_PRECISION
#define DAT     float
#define PRECIS  4
#else
#define DAT     double
#define PRECIS  8
#endif

#define GPU_ID        3
#define OVERLENGTH_X  1
#define OVERLENGTH_Y  1

#define zeros(A,nx,ny)  DAT *A##_d,*A##_h; A##_h = (DAT*)malloc(((nx)*(ny))*sizeof(DAT)); \
                        for(i=0; i < ((nx)*(ny)); i++){ A##_h[i]=(DAT)0.0; }              \
                        cudaMalloc(&A##_d      ,((nx)*(ny))*sizeof(DAT));                 \
                        cudaMemcpy( A##_d,A##_h,((nx)*(ny))*sizeof(DAT),cudaMemcpyHostToDevice);
#define free_all(A)     free(A##_h); cudaFree(A##_d);
#define gather(A,nx,ny) cudaMemcpy( A##_h,A##_d,((nx)*(ny))*sizeof(DAT),cudaMemcpyDeviceToHost);

#define for_ix          int ix = blockIdx.x*blockDim.x + threadIdx.x;
#define for_iy          int iy = blockIdx.y*blockDim.y + threadIdx.y;                              
#define Pres(ix,iy)     ( P[ix + (iy)*nx    ]) 
#define Velx(ix,iy)     (Vx[ix + (iy)*(nx+1)])        
#define Vely(ix,iy)     (Vy[ix + (iy)*nx    ]) 
// --------------------------------------------------------------------- //
// Physics
const DAT Lx  = 10.0;
const DAT Ly  = 10.0;
const DAT k   = 1.0;
const DAT rho = 1.0;
// Numerics
#define BLOCK_X 32
#define BLOCK_Y 32
#define GRID_X  320
#define GRID_Y  320
const int nx  = BLOCK_X*GRID_X - OVERLENGTH_X;
const int ny  = BLOCK_Y*GRID_Y - OVERLENGTH_Y;
const int nt  = 200;
const DAT dx  = Lx/((DAT)nx);
const DAT dy  = Ly/((DAT)ny);
const DAT dt  = min(dx,dy)/sqrt(k/rho)/4.1;
// --------------------------------------------------------------------- //
void save_info(int me, const int nx, const int ny){
    FILE* fid;
    fid=fopen("0_nxy.inf" ,"w"); fprintf(fid,"%d %d %d", PRECIS, nx, ny); fclose(fid);
}
#define save_info() save_info(me, nx, ny);

void save_array(DAT* A, int nx, int ny, int me, const char A_name[]){
    char* fname; FILE* fid; asprintf(&fname, "%d_%s.res" , me, A_name);
    fid=fopen(fname, "wb"); fwrite(A, sizeof(DAT), (nx)*(ny), fid); fclose(fid); free(fname);
}
#define SaveArray(A,nx,ny,A_name) gather(A,nx,ny); save_array(A##_h,nx,ny,me,A_name);

void  clean_cuda(){ 
    cudaError_t ce = cudaGetLastError();
    if(ce != cudaSuccess){ printf("ERROR launching GPU C-CUDA program: %s\n", cudaGetErrorString(ce)); cudaDeviceReset();}
}
// Timer
#include "sys/time.h"
double timer_start = 0;
double cpu_sec(){ struct timeval tp; gettimeofday(&tp,NULL); return tp.tv_sec+1e-6*tp.tv_usec; }
void   tic(){ timer_start = cpu_sec(); }
double toc(){ return cpu_sec()-timer_start; }
void   tim(const char *what, double n){ double s=toc(); printf("%s: %8.3f seconds",what,s);if(n>0)printf(", %8.3f GB/s", n/s); printf("\n"); }
// --------------------------------------------------------------------- //
// Computing physics kernels
__global__ void init(DAT* x, DAT* y, DAT* P, const DAT Lx, const DAT Ly, const DAT dx, const DAT dy, const int nx, const int ny){
    for_ix for_iy
    if (iy<ny && ix<nx){ x[ix + iy*nx] = (DAT)ix*dx + (-Lx+dx)/2.0; }
    if (iy<ny && ix<nx){ y[ix + iy*nx] = (DAT)iy*dy + (-Ly+dy)/2.0; }
    if (iy<ny && ix<nx){ P[ix + iy*nx] = exp(-(x[ix + iy*nx]*x[ix + iy*nx]) -(y[ix + iy*nx]*y[ix + iy*nx])); }
}
__global__ void compute_V(DAT* Vx, DAT* Vy, DAT* P, DAT dt_dx_rho, DAT dt_dy_rho, const int nx, const int ny){
    for_ix for_iy
    if (iy<ny && ix>0 && ix<nx){ Velx(ix,iy) = Velx(ix,iy) - dt_dx_rho*(Pres(ix,iy)-Pres(ix-1,iy)); }
    if (iy>0 && iy<ny && ix<nx){ Vely(ix,iy) = Vely(ix,iy) - dt_dy_rho*(Pres(ix,iy)-Pres(ix,iy-1)); }
}
__global__ void compute_P(DAT* Vx, DAT* Vy, DAT* P, DAT dtk_dx, DAT dtk_dy, const int nx, const int ny){
    for_ix for_iy
    if (iy<ny && ix<nx){ Pres(ix,iy) = Pres(ix,iy) - (dtk_dx*(Velx(ix+1,iy)-Velx(ix,iy)) + dtk_dy*(Vely(ix,iy+1)-Vely(ix,iy)) ); }
}
int main(){
    int i, it;
    size_t N=nx*ny, mem=N*sizeof(DAT);
    // Set up GPU
    int gpu_id=-1;
    dim3 grid, block;
    block.x = BLOCK_X; grid.x = GRID_X;
    block.y = BLOCK_Y; grid.y = GRID_Y;
    gpu_id = GPU_ID; cudaSetDevice(gpu_id); cudaGetDevice(&gpu_id);
    cudaDeviceReset(); cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // set L1 to prefered
    printf("Process uses GPU with id %d.\n",gpu_id); 
    printf("%dx%d, %1.3f GB, %d iterations.\n", nx,ny, 5*mem/1024./1024./1024., nt);
    printf("Launching (%dx%d) grid of (%dx%d) blocks.\n", grid.x, grid.y, block.x, block.y);
    // Initial arrays
    zeros(x  ,nx  ,ny  );
    zeros(y  ,nx  ,ny  );
    zeros(P  ,nx  ,ny  );
    zeros(Vx ,nx+1,ny  );
    zeros(Vy ,nx  ,ny+1);
    // Initial conditions
    init<<<grid,block>>>(x_d, y_d, P_d, Lx, Ly, dx, dy, nx, ny);              cudaDeviceSynchronize();
    DAT dt_dx_rho = dt/dx/rho;
    DAT dt_dy_rho = dt/dy/rho;
    DAT dtk_dx = dt*k/dx;
    DAT dtk_dy = dt*k/dy;
    // Action
    for (it=0;it<nt;it++){
        if (it==3){ tic(); } 
        compute_V<<<grid,block>>>(Vx_d, Vy_d, P_d, dt_dx_rho, dt_dy_rho, nx, ny);  cudaDeviceSynchronize();
        compute_P<<<grid,block>>>(Vx_d, Vy_d, P_d, dtk_dx, dtk_dy,       nx, ny);  cudaDeviceSynchronize();
    }//it
    tim("Time (s), Effective MTP (GB/s)", mem*(nt-3)*4/1024./1024./1024.);
    // save_info();
    // SaveArray(P ,nx  ,ny  ,"P" );
    // SaveArray(Vx,nx+1,ny  ,"Vx");
    // SaveArray(Vy,nx  ,ny+1,"Vy");
    free_all(x );
    free_all(y );
    free_all(P );
    free_all(Vx);
    free_all(Vy);
    clean_cuda();
}
