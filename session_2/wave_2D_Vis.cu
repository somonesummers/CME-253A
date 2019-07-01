// wave 2D GPU
// nvcc -arch=sm_70 -O3 wave_2D_Vis.cu
// run: ./a.out
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "cuda.h"

// #define USE_SINGLE_PRECISION    /* Comment this line using "//" if you want to use double precision.  */
#ifdef USE_SINGLE_PRECISION
#define DAT     float
#define PRECIS  4
#else
#define DAT     double
#define PRECIS  8
#endif

#define GPU_ID        0
#define OVERLENGTH_X  1
#define OVERLENGTH_Y  1

#define zeros(A,nx,ny)  DAT *A##_d,*A##_h; A##_h = (DAT*)malloc(((nx)*(ny))*sizeof(DAT)); \
                        for(i=0; i < ((nx)*(ny)); i++){ A##_h[i]=(DAT)0.0; }              \
                        cudaMalloc(&A##_d      ,((nx)*(ny))*sizeof(DAT));                 \
                        cudaMemcpy( A##_d,A##_h,((nx)*(ny))*sizeof(DAT),cudaMemcpyHostToDevice);
#define free_all(A)     free(A##_h); cudaFree(A##_d);
#define gather(A,nx,ny) cudaMemcpy( A##_h,A##_d,((nx)*(ny))*sizeof(DAT),cudaMemcpyDeviceToHost);
// --------------------------------------------------------------------- //
// Physics
const DAT Lx  = 10.0;
const DAT Ly  = 10.0;
const DAT k   = 1.0;
const DAT rho = 1.0;
const DAT mu  = 1.0;
// Numerics
#define BLOCK_X 32
#define BLOCK_Y 32
#define GRID_X  4
#define GRID_Y  4
const int nx = BLOCK_X*GRID_X - OVERLENGTH_X;
const int ny = BLOCK_Y*GRID_Y - OVERLENGTH_Y;
const int nt = 200;
const DAT dx = Lx/((DAT)nx);
const DAT dy = Ly/((DAT)ny);
const DAT dt = (min(dx,dy)*min(dx,dy))/(mu*4.1*3*4);
// --------------------------------------------------------------------- //
void save_info(int me, const int nx, const int ny){
    FILE* fid;
    if (me==0){ fid=fopen("0_nxy.inf" ,"w"); fprintf(fid,"%d %d %d", PRECIS, nx, ny); fclose(fid); }
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
// --------------------------------------------------------------------- //
// Computing physics kernels
__global__ void init(DAT* x, DAT* y, DAT* P, const DAT Lx, const DAT Ly, const DAT dx, const DAT dy, const int nx, const int ny){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    if (iy<ny && ix<nx){ x[ix + iy*nx] = (DAT)ix*dx + (-Lx+dx)/2.0; }
    if (iy<ny && ix<nx){ y[ix + iy*nx] = (DAT)iy*dy + (-Ly+dy)/2.0; }
    if (iy<ny && ix<nx){ P[ix + iy*nx] = exp(-(x[ix + iy*nx]*x[ix + iy*nx]) -(y[ix + iy*nx]*y[ix + iy*nx])); }
}
__global__ void compute_V(DAT* Vx, DAT* Vy, DAT* P, DAT* Txx, DAT* Tyy, DAT* Txy, const DAT dt, const DAT rho, const DAT dx, const DAT dy, const int nx, const int ny){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    if (iy<ny && ix>0 && ix<nx){
        Vx[ix+(iy)*(nx+1)] = Vx[ix+(iy)*(nx+1)] + dt/rho*(
               (-1)*(P[ix+(iy  )* nx   ] -   P[(ix-1)+(iy)* nx   ])/dx 
                + (Txx[ix+(iy  )* nx   ] - Txx[(ix-1)+(iy)* nx   ])/dx 
                + (Txy[ix+(iy+1)*(nx+1)] - Txy[ ix   +(iy)*(nx+1)])/dy);
    }
    if (iy>0 && iy<ny && ix<nx){
        Vy[ix+(iy)*(nx)] = Vy[ix+(iy)*(nx)] + dt/rho*(
                (-1)*(P[ ix   +(iy)* nx   ] -   P[ix+(iy-1)* nx   ])/dy 
                 + (Tyy[ ix   +(iy)* nx   ] - Tyy[ix+(iy-1)* nx   ])/dy 
                 + (Txy[(ix+1)+(iy)*(nx+1)] - Txy[ix+(iy  )*(nx+1)])/dx);
    }
}
__global__ void compute_P(DAT* Vx, DAT* Vy, DAT* P, const DAT dt, const DAT k, const DAT dx, const DAT dy, const int nx, const int ny){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    if (iy<ny && ix<nx){
        P[ix + iy*nx] = P[ix + iy*nx] - dt*k*(
                (Vx[(ix+1) +  iy   *(nx+1)]-Vx[ix + iy*(nx+1)])/dx 
              + (Vy[ ix    + (iy+1)* nx   ]-Vy[ix + iy* nx   ])/dy ); 
    }
}
__global__ void compute_T(DAT* Vx, DAT* Vy, DAT* P, DAT* Txx, DAT* Tyy, DAT* Txy, const DAT mu, const DAT dt, const DAT dx, const DAT dy, const int nx, const int ny){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    if (iy<ny && ix<nx){
        Txx[ix+iy*nx] = 2*mu*((Vx[(ix+1)+iy*(nx+1)]-Vx[ix+(iy)*(nx+1)])/dx  - 1/3*((Vx[(ix+1) + iy    *(nx+1)]-Vx[ix + iy*(nx+1)])/dx + (Vy[ ix    + (iy+1)* nx   ]-Vy[ix + iy* nx   ])/dy ));
        Tyy[ix+iy*nx] = 2*mu*((Vx[(ix)+(iy+1)*(nx+1)]-Vx[ix+(iy)*(nx+1)])/dy - 1/3*((Vx[(ix+1) + iy    *(nx+1)]-Vx[ix + iy*(nx+1)])/dx + (Vy[ ix    + (iy+1)* nx   ]-Vy[ix + iy* nx   ])/dy ));
    }
    if(iy<ny && ix<nx && ix>0  && iy >0){
        Txy[ix+iy*(nx+1)] = mu*((Vx[ix+(iy)*(nx+1)] - Vx[ix+(iy-1)*(nx+1)])/dy + (Vy[ix+(iy)*(nx)] - Vy[(ix-1)+(iy)*(nx)])/dx);
    }
}
int main(){
    int i, it;
    // Set up GPU
    int gpu_id=-1;
    int me = 0;
    dim3 grid, block;
    block.x = BLOCK_X; grid.x = GRID_X;
    block.y = BLOCK_Y; grid.y = GRID_Y;
    gpu_id = GPU_ID; cudaSetDevice(gpu_id); cudaGetDevice(&gpu_id);
    cudaDeviceReset(); cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // set L1 to prefered
    if (me==0){ printf("Process uses GPU with id %d.\n",gpu_id); }
    // Initial arrays
    zeros(x  ,nx  ,ny  );
    zeros(y  ,nx  ,ny  );
    zeros(P  ,nx  ,ny  );
    zeros(Vx ,nx+1,ny  );
    zeros(Vy ,nx  ,ny+1);
    zeros(Txx,(nx  )*(ny  ),1);
    zeros(Tyy,(nx  )*(ny  ),1);
    zeros(Txy,(nx+1)*(ny+1),1);
    // Initial conditions
    init<<<grid,block>>>(x_d, y_d, P_d, Lx, Ly, dx, dy, nx, ny);              cudaDeviceSynchronize();
    // Action
    for (it=0;it<nt;it++){
        compute_V<<<grid,block>>>(Vx_d, Vy_d, P_d, Txx_d, Tyy_d, Txy_d, dt, rho, dx, dy, nx, ny);  cudaDeviceSynchronize();
        compute_P<<<grid,block>>>(Vx_d, Vy_d, P_d, dt, k, dx, dy, nx, ny);  cudaDeviceSynchronize();
        compute_T<<<grid,block>>>(Vx_d, Vy_d, P_d, Txx_d, Tyy_d, Txy_d, mu, dt, dx, dy, nx, ny);  cudaDeviceSynchronize();
    }//it
    save_info();
    SaveArray(P ,nx  ,ny  ,"P" );
    SaveArray(Vx,nx+1,ny  ,"Vx");
    SaveArray(Vy,nx  ,ny+1,"Vy");
    free_all(x );
    free_all(y );
    free_all(P );
    free_all(Vx);
    free_all(Vy);
	free_all(Txx);
    free_all(Tyy);
    free_all(Txy);
    clean_cuda();
}
