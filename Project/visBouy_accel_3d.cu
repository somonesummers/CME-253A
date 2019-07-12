// 3D Bouyant Ball viscous code
// nvcc -arch=sm_70 -O3 wave_2D_Vis_v2.cu
// run: ./a.out
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "cuda.h"

//#define USE_SINGLE_PRECISION    /* Comment this line using "//" if you want to use double precision.  */
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
#define OVERLENGTH_Z  1
        
#define zeros(A,nx,ny,nz)  DAT *A##_d,*A##_h; A##_h = (DAT*)malloc(((nx)*(ny)*(nz))*sizeof(DAT)); \
                        for(i=0; i < ((nx)*(ny)*(nz)); i++){ A##_h[i]=(DAT)0.0; }              \
                        cudaMalloc(&A##_d      ,((nx)*(ny)*(nz))*sizeof(DAT));                 \
                        cudaMemcpy( A##_d,A##_h,((nx)*(ny)*(nz))*sizeof(DAT),cudaMemcpyHostToDevice);
#define free_all(A)     free(A##_h); cudaFree(A##_d);
#define gather(A,nx,ny,nz) cudaMemcpy( A##_h,A##_d,((nx)*(ny)*(nz))*sizeof(DAT),cudaMemcpyDeviceToHost);
// --------------------------------------------------------------------- //
// Physics
const DAT Lx  = 10.0;
const DAT Ly  = 10.0;
const DAT Lz  = 10.0;
const DAT k   = 1.0;
const DAT rhoi= 10.0;
const DAT eta = 1.0;
const DAT nu  = 6.0;
const DAT epsi= 1.0e-6;
// Numerics
#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCK_Z 8
#define GRID_X  4
#define GRID_Y  4 
#define GRID_z  4 
const int nx = BLOCK_X*GRID_X - OVERLENGTH_X;
const int ny = BLOCK_Y*GRID_Y - OVERLENGTH_Y;
const int nz = BLOCK_Z*GRID_Z - OVERLENGTH_Z;
const int nt = 40000;
const DAT dx = Lx/((DAT)nx);
const DAT dy = Ly/((DAT)ny);
const DAT dz = Lz/((DAT)nz);
const DAT dtV = (min(dx,dy,dz)*min(dx,dy,dz))/(eta*4.1*((DAT)4));
const DAT dtP = 4.1*eta/((DAT)(4*ny));
// --------------------------------------------------------------------- //
void save_info(int me, const int nx, const int ny, const int nz){
    FILE* fid;
    if (me==0){ fid=fopen("0_nxy.inf" ,"w"); fprintf(fid,"%d %d %d %d", PRECIS, nx, ny, nz); fclose(fid); }
}
#define save_info() save_info(me, nx, ny, nz);

void save_array(DAT* A, int nx, int ny, int nz, int me, const char A_name[]){
    char* fname; FILE* fid; asprintf(&fname, "%d_%s.res" , me, A_name);
    fid=fopen(fname, "wb"); fwrite(A, sizeof(DAT), (nx)*(ny)*(nz), fid); fclose(fid); free(fname);
}
#define SaveArray(A,nx,ny,nz,A_name) gather(A,nx,ny,nz); save_array(A##_h,nx,ny,nz,me,A_name);

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
__global__ void init(DAT* x, DAT* y, DAT* z, DAT* rho, const DAT Lx, const DAT Ly, const DAT Lz, const DAT dx, const DAT dy, const DAT dz, const int nx, const int ny, const int nz){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z
    if (iy<ny && ix<nx && iz<nz){ x[ix+iy*nx+iz*nx*ny] = (DAT)ix*dx + (-Lx+dx)/2.0; }
    if (iy<ny && ix<nx && iz<nz){ y[ix+iy*nx+iz*nx*ny] = (DAT)iy*dy + (-Ly+dy)/2.0; }
    if (iy<ny && ix<nx && iz<nz){ z[ix+iy*nx+iz*nx*ny] = (DAT)iz*dz + (-Lz+dz)/2.0; }
    if (iy<ny && ix<nx && iz<nz){ 
        if(x[ix+iy*nx+iz*nx*ny]*x[ix+iy*nx+iz*nx*ny] + y[ix+iy*nx+iz*nx*ny]*y[ix+iy*nx+iz*nx*ny] + z[ix+iy*nx+iz*nx*ny]*z[ix+iy*nx+iz*nx*ny] < 1){
            rho[ix+iy*nx+iz*nx*ny]=rhoi;
        }
    }
}
__global__ void compute_V(DAT* Vx, DAT* Vy, DAT* P, DAT* Txx, DAT* Tyy, DAT* Txy, const DAT dt, const DAT rho, const DAT dx, const DAT dy, const int nx, const int ny){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z
    if (ix>0 && iy<ny && ix<nx && iz<nz){
       
    }
    if (iy>0 && iy<ny && ix<nx && iz<nz){
        
    }
    if (iz>0 && iy<ny && ix<nx && iz<nz){
        
    }
}
__global__ void compute_P(DAT* Vx, DAT* Vy, DAT* Vz, DAT* P, const DAT dt, const DAT k, const DAT dx, const DAT dy, const DAT dz, const int nx, const int ny, const int nz){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z
    if (iy<ny && ix<nx && iz<nz){
        P[ix+(iy)*nx+(iz)*nx*ny] = P[ix+(iy)*nx+(iz)*nx*ny] - dtP*k*(
                  (Vx[(ix+1)+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )]-Vx[(ix  )+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )])/dx+
                  (Vy[(ix  )+(iy+1)*(nx  )+(iz  )*(nx  )*(ny+1)]-Vy[(ix  )+(iy  )*(nx  )+(iz  )*(nx  )*(ny+1)])/dy+
                  (Vz[(ix  )+(iy  )*(nx  )+(iz+1)*(nx  )*(ny  )]-Vz[(ix  )+(iy  )*(nx  )+(iz  )*(nx  )*(ny  )])/dz);
    }
}
__global__ void compute_T(DAT* Vx, DAT* Vy, DAT* Vz, DAT* Txx, DAT* Tyy, DAT* Tzz, DAT* Txy, DAT* Txz, DAT* Tyz, const DAT eta, const DAT dx, const DAT dy, const DAT dz, const int nx, const int ny, const int nz){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int iz = blockIdx.z*blockDim.z + threadIdx.z; // thread ID, dimension z
    if (iy<ny && ix<nx && iz<nz){
        Txx[ix+(iy)*nx+(iz)*nx*ny] = 2*eta*(
              (Vx[(ix+1)+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )]-Vx[(ix  )+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )])/dx - 
             ((Vx[(ix+1)+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )]-Vx[(ix  )+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )])/dx+
              (Vy[(ix  )+(iy+1)*(nx  )+(iz  )*(nx  )*(ny+1)]-Vy[(ix  )+(iy  )*(nx  )+(iz  )*(nx  )*(ny+1)])/dy+
              (Vz[(ix  )+(iy  )*(nx  )+(iz+1)*(nx  )*(ny  )]-Vz[(ix  )+(iy  )*(nx  )+(iz  )*(nx  )*(ny  )])/dz)/((DAT)3));
        Tyy[ix+(iy)*nx+(iz)*nx*ny] = 2*eta*(
              (Vy[(ix  )+(iy+1)*(nx  )+(iz  )*(nx  )*(ny+1)]-Vy[(ix  )+(iy  )*(nx  )+(iz  )*(nx  )*(ny+1)])/dy - 
             ((Vx[(ix+1)+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )]-Vx[(ix  )+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )])/dx+
              (Vy[(ix  )+(iy+1)*(nx  )+(iz  )*(nx  )*(ny+1)]-Vy[(ix  )+(iy  )*(nx  )+(iz  )*(nx  )*(ny+1)])/dy+
              (Vz[(ix  )+(iy  )*(nx  )+(iz+1)*(nx  )*(ny  )]-Vz[(ix  )+(iy  )*(nx  )+(iz  )*(nx  )*(ny  )])/dz)/((DAT)3));
        Tzz[ix+(iy)*nx+(iz)*nx*ny] = 2*eta*(
              (Vz[(ix  )+(iy  )*(nx  )+(iz+1)*(nx  )*(ny  )]-Vz[(ix  )+(iy  )*(nx  )+(iz  )*(nx  )*(ny  )])/dz - 
             ((Vx[(ix+1)+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )]-Vx[(ix  )+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )])/dx+
              (Vy[(ix  )+(iy+1)*(nx  )+(iz  )*(nx  )*(ny+1)]-Vy[(ix  )+(iy  )*(nx  )+(iz  )*(nx  )*(ny+1)])/dy+
              (Vz[(ix  )+(iy  )*(nx  )+(iz+1)*(nx  )*(ny  )]-Vz[(ix  )+(iy  )*(nx  )+(iz  )*(nx  )*(ny  )])/dz)/((DAT)3));
    }
    if(iy<ny && ix<nx && iz<nz && ix>0  && iy >0){
        Txy[(ix)+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny+1)] = eta*(
            (Vx[(ix)+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )] - Vx[(ix  )+(iy-1)*(nx+1)+(iz  )*(nx+1)*(ny  )])/dy + 
            (Vy[(ix)+(iy  )*(nx  )+(iz  )*(nx  )*(ny+1)] - Vy[(ix-1)+(iy  )*(nx  )+(iz  )*(nx  )*(ny+1)])/dx);
    }
    if(iy<ny && ix<nx && iz<nz && ix>0  && iz >0){
        Txz[(ix)+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )] = eta*(
           (Vx[(ix)+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )] - Vx[(ix  )+(iy  )*(nx+1)+(iz-1)*(nx+1)*(ny  )])/dz + 
           (Vz[(ix)+(iy  )*(nx  )+(iz  )*(nx  )*(ny  )] - Vz[(ix-1)+(iy  )*(nx  )+(iz  )*(nx  )*(ny  )])/dx);
    }
    if(iy<ny && ix<nx && iz<nz && iy>0  && iz >0){
        Tyz[(ix)+(iy  )*(nx  )+(iz  )*(nx  )*(ny+1)] = eta*(
           (Vy[(ix)+(iy  )*(nx  )+(iz  )*(nx  )*(ny+1)] - Vy[(ix)+(iy  )*(nx  )+(iz-1)*(nx  )*(ny+1)])/dz + 
           (Vz[(ix)+(iy  )*(nx  )+(iz  )*(nx  )*(ny  )] - Vz[(ix)+(iy-1)*(nx  )+(iz  )*(nx  )*(ny  )])/dy);
    }
}
int main(){
    int i, it;
    size_t N=nx*ny*nz, mem=N*sizeof(DAT);
    // Set up GPU
    int gpu_id=-1;
    int me = 0;
    dim3 grid, block;
    block.x = BLOCK_X; grid.x = GRID_X;
    block.y = BLOCK_Y; grid.y = GRID_Y;
    block.z = BLOCK_Z; grid.z = GRID_Z;
    gpu_id = GPU_ID; cudaSetDevice(gpu_id); cudaGetDevice(&gpu_id);
    cudaDeviceReset(); cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // set L1 to prefered
    printf("Process uses GPU with id %d.\n",gpu_id);
    printf("%dx%d, %1.3f GB, %d iterations.\n", nx,ny, 5*mem/1024./1024./1024., nt);
    printf("Launching (%dx%dx%d) grid of (%dx%dx%d) blocks.\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
    // Initial arrays
    zeros(x    ,nx  ,ny  ,nz  );
    zeros(y    ,nx  ,ny  ,nz  );
    zeros(z    ,nx  ,ny  ,nz  );
    zeros(rho  ,nx  ,ny  ,nz  );
    zeros(P    ,nx  ,ny  ,nz  );
    zeros(Txx  ,nx  ,ny  ,nz  );
    zeros(Tyy  ,nx  ,ny  ,nz  );
    zeros(Tzz  ,nx  ,ny  ,nz  );
    zeros(Txy  ,nx+1,ny+1,nz  );
    zeros(Txz  ,nx+1,ny  ,nz+1);
    zeros(Tyz  ,nx  ,ny+1,nz+1);
    zeros(Vx   ,nx+1,ny  ,nz  );
    zeros(Vy   ,nx  ,ny+1,nz  );
    zeros(Vz   ,nx+1,ny  ,nz+1);
    zeros(dVxdt,nx+1,ny  ,nz  );
    zeros(dVydt,nx  ,ny+1,nz  );
    zeros(dVzdt,nx+1,ny  ,nz+1);
    zeros(Rx   ,nx+1,ny  ,nz  );
    zeros(Ry   ,nx  ,ny+1,nz  );
    zeros(Rz   ,nx+1,ny  ,nz+1);
    // Initial conditions
    init<<<grid,block>>>(x_d, y_d, z_d, rho_d, Lx, Ly, Lz, dx, dy, dz, nx, ny, nz);              cudaDeviceSynchronize();
    // Action
    for (it=0;it<nt;it++){
        if (it==1){ tic(); } 
        compute_V<<<grid,block>>>(Vx_d, Vy_d, P_d, Txx_d, Tyy_d, Txy_d, dt, rho, dx, dy, nx, ny);  cudaDeviceSynchronize();
        compute_P<<<grid,block>>>(Vx_d, Vy_d, P_d, dt, k, dx, dy, nx, ny);  cudaDeviceSynchronize();
        compute_T<<<grid,block>>>(Vx_d, Vy_d, P_d, Txx_d, Tyy_d, Txy_d, eta, dt, dx, dy, nx, ny);  cudaDeviceSynchronize();
    }//it
    tim("Time (s), Effective MTP (GB/s)", mem*(nt-3)*4/1024./1024./1024.);
    save_info();
    SaveArray(P ,nx  ,ny  ,nz  ,"P" );
    SaveArray(Vx,nx+1,ny  ,nz  ,"Vx");
    SaveArray(Vy,nx  ,ny+1,nz  ,"Vy");
    SaveArray(Vz,nx  ,ny  ,nz+1,"Vz");
    SaveArray(Txx,nx  ,ny  ,nz  ,"Txx");
    SaveArray(Tyy,nx  ,ny  ,nz  ,"Tyy");
    SaveArray(Tzz,nx  ,ny  ,nz  ,"Tzz");
    SaveArray(Txy,nx+1,ny+1,nz  ,"Txy");
    SaveArray(Txz,nx+1,ny  ,nz+1,"Txz");
    SaveArray(Tyz,nx  ,ny+1,nz+1,"Tyz");
    free_all(x );
    free_all(y );
    free_all(z );
    free_all(rho);
    free_all(P );
    free_all(Vx);
    free_all(Vy);
    free_all(Vz);
    free_all(dVxdt);
    free_all(dVydt);
    free_all(dVzdt);
    free_all(Rx);
    free_all(Ry);
    free_all(Rz);
	free_all(Txx);
    free_all(Tyy);
    free_all(Tzz);
    free_all(Txy);
    free_all(Txz);
    free_all(Tyz);
    clean_cuda();
}
