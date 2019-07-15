// Wave 1D GPU Cuda aware MPI
// nvcc -arch=sm_52 --compiler-bindir mpic++ --compiler-options -O3 MPI_Wave_1D.cu
// mpirun -np XX a.out
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "cuda.h"
#include "mpi.h"

#define NDIMS      1
#define DIMS_X     0
#define USE_SINGLE_PRECIS

#define GPU_ID     0
#define BLOCK_X    100
#define GRID_X     1
#define OVERLENGTH 1

#ifdef USE_SINGLE_PRECIS
#define DAT      float
#define MPI_DAT  MPI_REAL
#else
#define DAT      double
#define MPI_DAT  MPI_DOUBLE_PRECISION
#endif
#define NREQS          (2*2*NDIMS)
#define zeros(A,N)      DAT *A##_d,*A##_h; A##_h = (DAT*)malloc((N)*sizeof(DAT)); \
                        for(i=0; i < (N); i++){ A##_h[i]=(DAT)0.0;}               \
                        cudaMalloc(&A##_d      ,(N)*sizeof(DAT));                 \
                        cudaMemcpy( A##_d,A##_h,(N)*sizeof(DAT),cudaMemcpyHostToDevice);
#define zeros_d(A,N)    DAT *A##_d; cudaMalloc(&A##_d,(N)*sizeof(DAT));
#define free_all(A)     free(A##_h); cudaFree(A##_d);
#define gather(A,N)     cudaMemcpy(A##_h,A##_d,(N)*sizeof(DAT),cudaMemcpyDeviceToHost);

void save_array(DAT* A, int N, int me, const char A_name[]){
char* fname; FILE* fid; asprintf(&fname, "%d_%s.dat" , me, A_name);
fid=fopen(fname, "wb"); fwrite(A, sizeof(DAT), N, fid); fclose(fid); free(fname);
}
#define SaveArray(A,N,A_name) gather(A,N); save_array(A##_h, N, me, A_name);

void  clean_cuda(){ 
  cudaError_t ce = cudaGetLastError();
  if(ce != cudaSuccess){ printf("ERROR launching GPU C-CUDA program: %s\n", cudaGetErrorString(ce)); cudaDeviceReset();}
}
// --------------------------------------------------------------------- //
// Physics
const DAT Lx  = 10.0; // Global
const DAT k   = 1.0;
const DAT rho = 1.0;
// Numerics
const int nx = BLOCK_X*GRID_X-OVERLENGTH;  // Local
const int nt = 2800;
size_t  Nix;         // Global nx
DAT     dt, dx;
// Computing physics kernels
__global__ void init(int coords[], DAT* x, DAT* P, const DAT Lx, DAT dx, const int nx){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    if (ix<nx){ x[ix] = (DAT)(coords[0]*(nx-2) + ix)*dx + (DAT)0.5*(-Lx+dx); }
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
// MPI stuff
__global__ void write_to_mpi_sendbuffer_l0(DAT* V_send_l0, DAT* V){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    if(ix==2){      V_send_l0[0] = V[ix]; }
}
__global__ void write_to_mpi_sendbuffer_r0(DAT* V_send_r0, DAT* V, const int nx){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    if(ix==(nx-2)){ V_send_r0[0] = V[ix]; }
}
__global__ void read_from_mpi_recvbuffer_l0(DAT* V, DAT* V_recv_l0){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    if(ix==0 ){ V[ix] = V_recv_l0[0]; }
}
__global__ void read_from_mpi_recvbuffer_r0(DAT* V, DAT* V_recv_r0, const int nx){
    int ix = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    if(ix==nx){ V[ix] = V_recv_r0[0]; }
}

#define update_sides() \
if (neighbours_l[0] != MPI_PROC_NULL)   write_to_mpi_sendbuffer_l0<<<grid,block>>>(V_send_l0_d,V_d);     cudaDeviceSynchronize(); \
if (neighbours_r[0] != MPI_PROC_NULL)   write_to_mpi_sendbuffer_r0<<<grid,block>>>(V_send_r0_d,V_d,nx);  cudaDeviceSynchronize(); \
if (neighbours_l[0] != MPI_PROC_NULL){  MPI_Irecv((DAT*)V_recv_l0_d, 1, MPI_DAT, neighbours_l[0], tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
if (neighbours_r[0] != MPI_PROC_NULL){  MPI_Irecv((DAT*)V_recv_r0_d, 1, MPI_DAT, neighbours_r[0], tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
if (neighbours_l[0] != MPI_PROC_NULL){  MPI_Isend((DAT*)V_send_l0_d, 1, MPI_DAT, neighbours_l[0], tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
if (neighbours_r[0] != MPI_PROC_NULL){  MPI_Isend((DAT*)V_send_r0_d, 1, MPI_DAT, neighbours_r[0], tag, topo_comm, &(req[reqnr]));  reqnr++;  } \
MPI_Waitall(reqnr,req,MPI_STATUSES_IGNORE);  reqnr=0;  for (int j=0; j<NREQS; j++){ req[j]=MPI_REQUEST_NULL; }; \
if (neighbours_l[0] != MPI_PROC_NULL)   read_from_mpi_recvbuffer_l0<<<grid,block>>>(V_d, V_recv_l0_d);     cudaDeviceSynchronize(); \
if (neighbours_r[0] != MPI_PROC_NULL)   read_from_mpi_recvbuffer_r0<<<grid,block>>>(V_d, V_recv_r0_d, nx); cudaDeviceSynchronize(); 

int main(int argc, char *argv[]){
    int i, it;
    // Set up GPU
    int gpu_id=-1;
    dim3 grid, block;
    block.x = BLOCK_X; grid.x = GRID_X;
    // MPI
    int dims[3]={DIMS_X,1,1};
    int coords[3]={0,0,0};
    int* coords_d=NULL;
    int nprocs=-1, me=-1, me_loc=-1;
    int neighbours_l[1]={0};
    int neighbours_r[1]={0};
    int reqnr=0, tag=0;
    int periods[NDIMS]={0};
    int reorder=1;
    MPI_Comm    topo_comm=MPI_COMM_NULL;
    MPI_Request req[NREQS]={MPI_REQUEST_NULL};
    cudaSetDeviceFlags(cudaDeviceMapHost); // DEBUG: needs to be set before context creation !
    const char* me_str     = getenv("OMPI_COMM_WORLD_RANK");
    const char* me_loc_str = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    me     = atoi(me_str);
    me_loc = atoi(me_loc_str);
    gpu_id = me_loc;
    //gpu_id = GPU_ID; // if no MPI 
    cudaSetDevice(gpu_id); cudaGetDevice(&gpu_id);
    cudaDeviceReset(); cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // set L1 to prefered 
    MPI_Init(&argc,&argv); // start of MPI world
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Dims_create(nprocs, NDIMS, dims);
    MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, periods, reorder, &topo_comm);
    MPI_Comm_rank(topo_comm, &me);
    MPI_Cart_coords(topo_comm, me, NDIMS, coords);
    cudaMalloc(&coords_d,3*sizeof(int)); cudaMemcpy(coords_d ,coords,3*sizeof(int),cudaMemcpyHostToDevice);
    for (int i=0; i<NDIMS; i++){ MPI_Cart_shift(topo_comm,i,1,&(neighbours_l[i]),&(neighbours_r[i])); }
    if (me==0){ printf("nprocs=%d, dims(1)=%d\n",nprocs,dims[0]); }
    printf("Process %d uses GPU with id %d.\n",me,gpu_id);
    Nix = ((nx-2)*dims[0])+2; // Global nx
    dx  = Lx/((DAT)Nix);
    dt  = dx/sqrt(k/rho)/2.1;
    // Initial arrays
    zeros(x, nx  );    
    zeros(P, nx  );
    zeros(V, nx+1);
    // MPI buffers
    zeros_d(V_recv_l0,1);
    zeros_d(V_send_l0,1);
    zeros_d(V_recv_r0,1);
    zeros_d(V_send_r0,1);
    // Initial conditions
    init<<<grid,block>>>(coords_d, x_d, P_d, Lx, dx, nx); cudaDeviceSynchronize();
    // Action
    for (it=0;it<nt;it++){
        compute_V<<<grid,block>>>(V_d, P_d, dt, rho, dx, nx); cudaDeviceSynchronize();
        update_sides();
        compute_P<<<grid,block>>>(V_d, P_d, dt, k  , dx, nx); cudaDeviceSynchronize();
    }//it
    SaveArray(P,nx,"P_c");
    free_all(x);
    free_all(P);
    free_all(V);
    clean_cuda();
    // MPI
    cudaFree(V_recv_l0_d);
    cudaFree(V_send_l0_d);
    cudaFree(V_recv_r0_d);
    cudaFree(V_send_r0_d);
    MPI_Finalize();
    return 0;
}
