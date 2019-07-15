clear
% mpicc -O3 MPI_Wave_1D.c
% nvcc -arch=sm_52 --compiler-bindir mpic++ -O3 MPI_Wave_1D.cu
% mpirun -np XX a.out
nprocs = 4;
% load data
for ip = 1:nprocs
    me = ip-1; fname = [num2str(me) '_P_c.dat'];
    id = fopen(fname); P_loc = fread(id,'float'); fclose(id);
    nx = length(P_loc);
    i1 = 1 + (ip-1)*(nx-2);
    P(i1:i1+nx-2) = P_loc(1:end-1);
end

figure(3),clf,colormap(jet)
plot(P,'-d'),axis([0 length(P) 0 1]),title('MPI Wave 1-D'),drawnow

% system('rm *.dat *.out')
