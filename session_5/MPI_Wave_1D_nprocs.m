clear,figure(1),clf
% physics
Lx  = 10;
k   = 1;
rho = 1;
% numerics
np  = 10;          % number of processes
nx  = 150/np;      % LOCAL number of grid points at each processes
dx  = Lx/nx;
nt  = 200;
dt  = dx/sqrt(k/rho)/2.1;
nxg = (nx-2)*np+2;  % global number of grid points
dxg = Lx/(nxg-1);
% init local vectors
for ip = 1:np
    i1 = 1 + (ip-1)*(nx-2);
    for ix=1:nx
        x(ix,ip) = ( (ip-1)*(nx-2) + (ix-1) )*dxg + (-Lx+dxg)/2;
        P(ix,ip) = exp(-x(ix,ip).^2);
    end
    for ix=1:nx+1
        V(ix,ip) = 0;
    end
end
% action
for it=1:nt
    % compute physics V locally
    for ip = 1:np
        V(2:end-1,ip) = V(2:end-1,ip) - dt*diff(P(:,ip))/dx/rho;
    end
    % update boundaries
    for ip = 1:np-1
        V(end,ip  ) = V(    3,ip+1);
        V(  1,ip+1) = V(end-2,ip  );
    end
    % compute physics P locally
    for ip = 1:np
        P(:,ip)       = P(:,ip)       - dt*diff(V(:,ip))/dx*k;
    end
    % global picture
    for ip = 1:np
         i1 = 1 + (ip-1)*(nx-2);
         Pg(i1:i1+nx-2) = P(1:end-1,ip);
    end
    % post-process
    plot(Pg,'-d'),title(it),axis([0 nxg 0 1]),drawnow
end
