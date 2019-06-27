clear % Wave 1D loop
% Physics
Lx  = 10;
k   = 1;
rho = 1;
% Numerics
nx  = 100;
dx  = Lx/nx;
nt  = 200;
dt  = min([dx,dy])^2/mu/2.1; 
% Initial arrays
x    = zeros(nx  ,1);
P    = zeros(nx  ,1);
Pini = zeros(nx  ,1);
V    = zeros(nx+1,1);
% Initial conditions
for ix = 1:nx
    x(ix) = (ix-1)*dx + (-Lx+dx)/2;
end
for ix = 1:nx
    P(ix) = exp(-x(ix)^2); Pini(ix)=P(ix);
end
% Action
for it = 1:nt
    for ix = 2:nx
        V(ix) = V(ix) - dt*(P(ix)-P(ix-1))/dx/rho;
    end
    for ix = 1:nx
        P(ix) = P(ix) - dt*(V(ix+1)-V(ix))/dx*k;
    end
    % Plot
    figure(1),clf,plot(x,Pini,'-g',x,P,'-d'),title(it)
    drawnow
end
