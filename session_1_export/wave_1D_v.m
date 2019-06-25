clear % Wave 1D vectorised
% Physics
Lx  = 10;
k   = 1;
rho = 1;
% Numerics
nx  = 100;
dx  = Lx/nx;
nt  = 200;
dt  = dx/sqrt(k/rho)/2.1;
x   = (-Lx+dx)/2:dx:(Lx-dx)/2;
% Initial conditions
P   = exp(-x.^2); Pini = P;
Vx  = [0*P 0];
% Action
for it = 1:nt
    Vx(2:end-1) = Vx(2:end-1) - dt*diff(P)/dx/rho;
    P           = P           - dt*diff(Vx)/dx*k;
    % Plot
    figure(1),clf,plot(x,Pini,'-g',x,P,'-d'),title(it)
    drawnow
end
