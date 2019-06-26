clear % diffusion 1D vectorised
% Physics
Lx = 10;
D  = 1;
% Numerics
nx = 100;
dx = Lx/nx;
nt = 200;
dt = dx^2/D/2.1;
x  = (-Lx+dx)/2:dx:(Lx-dx)/2;
% Initial conditions
T  = exp(-x.^2); Tini = T;% figure(1),clf,plot(x,T,'-d'),drawnow
% Action
for it = 1:nt
    qx         = -D*diff(T )/dx;
    T(2:end-1) = T(2:end-1) + dt*(-diff(qx)/dx);
    % Plot
    figure(1),clf,plot(x,Tini,'-g',x,T,'-d'),title(it)
    drawnow
end
