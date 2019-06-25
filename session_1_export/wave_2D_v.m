clear % Wave 2D loop
% Physics
Lx  = 10;
Ly  = 10;
k   = 1;
rho = 1;
% Numerics
nx  = 100;
ny  = 40;
dx  = Lx/nx;
dy  = Ly/ny;
nt  = 200;
dt  = min([dx,dy])/sqrt(k/rho)/2.1;
% Initial arrays
x   = (-Lx+dx)/2:dx:(Lx-dx)/2;
y   = (-Ly+dy)/2:dy:(Ly-dy)/2;
[x2,y2] = ndgrid(x,y);
% Initial Conditions
P    = exp(-(x2.^2+y2.^2));
Pini = P;
Vx   = zeros(nx+1,ny  );
Vy   = zeros(nx  ,ny+1);
%Action
for it = 1:nt
    Vx(2:end-1,:) = Vx(2:end-1,:) - dt*diff(P,1,1)/dx/rho;
    Vy(:,2:end-1) = Vy(:,2:end-1) - dt*diff(P,1,2)/dy/rho;
    P             = P           - dt*(diff(Vx,1,1)/dx + diff(Vy,1,2)/dy)*k;
    % Plot
    figure(1),clf,pcolor(x2,y2,P),title(it)
    axis equal
    drawnow
    
end