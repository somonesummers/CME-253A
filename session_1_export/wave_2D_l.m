%wave_2D_l

clear % Wave 2D loop
% Physics
Lx  = 10;
Ly  = 10;
k   = 1;
rho = 1;
% Numerics
nx  = 100;
ny  = 100;
dx  = Lx/nx;
dy  = Ly/ny;
nt  = 200;
dt  = min([dx,dy])/sqrt(k/rho)/2.1;
% Initial arrays
x    = zeros((nx  )*(ny  ),1);
y    = zeros((nx  )*(ny  ),1);
P    = zeros((nx  )*(ny  ),1);
Tyy    = zeros((nx  )*(ny  ),1);
Txx    = zeros((nx  )*(ny  ),1);
Txy    = zeros((nx  )*(ny  ),1);
Pini = zeros((nx  )*(ny  ),1);
Vx   = zeros((nx+1)*(ny  ),1);
Vy   = zeros((nx  )*(ny+1),1);
% Initial conditions
for ix = 1:nx
    for iy = 1:ny
        x(ix+(iy-1)*nx) = (ix-1)*dx + (-Lx+dx)/2;
        y(ix+(iy-1)*nx) = (iy-1)*dy + (-Ly+dy)/2;
        P(ix+(iy-1)*nx) = exp(-(x(ix+(iy-1)*nx)^2+y(ix+(iy-1)*nx)^2));
        
        Pini(ix+(iy-1)*nx)=P(ix+(iy-1)*nx);
    end
end
% Action
for it = 1:nt
    for ix = 2:nx
        for iy = 1:ny
            Vx(ix+(iy-1)*(nx+1)) = Vx(ix+(iy-1)*(nx+1)) - dt*(P(ix+(iy-1)*nx)-P((ix-1)+(iy-1)*nx))/dx/rho;
        end
    end
    for ix = 1:nx
        for iy = 2:ny
            Vy(ix+(iy-1)*(nx)) = Vy(ix+(iy-1)*(nx)) - dt*(P(ix+(iy-1)*nx)-P(ix+(iy-2)*nx))/dy/rho;
        end
    end
    for ix = 1:nx
        for iy = 1:ny
            P(ix+(iy-1)*nx) = P(ix+(iy-1)*nx) - dt*(Vx((ix+1)+(iy-1)*(nx+1))-Vx(ix+(iy-1)*(nx+1)))/dx*k...
                - dt*(Vy(ix+(iy)*(nx))-Vy(ix+(iy-1)*(nx)))/dy*k;
        end
    end
    % Plot
    figure(1),clf,scatter(x,y,[20],P),title(it)
    view(2)
    axis equal
    drawnow
end
