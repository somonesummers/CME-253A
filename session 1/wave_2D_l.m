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
tic
for ix = 1:nx
    for iyM = 1:ny, iy = iyM-1;
        x(ix+(iy)*nx) = (ix-1)*dx + (-Lx+dx)/2;
        y(ix+(iy)*nx) = (iy)*dy + (-Ly+dy)/2;
        P(ix+(iy)*nx) = exp(-(x(ix+(iy)*nx)^2+y(ix+(iy)*nx)^2));
        
        Pini(ix+(iy)*nx)=P(ix+(iy)*nx);
    end
end
% Action
for it = 1:nt
    for ix = 2:nx
        for iyM = 1:ny, iy = iyM-1;
            Vx(ix+(iy)*(nx+1)) = Vx(ix+(iy)*(nx+1)) - dt*(P(ix+(iy)*nx)-P((ix-1)+(iy)*nx))/dx/rho;
        end
    end
    for ix = 1:nx
        for iyM = 2:ny, iy = iyM-1;
            Vy(ix+(iy)*(nx)) = Vy(ix+(iy)*(nx)) - dt*(P(ix+(iy)*nx)-P(ix+(iy-1)*nx))/dy/rho;
        end
    end
    for ix = 1:nx
        for iyM = 1:ny, iy = iyM-1;
            P(ix+(iy)*nx) = P(ix+(iy)*nx) - dt*(Vx((ix+1)+(iy)*(nx+1))-Vx(ix+(iy)*(nx+1)))/dx*k...
                - dt*(Vy(ix+(iy+1)*(nx))-Vy(ix+(iy)*(nx)))/dy*k;
        end
    end
    % Plot
%     figure(1),clf,scatter(x,y,[20],P),title(it)
%     view(2)
%     axis equal
%     drawnow
end
toc
figure(1),clf,scatter(x,y,[50],P,'filled'),title(it)
view(2)
colorbar
axis equal
save('p_loop_a.mat','P');