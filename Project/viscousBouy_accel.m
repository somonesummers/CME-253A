%wave_2D_l

clear % Wave 2D loop
% Physics
Lx   = 10;
Ly   = 10;
k    = 1;
rhoi = 10;
eta  = 1;
g    = -10;
% Numerics
OVERLENGTH_X = 1;
OVERLENGTH_Y = 1;
BLOCK_X = 16;
BLOCK_Y = 16;
GRID_X  = 4;
GRID_Y  = 4;
nx  = BLOCK_X*GRID_X - OVERLENGTH_X;
ny  = BLOCK_Y*GRID_Y - OVERLENGTH_Y;
dx  = Lx/nx;
dy  = Ly/ny;
dtP = 4.1*eta/ny/10;
dtV = min([dx,dy]).^2/(eta*4.1); 
nt  = 10000;
plot_step = 10;
nu  = 6;
epsi= 1e-8;
% Initial arrays
evol = [];
x    = zeros((nx  )*(ny  ),1);
y    = zeros((nx  )*(ny  ),1);
xe   = zeros((nx+1)*(ny+1),1);
ye   = zeros((nx+1)*(ny+1),1);
P    = zeros((nx  )*(ny  ),1);
Txx  = zeros((nx  )*(ny  ),1);
Tyy  = zeros((nx  )*(ny  ),1);
Txyc = zeros((nx  )*(ny  ),1);
Txy  = zeros((nx+1)*(ny+1),1);
Vx   = zeros((nx+1)*(ny  ),1);
Vy   = zeros((nx  )*(ny+1),1);
dVxdt= zeros((nx+1)*(ny  ),1);
dVydt= zeros((nx  )*(ny+1),1);
Rx   = zeros((nx+1)*(ny  ),1);
Ry   = zeros((nx  )*(ny+1),1);
rad  = zeros((nx  )*(ny  ),1);
rho  =  ones((nx  )*(ny  ),1);
% Initial conditions
for ix = 1:nx+1
    for iyM = 1:ny+1, iy = iyM-1;
        if(iyM <= ny && ix <= nx)
            x(ix+(iy)*nx) = (ix-1)*dx + (-Lx+dx)/2;
            y(ix+(iy)*nx) = (iy)*dy + (-Ly+dy)/2;
            rad(ix+(iy)*nx) = x(ix+(iy)*nx)^2 + y(ix+(iy)*nx)^2;
        end
        xe(ix+(iy)*(nx+1)) = (ix-1)*dx + (-Lx)/2;
        ye(ix+(iy)*(nx+1)) = (iy)*dy + (-Ly)/2;
    end
end
rho(rad < 1) = rhoi;
% Action
for it = 1:nt
     %Pressue/Txx/Tyy Updates
    for ix = 1:nx
        for iyM = 1:ny, iy = iyM-1;
            P(ix+(iy)*nx) = P(ix+(iy)*nx) - dtP*k*(...
                  (Vx((ix+1)+(iy  )*(nx+1))-Vx(ix+(iy)*(nx+1)))/dx+...
                  (Vy( ix   +(iy+1)*(nx  ))-Vy(ix+(iy)*(nx  )))/dy);
            Txx(ix+(iy)*nx) = 2*eta*(...
                     (Vx((ix+1)+(iy  )*(nx+1))-Vx(ix+(iy)*(nx+1)))/dx - ...
                1/3*((Vx((ix+1)+(iy  )*(nx+1))-Vx(ix+(iy)*(nx+1)))/dx +...
                     (Vy( ix   +(iy+1)*(nx  ))-Vy(ix+(iy)*(nx  )))/dy));
            Tyy(ix+(iy)*nx) = 2*eta*(...
                    (Vy( ix   +(iy+1)*(nx  ))-Vy(ix+(iy)*(nx  )))/dy - ...
               1/3*((Vx((ix+1)+(iy  )*(nx+1))-Vx(ix+(iy)*(nx+1)))/dx+...
                    (Vy( ix   +(iy+1)*(nx  ))-Vy(ix+(iy)*(nx  )))/dy));
        end
    end
    %Txy Updates (shear stress at all bondaries = 0)
    for ix = 2:nx
        for iyM = 2:ny, iy = iyM-1;
            Txy(ix+(iy)*(nx+1)) = eta*(...
                   (Vx(ix+(iy)*(nx+1)) - Vx(ix+(iy-1)*(nx+1)))/dy + ...
                   (Vy(ix+(iy)*(nx)) - Vy((ix-1)+(iy)*(nx)))/dx);
        end
    end
    %Velocity Updates
    for ix = 2:nx
        for iyM = 1:ny, iy = iyM-1;
            Rx(ix+(iy)*(nx+1)) = 1 * (...
                -1*(P(ix+(iy)*nx)-P((ix-1)+(iy)*nx))/dx...             
                + (Txx(ix+(iy)*nx) - Txx((ix-1)+(iy)*nx))/dx...        
                + (Txy(ix+(iy+1)*(nx+1)) - Txy(ix+(iy)*(nx+1)))/dy);
            dVxdt(ix+(iy)*(nx+1)) = (1-nu/nx)*dVxdt(ix+(iy)*(nx+1)) + Rx(ix+(iy)*(nx+1));
            Vx(ix+(iy)*(nx+1)) = Vx(ix+(iy)*(nx+1)) + dtV*(dVxdt(ix+(iy)*(nx+1)));
                      
        end
    end
    for ix = 1:nx
        for iyM = 2:ny, iy = iyM-1;
            Ry(ix+(iy)*(nx)) = 1 * (...
                -1*(P(ix+(iy)*nx)-P(ix+(iy-1)*nx))/dy...                 %dP/dy
                + (Tyy(ix+(iy)*nx) - Tyy(ix+(iy-1)*nx))/dy...            %dTyy/dy
                + (Txy((ix+1)+(iy)*(nx+1)) - Txy(ix+(iy)*(nx+1)))/dx...  %dTxy/dx
                + .5*g*(rho(ix+(iy-1)*(nx)) + rho(ix+(iy)*(nx))));       %Gravity
            dVydt(ix+(iy)*(nx)) = (1-nu/ny)*dVydt(ix+(iy)*(nx)) + Ry(ix+(iy)*(nx));
            Vy(ix+(iy)*(nx)) = Vy(ix+(iy)*(nx)) + dtV*dVydt(ix+(iy)*(nx));    
        end
    end
    err = max([max(abs(Rx(:))), max(abs(Ry(:)))]);
    evol = [evol, err]; 
    if err<epsi, break; end
%     if(mod(it,plot_step)==0)
%         %Plot
%         figure(2)
%         subplot(221)
%         semilogy(evol);
%         
%         subplot(222)
%         imagesc(x(1:nx),y(1:nx:end),flipud(reshape(P,nx,ny)')),title("Pressure " + it)
%         axis equal
%         colorbar
% 
%         subplot(223)
%         imagesc(xe(1:nx+1),y(1:nx:end),flipud(reshape(Vx,nx+1,ny)')),title("Vx")
%         axis equal
%         colorbar
% 
%         subplot(224)
%         imagesc(x(1:nx),ye(1:(nx+1):end),flipud(reshape(Vy,nx,ny+1)')),title("Vy")
%         axis equal
%         colorbar
%         drawnow
%     end
end
%%
figure(1)
subplot(221)
semilogy(evol);

subplot(222)
imagesc(x(1:nx),y(1:nx:end),flipud(reshape(P,nx,ny)')),title("Pressure " + it)
axis equal
colorbar

subplot(223)
imagesc(xe(1:nx+1),y(1:nx:end),flipud(reshape(Vx,nx+1,ny)')),title("Vx")
axis equal
colorbar

subplot(224)
imagesc(x(1:nx),ye(1:(nx+1):end),flipud(reshape(Vy,nx,ny+1)')),title("Vy")
axis equal
colorbar

save('p_l_v.mat','P','Vx','Vy','Txx','Tyy','Txy');