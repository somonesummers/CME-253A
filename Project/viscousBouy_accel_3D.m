%wave_3D_l

clear % Wave 3D loop
% Physics
Lx   = 10;
Ly   = 10;
Lz   = 10;
k    = 1;
rhoi = 10;
eta  = 1;
g    = -10;
% Numerics
OVERLENGTH_X = 1;
OVERLENGTH_Y = 1;
OVERLENGTH_Z = 1;
BLOCK_X = 8;
BLOCK_Y = 8;
BLOCK_Z = 8;
GRID_X  = 4;
GRID_Y  = 4;
GRID_Z  = 4;
nx  = BLOCK_X*GRID_X - OVERLENGTH_X;
ny  = BLOCK_Y*GRID_Y - OVERLENGTH_Y;
nz  = BLOCK_Z*GRID_Z - OVERLENGTH_Z;
nt  = 20000;
dx  = Lx/nx;
dy  = Ly/ny;
dz  = Lz/nz;
dtP = 4.1*eta/ny/4;
dtV = min([dx,dy,dz]).^2/(eta*4.1)/4; 
nu  = 6;
epsi= 1e-6;
% Initial arrays
evol = [];
x    = zeros((nx  )*(ny  )*(nz  ),1);
y    = zeros((nx  )*(ny  )*(nz  ),1);
z    = zeros((nx  )*(ny  )*(nz  ),1);
xe   = zeros((nx+1)*(ny+1)*(nz  ),1);
ye   = zeros((nx+1)*(ny+1)*(nz  ),1);
ze   = zeros((nx+1)*(ny+1)*(nz+1),1);
P    = zeros((nx  )*(ny  )*(nz  ),1);
Txx  = zeros((nx  )*(ny  )*(nz  ),1);
Tyy  = zeros((nx  )*(ny  )*(nz  ),1);
Tzz  = zeros((nx  )*(ny  )*(nz  ),1);
Txy  = zeros((nx+1)*(ny+1)*(nz  ),1);
Txz  = zeros((nx+1)*(ny  )*(nz+1),1);
Tyz  = zeros((nx  )*(ny+1)*(nz+1),1);
Vx   = zeros((nx+1)*(ny  )*(nz  ),1);
Vy   = zeros((nx  )*(ny+1)*(nz  ),1);
Vz   = zeros((nx  )*(ny  )*(nz+1),1);
dVxdt= zeros((nx+1)*(ny  )*(nz  ),1);
dVydt= zeros((nx  )*(ny+1)*(nz  ),1);
dVzdt= zeros((nx  )*(ny  )*(nz+1),1);
Rx   = zeros((nx+1)*(ny  )*(nz  ),1);
Ry   = zeros((nx  )*(ny+1)*(nz  ),1);
Rz   = zeros((nx  )*(ny  )*(nz+1),1);
rad  = zeros((nx  )*(ny  )*(nz  ),1);
rho  = zeros((nx  )*(ny  )*(nz  ),1);
% Initial conditions
for ix = 1:nx+1
    for iyM = 1:ny+1, iy = iyM-1;
        for izM = 1:nz+1, iz = izM-1;
            if(ix <= nx && iyM <= ny && izM <= nz)
                x(ix+(iy)*nx+(iz)*nx*ny) = (ix-1)*dx + (-Lx+dx)/2;
                y(ix+(iy)*nx+(iz)*nx*ny) = (iy  )*dy + (-Ly+dy)/2;
                z(ix+(iy)*nx+(iz)*nx*ny) = (iz  )*dz + (-Lz+dz)/2;
                rad(ix+(iy)*nx+(iz)*nx*ny) = x(ix+(iy)*nx+(iz)*nx*ny)^2 + y(ix+(iy)*nx+(iz)*nx*ny)^2 + z(ix+(iy)*nx+(iz)*nx*ny)^2;
            end
            xe(ix+(iy)*(nx+1)+(iz)*(nx+1)*(ny+1)) = (ix-1)*dx + (-Lx)/2;
            ye(ix+(iy)*(nx+1)+(iz)*(nx+1)*(ny+1)) = (iy)*dy + (-Ly)/2;
            ze(ix+(iy)*(nx+1)+(iz)*(nx+1)*(ny+1)) = (iz)*dy + (-Lz)/2;
        end
    end
end
rho(rad < 1) = rhoi;
% Action
for it = 1:nt
     %Pressue/Txx/Tyy Updates
    for ix = 1:nx
        for iyM = 1:ny, iy = iyM-1;
            for izM = 1:nz, iz = izM-1;
            P(ix+(iy)*nx+(iz)*nx*ny) = P(ix+(iy)*nx+(iz)*nx*ny) - dtP*k*(...
                  (Vx((ix+1)+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  ))-Vx((ix  )+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )))/dx+...
                  (Vy((ix  )+(iy+1)*(nx  )+(iz  )*(nx  )*(ny+1))-Vy((ix  )+(iy  )*(nx  )+(iz  )*(nx  )*(ny+1)))/dy+...
                  (Vz((ix  )+(iy  )*(nx  )+(iz+1)*(nx  )*(ny  ))-Vz((ix  )+(iy  )*(nx  )+(iz  )*(nx  )*(ny  )))/dz);
            Txx(ix+(iy)*nx+(iz)*nx*ny) = 2*eta*(...
                  (Vx((ix+1)+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  ))-Vx((ix  )+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )))/dx - ...
                 ((Vx((ix+1)+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  ))-Vx((ix  )+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )))/dx+...
                  (Vy((ix  )+(iy+1)*(nx  )+(iz  )*(nx  )*(ny+1))-Vy((ix  )+(iy  )*(nx  )+(iz  )*(nx  )*(ny+1)))/dy+...
                  (Vz((ix  )+(iy  )*(nx  )+(iz+1)*(nx  )*(ny  ))-Vz((ix  )+(iy  )*(nx  )+(iz  )*(nx  )*(ny  )))/dz)/3);
            Tyy(ix+(iy)*nx+(iz)*nx*ny) = 2*eta*(...
                  (Vy((ix  )+(iy+1)*(nx  )+(iz  )*(nx  )*(ny+1))-Vy((ix  )+(iy  )*(nx  )+(iz  )*(nx  )*(ny+1)))/dy - ...
                 ((Vx((ix+1)+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  ))-Vx((ix  )+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )))/dx+...
                  (Vy((ix  )+(iy+1)*(nx  )+(iz  )*(nx  )*(ny+1))-Vy((ix  )+(iy  )*(nx  )+(iz  )*(nx  )*(ny+1)))/dy+...
                  (Vz((ix  )+(iy  )*(nx  )+(iz+1)*(nx  )*(ny  ))-Vz((ix  )+(iy  )*(nx  )+(iz  )*(nx  )*(ny  )))/dz)/3);
            Tzz(ix+(iy)*nx+(iz)*nx*ny) = 2*eta*(...
                  (Vz((ix  )+(iy  )*(nx  )+(iz+1)*(nx  )*(ny  ))-Vz((ix  )+(iy  )*(nx  )+(iz  )*(nx  )*(ny  )))/dz - ...
                 ((Vx((ix+1)+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  ))-Vx((ix  )+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )))/dx+...
                  (Vy((ix  )+(iy+1)*(nx  )+(iz  )*(nx  )*(ny+1))-Vy((ix  )+(iy  )*(nx  )+(iz  )*(nx  )*(ny+1)))/dy+...
                  (Vz((ix  )+(iy  )*(nx  )+(iz+1)*(nx  )*(ny  ))-Vz((ix  )+(iy  )*(nx  )+(iz  )*(nx  )*(ny  )))/dz)/3);
            end
        end
    end
    %Sheat Stress Updates (shear stress at all bondaries = 0)
    for ix = 1:nx
        for iyM = 1:ny, iy = iyM-1;
            for izM = 1:nz, iz = izM-1;
                if(ix > 1 && iyM > 1)
                    Txy((ix)+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny+1)) = eta*(...
                       (Vx((ix)+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )) - Vx((ix  )+(iy-1)*(nx+1)+(iz  )*(nx+1)*(ny  )))/dy + ...
                       (Vy((ix)+(iy  )*(nx  )+(iz  )*(nx  )*(ny+1)) - Vy((ix-1)+(iy  )*(nx  )+(iz  )*(nx  )*(ny+1)))/dx);
                end
                if(ix > 1 && izM > 1)
                    Txz((ix)+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )) = eta*(...
                       (Vx((ix)+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )) - Vx((ix  )+(iy  )*(nx+1)+(iz-1)*(nx+1)*(ny  )))/dz + ...
                       (Vz((ix)+(iy  )*(nx  )+(iz  )*(nx  )*(ny  )) - Vz((ix-1)+(iy  )*(nx  )+(iz  )*(nx  )*(ny  )))/dx);
                end
                if(iyM > 1 && izM > 1)
                    Tyz((ix)+(iy  )*(nx  )+(iz  )*(nx  )*(ny+1)) = eta*(...
                       (Vy((ix)+(iy  )*(nx  )+(iz  )*(nx  )*(ny+1)) - Vy((ix)+(iy  )*(nx  )+(iz-1)*(nx  )*(ny+1)))/dz + ...
                       (Vz((ix)+(iy  )*(nx  )+(iz  )*(nx  )*(ny  )) - Vz((ix)+(iy-1)*(nx  )+(iz  )*(nx  )*(ny  )))/dy);
                end
            end
        end
    end
    %Velocity Updates
    for ix = 1:nx
        for iyM = 1:ny, iy = iyM-1;
             for izM = 1:nz, iz = izM-1;
                 if ix > 1
                    Rx(ix+(iy)*(nx+1)+(iz)*(nx+1)*ny) = 1 * (...
                        -1*(P( ix +(iy  )* nx   +(iz  )* nx   * ny   ) -   P((ix-1)+(iy  )* nx   +(iz  )* nx   * ny   ))/dx...             
                       + (Txx( ix +(iy  )* nx   +(iz  )* nx   * ny   ) - Txx((ix-1)+(iy  )* nx   +(iz  )* nx   * ny   ))/dx...        
                       + (Txy((ix)+(iy+1)*(nx+1)+(iz  )*(nx+1)*(ny+1)) - Txy((ix  )+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny+1)))/dy...
                       + (Txz((ix)+(iy  )*(nx+1)+(iz+1)*(nx+1)*(ny  )) - Txz((ix  )+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )))/dz);
                    dVxdt(ix+(iy)*(nx+1)+(iz)*(nx+1)*ny) = (1-nu/nx)*dVxdt(ix+(iy)*(nx+1)+(iz)*(nx+1)*ny) + Rx(ix+(iy)*(nx+1)+(iz)*(nx+1)*ny);
                    Vx(ix+(iy)*(nx+1)+(iz)*(nx+1)*ny) = Vx(ix+(iy)*(nx+1)+(iz)*(nx+1)*ny) + dtV*dVxdt(ix+(iy)*(nx+1)+(iz)*(nx+1)*ny);
                 end
                 if iyM > 1
                    Ry(ix+(iy)*(nx  )+(iz)*(nx  )*(ny+1)) = 1 * (...
                        -1*(P((ix  )+(iy  )* nx   +(iz  )* nx   * ny   ) -   P((ix  )+(iy-1)* nx   +(iz  )* nx   * ny   ))/dy...             
                       + (Tyy((ix  )+(iy  )* nx   +(iz  )* nx   * ny   ) - Tyy((ix  )+(iy-1)* nx   +(iz  )* nx   * ny   ))/dy...        
                       + (Txy((ix+1)+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny+1)) - Txy((ix  )+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny+1)))/dx...
                       + (Tyz((ix  )+(iy  )*(nx  )+(iz+1)*(nx  )*(ny+1)) - Tyz((ix  )+(iy  )*(nx  )+(iz  )*(nx  )*(ny+1)))/dz...
                  + .5*g*(rho((ix  )+(iy  )* nx   +(iz  )* nx   * ny   ) + rho((ix  )+(iy-1)* nx   +(iz  )* nx   * ny   )));
                    dVydt(ix+(iy)*(nx  )+(iz)*(nx  )*(ny+1)) = (1-nu/ny)*dVydt(ix+(iy)*(nx  )+(iz)*(nx  )*(ny+1)) + Ry(ix+(iy)*(nx  )+(iz)*(nx  )*(ny+1));
                    Vy(ix+(iy)*(nx  )+(iz)*(nx  )*(ny+1)) = Vy(ix+(iy)*(nx  )+(iz)*(nx  )*(ny+1)) + dtV*dVydt(ix+(iy)*(nx  )+(iz)*(nx  )*(ny+1));
                 end
                 if izM > 1
                    Rz(ix+(iy)*(nx  )+(iz)*(nx  )*(ny  )) = 1 * (...
                        -1*(P((ix  )+(iy  )* nx   +(iz  )* nx   * ny   ) -   P((ix  )+(iy  )* nx   +(iz-1)* nx   * ny   ))/dz...             
                       + (Tzz((ix  )+(iy  )* nx   +(iz  )* nx   * ny   ) - Tzz((ix  )+(iy  )* nx   +(iz-1)* nx   * ny   ))/dz...        
                       + (Txz((ix+1)+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )) - Txz((ix  )+(iy  )*(nx+1)+(iz  )*(nx+1)*(ny  )))/dx...
                       + (Tyz((ix  )+(iy+1)*(nx  )+(iz  )*(nx  )*(ny+1)) - Tyz((ix  )+(iy  )*(nx  )+(iz  )*(nx  )*(ny+1)))/dy);
                    dVzdt(ix+(iy)*(nx  )+(iz)*(nx  )*(ny  )) = (1-nu/nz)*dVzdt(ix+(iy)*(nx  )+(iz)*(nx  )*(ny  )) + Rz(ix+(iy)*(nx  )+(iz)*(nx  )*(ny  ));
                    Vz(ix+(iy)*(nx  )+(iz)*(nx  )*(ny  )) = Vz(ix+(iy)*(nx  )+(iz)*(nx  )*(ny  )) + dtV*dVzdt(ix+(iy)*(nx  )+(iz)*(nx  )*(ny  ));
                 end
             end
        end
    end
    err = max([max(abs(Rx(:))), max(abs(Ry(:))),max(abs(Rz(:)))]);
    evol = [evol, err]; 
    if err<epsi, break; end
    if(mod(it,50)==0)
        %% Plot
        figure(2)
        clf
        subplot(221)
        semilogy(evol);

        subplot(222)
        temp1 = reshape(P,nx,ny,nz);
        imagesc(x(1:nx),y(1:nx:nx*ny),squeeze(temp1(:,:,(nz+1)/2))'),title("Pr")
        xlabel('x')
        ylabel('y')
        set(gca,'YDir','normal')
        colorbar
        
        subplot(223)
        temp2 = reshape(Vy,nx,ny+1,nz);
        imagesc(x(1:nx),ye(1:nx+1:(nx+1)*(ny+1)),squeeze(temp2(:,:,(nz+1)/2))'),title("Vy")
        set(gca,'YDir','normal')
        xlabel('x')
        ylabel('y')
        axis equal
        colorbar

        subplot(224)
        temp3 = reshape(Vx,nx+1,ny,nz);
        imagesc(x(1:nx),ye(1:nx:(nx+1)*(ny+1)),squeeze(temp3(:,:,(nz+1)/2))'),title("Vx")
        set(gca,'YDir','normal')
        axis equal
        colorbar
        drawnow
     end
end
it
%% 
figure(3)
Vyp = reshape(Vy,nx,ny+1,nz);
Vxp = reshape(Vx,nx+1,ny,nz);
Vxp = squeeze(Vxp(:,:,(nz+1)/2));
Vyp = squeeze(Vyp(:,:,(nz+1)/2));
Vxp = .5*(Vxp(1:end-1,:) + Vxp(2:end,:));
Vyp = .5*(Vyp(:,1:end-1) + Vyp(:,2:end));
quiver(Vxp,Vyp)
%%
save('p_l_v.mat','P','Vx','Vy','Vz','Txx','Tyy','Tzz','Txy','Txz','Tyz','evol');