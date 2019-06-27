%wave_2D_l

clear % Wave 2D loop
% Physics
Lx  = 10;
Ly  = 4;
k   = 1;
rho = 1;
mu   = 1;
% Numerics
nx  = 100;
ny  = 100;
dx  = Lx/nx;
dy  = Ly/ny;
T = 3;
dt  = min([dx,dy])^2/mu/4.1/3/4; 
nt = round(T/dt);
plot_step = round(.1/dt);
% Initial arrays
x    = zeros((nx  )*(ny  ),1);
y    = zeros((nx  )*(ny  ),1);
xe   = zeros((nx+1)*(ny+1),1);
ye   = zeros((nx+1)*(ny+1),1);
P    = zeros((nx  )*(ny  ),1);
Pini = zeros((nx  )*(ny  ),1);
Txx  = zeros((nx  )*(ny  ),1);
Tyy  = zeros((nx  )*(ny  ),1);
Txyc = zeros((nx  )*(ny  ),1);
Txy  = zeros((nx+1)*(ny+1),1);
Vx   = zeros((nx+1)*(ny  ),1);
Vy   = zeros((nx  )*(ny+1),1);
% Initial conditions
for ix = 1:nx+1
    for iyM = 1:ny+1, iy = iyM-1;
        if(iyM <= ny && ix <= nx)
            x(ix+(iy)*nx) = (ix-1)*dx + (-Lx+dx)/2;
            y(ix+(iy)*nx) = (iy)*dy + (-Ly+dy)/2;
            P(ix+(iy)*nx) = exp(-(x(ix+(iy)*nx)^2+y(ix+(iy)*nx)^2));
            Pini(ix+(iy)*nx)=P(ix+(iy)*nx);
        end
        xe(ix+(iy)*(nx+1)) = (ix-1)*dx + (-Lx)/2;
        ye(ix+(iy)*(nx+1)) = (iy)*dy + (-Ly)/2;
    end
end
% Action
for it = 1:nt
    %Velocity Updates
    for ix = 2:nx
        for iyM = 1:ny, iy = iyM-1;
            Vx(ix+(iy)*(nx+1)) = Vx(ix+(iy)*(nx+1)) + dt/rho*(...
                -1*(P(ix+(iy)*nx)-P((ix-1)+(iy)*nx))/dx...             %dP/dx
                + (Txx(ix+(iy)*nx) - Txx((ix-1)+(iy)*nx))/dx...        %dTxx/dx
                + (Txy(ix+(iy+1)*(nx+1)) - Txy(ix+(iy)*(nx+1)))/dy);       %dTxy/dy
        end
    end
    for ix = 1:nx
        for iyM = 2:ny, iy = iyM-1;
            Vy(ix+(iy)*(nx)) = Vy(ix+(iy)*(nx)) + dt/rho*(...
                -1*(P(ix+(iy)*nx)-P(ix+(iy-1)*nx))/dy...                 %dP/dy
                + (Tyy(ix+(iy)*nx) - Tyy(ix+(iy-1)*nx))/dy...            %dTyy/dy
                + (Txy((ix+1)+(iy)*(nx+1)) - Txy(ix+(iy)*(nx+1)))/dx); %dTxy/dx
        end
    end
    %Pressue/Txx/Tyy Updates
    for ix = 1:nx
        for iyM = 1:ny, iy = iyM-1;
            div = (Vx((ix+1)+(iy)*(nx+1))-Vx(ix+(iy)*(nx+1)))/dx+...
                  (Vy(ix+(iy+1)*(nx))-Vy(ix+(iy)*(nx)))/dy;
            P(ix+(iy)*nx) = P(ix+(iy)*nx) - dt*k*(div);
            Txx(ix+(iy)*nx) = 2*mu*(...
                    (Vx((ix+1)+(iy)*(nx+1))-Vx(ix+(iy)*(nx+1)))/dx - ...
                    1/3*(div));
            Tyy(ix+(iy)*nx) = 2*mu*(...
                    (Vy(ix+(iy+1)*(nx))-Vy(ix+(iy)*(nx)))/dy - ...
                    1/3*(div));
        end
    end
    %Txy Updates (shear stress at all bondaries = 0)
    for ix = 2:nx
        for iyM = 2:ny, iy = iyM-1;
            Txy(ix+(iy)*(nx+1)) = mu*(...
                   (Vx(ix+(iy)*(nx+1)) - Vx(ix+(iy-1)*(nx+1)))/dy + ...
                   (Vy(ix+(iy)*(nx)) - Vy((ix-1)+(iy)*(nx)))/dx);
        end
    end
    ttime = round(dt*it,2);
    if(mod(it,plot_step)==0)
        % Plot
        figure(2)
        subplot(2,2,1)
        scatter(x,y,[20],P),title("Pressure " + ttime)
        axis equal
        caxis([-.1 1])
        colorbar

        subplot(2,2,2)
        scatter(x,y,[20],Txx),title("Txx")
        axis equal
        colorbar

        subplot(2,2,3)
        scatter(x,y,[20],Tyy),title("Tyy")
        axis equal
        colorbar

        subplot(2,2,4)
        scatter(xe,ye,[20],Txy),title("Txy")
        axis equal
        colorbar

        drawnow
    end
end
save('p_l_v.mat','P');