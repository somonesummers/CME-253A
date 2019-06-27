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
nt = round(T/dt)
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
    for iy = 1:ny+1
        if(iy <= ny && ix <= nx)
            x(ix+(iy-1)*nx) = (ix-1)*dx + (-Lx+dx)/2;
            y(ix+(iy-1)*nx) = (iy-1)*dy + (-Ly+dy)/2;
            P(ix+(iy-1)*nx) = exp(-(x(ix+(iy-1)*nx)^2+y(ix+(iy-1)*nx)^2));
            Pini(ix+(iy-1)*nx)=P(ix+(iy-1)*nx);
        end
        xe(ix+(iy-1)*(nx+1)) = (ix-1)*dx + (-Lx)/2;
        ye(ix+(iy-1)*(nx+1)) = (iy-1)*dy + (-Ly)/2;
    end
end
% Action
for it = 1:nt
    %Velocity Updates
    for ix = 2:nx
        for iy = 1:ny
            Vx(ix+(iy-1)*(nx+1)) = Vx(ix+(iy-1)*(nx+1)) + dt/rho*(...
                -1*(P(ix+(iy-1)*nx)-P((ix-1)+(iy-1)*nx))/dx...             %dP/dx
                + (Txx(ix+(iy-1)*nx) - Txx((ix-1)+(iy-1)*nx))/dx...        %dTxx/dx
                + (Txy(ix+(iy)*(nx+1)) - Txy(ix+(iy-1)*(nx+1)))/dy);       %dTxy/dy
        end
    end
    for ix = 1:nx
        for iy = 2:ny
            Vy(ix+(iy-1)*(nx)) = Vy(ix+(iy-1)*(nx)) + dt/rho*(...
                -1*(P(ix+(iy-1)*nx)-P(ix+(iy-2)*nx))/dy...                 %dP/dy
                + (Tyy(ix+(iy-1)*nx) - Tyy(ix+(iy-2)*nx))/dy...            %dTyy/dy
                + (Txy((ix+1)+(iy-1)*(nx+1)) - Txy(ix+(iy-1)*(nx+1)))/dx); %dTxy/dx
        end
    end
    %Pressue/Txx/Tyy Updates
    for ix = 1:nx
        for iy = 1:ny
            div = (Vx((ix+1)+(iy-1)*(nx+1))-Vx(ix+(iy-1)*(nx+1)))/dx+...
                  (Vy(ix+(iy)*(nx))-Vy(ix+(iy-1)*(nx)))/dy;
            P(ix+(iy-1)*nx) = P(ix+(iy-1)*nx) - dt*k*(div);
            Txx(ix+(iy-1)*nx) = 2*mu*(...
                    (Vx((ix+1)+(iy-1)*(nx+1))-Vx(ix+(iy-1)*(nx+1)))/dx - ...
                    1/3*(div));
            Tyy(ix+(iy-1)*nx) = 2*mu*(...
                    (Vy(ix+(iy)*(nx))-Vy(ix+(iy-1)*(nx)))/dy - ...
                    1/3*(div));
        end
    end
    %Txy Updates (shear stress at all bondaries = 0)
    for ix = 2:nx
        for iy = 2:ny
            Txy(ix+(iy-1)*(nx+1)) = mu*(...
                   (Vx(ix+(iy-1)*(nx+1)) - Vx(ix+(iy-2)*(nx+1)))/dy + ...
                   (Vy(ix+(iy-1)*(nx)) - Vy((ix-1)+(iy-1)*(nx)))/dx);
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