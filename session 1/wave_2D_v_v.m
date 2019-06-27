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
x   = (-Lx+dx)/2:dx:(Lx-dx)/2;
y   = (-Ly+dy)/2:dy:(Ly-dy)/2;
xe  = -Lx/2:dx:Lx/2;
ye  = -Ly/2:dy:Ly/2;
[x2,y2] = ndgrid(x,y);
[x2e,y2e] = ndgrid(xe,ye);
% Initial Conditions
P    = exp(-(x2.^2+y2.^2));
Pini = P;
Vx   = zeros(nx+1,ny  );
Vy   = zeros(nx  ,ny+1);
Txx  = zeros((nx  ),(ny  ));
Tyy  = zeros((nx  ),(ny  ));
Txyc = zeros((nx  ),(ny  ));
Txy  = zeros((nx+1),(ny+1));
%Action
for it = 1:nt
    %Pressure
    div = diff(Vx,1,1)/dx + diff(Vy,1,2)/dy;
    P = P - dt*(div)*k;
    %Taus
    Txx = 2*mu*(diff(Vx,1,1)/dx - div/3);
    Tyy = 2*mu*(diff(Vy,1,2)/dy - div/3);
    %extend Velocities with BCs
    Vxe = [Vx(:,1),Vx,Vx(:,end)];
    Vye = [Vy(1,:);Vy;Vy(end,:)];
    Txy = mu*(diff(Vxe,1,2)/dy + diff(Vye,1,1)/dx);
    Txyc = (Txy(1:end-1,1:end-1) + Txy(1:end-1,2:end) + ...
            Txy(2:end,1:end-1) + Txy(2:end,2:end))/4;
    %Velocities
    Vx(2:end-1,:) = Vx(2:end-1,:) + dt/rho*(...
        -1*diff(P,1,1)/dx +...
        diff(Txx,1,1)/dx + ...
        diff(Txy(2:end-1,:),1,2)/dy);
    Vy(:,2:end-1) = Vy(:,2:end-1) + dt/rho*(...
        -1*diff(P,1,2)/dy +...
        diff(Tyy,1,2)/dy + ...
        diff(Txy(:,2:end-1),1,1)/dx);
    
    ttime = round(dt*it,2);
    % Plot
    if(mod(it,plot_step)==0)
        figure(1),clf
        subplot(2,2,1)
        imagesc(x,y,P'),title("Pressure " + ttime)
        axis equal
        caxis([-.1 1])
        colorbar

        subplot(2,2,2)
        imagesc(x,y,Txx'),title("Txx")
        axis equal
        %caxis([-.1 .1])
        colorbar

        subplot(2,2,3)
        imagesc(x,y,Tyy'),title("Tyy")
        axis equal
        %caxis([-.1 .1])
        colorbar

        subplot(2,2,4)
        imagesc(xe,ye,Txy'),title("Txy")
        axis equal
        %caxis([-.1 .1])
        colorbar

        drawnow
        
    end
end
save('p_v_v.mat','P');