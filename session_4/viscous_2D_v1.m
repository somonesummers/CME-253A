clear 
% Physics
Lx   = 10;
Ly   = 10;
% k    = 1e-1;
rhog = 10;
eta  = 1;
% Numerics
nx   = 100;
ny   = 101;
dx   = Lx/nx;
dy   = Ly/ny;
nt   = 10000;
nu   = 6;
epsi = 1e-6;
% dtP  = min(dx,dy)/k/2.5;
dtP  = 4.1*eta/ny/10;
dtV  = min(dx,dy).^2/eta/4.1;
x    = (-Lx+dx)/2:dx:(Lx-dx)/2;
y    = (-Ly+dy)/2:dy:(Ly-dy)/2;
[X2,Y2] = ndgrid(x,y);
% Initial conditions
Pr    = zeros(nx  ,ny  );
Vx    = zeros(nx+1,ny  );
Vy    = zeros(nx  ,ny+1);
dVxdt = zeros(nx-1,ny  );
dVydt = zeros(nx  ,ny-1);
txx   = zeros(nx  ,ny  );
tyy   = zeros(nx  ,ny  );
txy   = zeros(nx+1,ny+1);
Rog   = ones(nx  ,ny  );
rad   = X2.^2 + Y2.^2;
Rog(rad<1) = rhog;
% Action
evol = [];
for it = 1:nt
    divV          = diff(Vx,1,1)/dx + diff(Vy,1,2)/dy;
    Pr            = Pr - dtP*divV;
    txx           = 2*eta*(diff(Vx,1,1)/dx - 1/3*divV);
    tyy           = 2*eta*(diff(Vy,1,2)/dy - 1/3*divV);
    txy(2:end-1,2:end-1) = eta*(diff(Vx(2:end-1,:),1,2)/dy + diff(Vy(:,2:end-1),1,1)/dx);
    Rx            = -diff(Pr,1,1)/dx + diff(txx,1,1)/dx + diff(txy(2:end-1,:),1,2)/dy;
    %Ry            = -diff(Pr,1,2)/dy + diff(tyy,1,2)/dy + diff(txy(:,2:end-1),1,1)/dx + 0.5*(Rog(:,1:end-1)+Rog(:,2:end));
    Ry            = -diff(Pr,1,2)/dy + diff(tyy,1,2)/dy + diff(txy(:,2:end-1),1,1)/dx - 0.5*(Rog(:,1:end-1)+Rog(:,2:end));
    dVxdt         = dVxdt*(1-nu/nx) + Rx;
    dVydt         = dVydt*(1-nu/ny) + Ry;
    Vx(2:end-1,:) = Vx(2:end-1,:) + dtV*dVxdt;
    Vy(:,2:end-1) = Vy(:,2:end-1) + dtV*dVydt;
    err = max([max(abs(Rx(:))), max(abs(Ry(:))), max(abs(divV(:)))]); evol = [evol, err]; if err<epsi, break; end
    if mod(it,200)==0
        % Plot
        figure(1),clf
        subplot(221),semilogy(evol,'-d')
        subplot(222),imagesc(x,y,Pr'),title('Pr'),axis xy,axis image,colorbar
        subplot(223),imagesc(x,y,Vx'),title('Vx'),axis xy,axis image,colorbar
        subplot(224),imagesc(x,y,Vy'),title('Vy'),axis xy,axis image,colorbar
        drawnow
    end
end
