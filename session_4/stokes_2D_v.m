clear % Stokes 2D vectorised
% Physics
Lx   = 10;
Ly   = 10;
rhog = 1;
eta  = 1;
% Numerics
nx   = 100;
ny   = 101;
nout = 500;
nt   = 10000;
epsi = 1e-6;
Vdmp = 6;
% Preprocessing
dx   = Lx/nx;
dy   = Ly/ny;
x    = (-Lx+dx)/2:dx:(Lx-dx)/2;
y    = (-Ly+dy)/2:dy:(Ly-dy)/2;
[X2,Y2] = ndgrid(x,y);
% Initial conditions
Pr    = zeros(nx  ,ny  );
Vx    = zeros(nx+1,ny  );
Vy    = zeros(nx  ,ny+1);
txx   = zeros(nx  ,ny  );
tyy   = zeros(nx  ,ny  );
txy   = zeros(nx+1,ny+1);
dVxdt = zeros(nx-1,ny  );
dVydt = zeros(nx  ,ny-1);
Rog   = zeros(nx  ,ny  );
rad   = X2.^2+Y2.^2;
Rog(rad<1) = rhog;
evol  = [];
% Action
for it = 1:nt
    dtPr   = 4.1*eta/ny;
    dtV    = min(dx,dy)^2/eta/4.1/10;
    divV   = diff(Vx,1,1)/dx + diff(Vy,1,2)/dy;
    Pr     = Pr - dtPr*divV;
    txx    = 2*eta*(diff(Vx,1,1)/dx - 1/3*divV);
    tyy    = 2*eta*(diff(Vy,1,2)/dy - 1/3*divV);
    txy(2:end-1,2:end-1) = eta*(diff(Vx(2:end-1,:),1,2)/dy + diff(Vy(:,2:end-1),1,1)/dx);
    Rx     = -diff(Pr,1,1)/dx + diff(txx,1,1)/dx + diff(txy(2:end-1,:),1,2)/dy;
    Ry     = -diff(Pr,1,2)/dy + diff(tyy,1,2)/dy + diff(txy(:,2:end-1),1,1)/dx + 0.5*(Rog(:,1:end-1)+Rog(:,2:end));
    dVxdt  = (1-Vdmp/nx)*dVxdt + Rx;
    dVydt  = (1-Vdmp/ny)*dVydt + Ry;
    Vx(2:end-1,:) = Vx(2:end-1,:) + dtV*dVxdt;
    Vy(:,2:end-1) = Vy(:,2:end-1) + dtV*dVydt;
    err = max(abs([divV(:); Rx(:); Ry(:)])) ; if err<epsi, break; end; evol=[evol;err];
    % Plot
    if mod(it,nout)==0
        figure(1),clf
        subplot(311),imagesc(Pr'),title('Pr'),axis image;axis xy;colorbar
        subplot(312),imagesc(Vx'),title('Vx'),axis image;axis xy;colorbar
        subplot(313),imagesc(Vy'),title('Vy'),axis image;axis xy;colorbar
        drawnow
    end
end
figure(2),clf,semilogy(evol),title(it)
