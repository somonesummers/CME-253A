clear,figure(1),clf
% numerics
dx = 2;
nx = 50;
Lx = nx*dx;
dxg = 2*Lx/(2*(nx));
nt = 200;
% init vectors
TL = 1*ones(nx,1);
TR = 2*ones(nx,1);
Tg = [TL(1:end-1);TR(2:end)];
nn = 1:size(Tg,1);
% action (small error is from dx not being the same in 2 cases)
for it=1:nt
    TL(2:end-1) = TL(2:end-1) + diff(diff(TL))/dx^2;
    TR(2:end-1) = TR(2:end-1) + diff(diff(TR))/dx^2;
    % update boundaries (MPI)
    TL(end) = TR(2);  TR(1) = TL(end-1);
    % global picture
    T = [TL(1:end-1);TR(2:end)];
    % post-process
    %plot(nn,T,'ro',nn,Tg,'-b'),title(it),drawnow
    % compute physics locally
    Tg(2:end-1) = Tg(2:end-1) + diff(diff(Tg))/dxg^2;
end
max(T-Tg)
