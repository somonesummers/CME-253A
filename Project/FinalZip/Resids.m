clear
% Load the DATA and infos
cd d32
nxy = load('0_nxyz.inf');  PRECIS=nxy(1); nx=nxy(2); ny=nxy(3); nz=nxy(4);
if (PRECIS==8), DAT = 'double';  elseif (PRECIS==4), DAT = 'single';  end 
id = fopen('0_P.res' ); P  = fread(id,DAT); fclose(id); Pfs  = reshape(P ,nx  ,ny  ,nz  );
id = fopen('0_Vx.res'); Vx = fread(id,DAT); fclose(id); Vxfs = reshape(Vx,nx+1,ny  ,nz  );
id = fopen('0_Vy.res'); Vy = fread(id,DAT); fclose(id); Vyfs = reshape(Vy,nx  ,ny+1,nz  );
id = fopen('0_Vz.res'); Vz = fread(id,DAT); fclose(id); Vzfs = reshape(Vz,nx  ,ny  ,nz+1);
id = fopen('0_Rx.res'); Rx = fread(id,DAT); fclose(id); Rxfs = reshape(Vx,nx+1,ny  ,nz  );
id = fopen('0_Ry.res'); Ry = fread(id,DAT); fclose(id); Ryfs = reshape(Vy,nx  ,ny+1,nz  );
id = fopen('0_Rz.res'); Rz = fread(id,DAT); fclose(id); Rzfs = reshape(Vz,nx  ,ny  ,nz+1);
id = fopen('0_Txx.res'); Txx = fread(id,DAT); fclose(id); Txxfs = reshape(Txx,nx  ,ny  ,nz  );
id = fopen('0_Tyy.res'); Tyy = fread(id,DAT); fclose(id); Tyyfs = reshape(Tyy,nx  ,ny  ,nz  );
id = fopen('0_Tzz.res'); Tzz = fread(id,DAT); fclose(id); Tzzfs = reshape(Tyy,nx  ,ny  ,nz  );
id = fopen('0_Txy.res'); Txy = fread(id,DAT); fclose(id); Txyfs = reshape(Txy,nx+1,ny+1,nz  );
id = fopen('0_Txz.res'); Txz = fread(id,DAT); fclose(id); Txzfs = reshape(Txz,nx+1,ny  ,nz+1);
id = fopen('0_Tyz.res'); Tyz = fread(id,DAT); fclose(id); Tyzfs = reshape(Tyz,nx  ,ny+1,nz+1);
cd ../d24
nxy = load('0_nxyz.inf');  PRECIS=nxy(1); nx=nxy(2); ny=nxy(3); nz=nxy(4);
if (PRECIS==8), DAT = 'double';  elseif (PRECIS==4), DAT = 'single';  end 
id = fopen('0_P.res' ); P  = fread(id,DAT); fclose(id); Ps  = reshape(P ,nx  ,ny  ,nz  );
id = fopen('0_Vx.res'); Vx = fread(id,DAT); fclose(id); Vxs = reshape(Vx,nx+1,ny  ,nz  );
id = fopen('0_Vy.res'); Vy = fread(id,DAT); fclose(id); Vys = reshape(Vy,nx  ,ny+1,nz  );
id = fopen('0_Vz.res'); Vz = fread(id,DAT); fclose(id); Vzs = reshape(Vz,nx  ,ny  ,nz+1);
id = fopen('0_Rx.res'); Rx = fread(id,DAT); fclose(id); Rxs = reshape(Vx,nx+1,ny  ,nz  );
id = fopen('0_Ry.res'); Ry = fread(id,DAT); fclose(id); Rys = reshape(Vy,nx  ,ny+1,nz  );
id = fopen('0_Rz.res'); Rz = fread(id,DAT); fclose(id); Rzs = reshape(Vz,nx  ,ny  ,nz+1);
id = fopen('0_Txx.res'); Txx = fread(id,DAT); fclose(id); Txxs = reshape(Txx,nx  ,ny  ,nz  );
id = fopen('0_Tyy.res'); Tyy = fread(id,DAT); fclose(id); Tyys = reshape(Tyy,nx  ,ny  ,nz  );
id = fopen('0_Tzz.res'); Tzz = fread(id,DAT); fclose(id); Tzzs = reshape(Tyy,nx  ,ny  ,nz  );
id = fopen('0_Txy.res'); Txy = fread(id,DAT); fclose(id); Txys = reshape(Txy,nx+1,ny+1,nz  );
id = fopen('0_Txz.res'); Txz = fread(id,DAT); fclose(id); Txzs = reshape(Txz,nx+1,ny  ,nz+1);
id = fopen('0_Tyz.res'); Tyz = fread(id,DAT); fclose(id); Tyzs = reshape(Tyz,nx  ,ny+1,nz+1);
cd ..
%%
ss = 191;
Pdn = imresize3(Pfs,ss/255);
Vydn = imresize3(Vyfs,ss/255);
Vxdn = imresize3(Vxfs,ss/255);
rP = norm(squeeze(reshape(Pdn-Ps,ss^3,1,1)),2);
rVy = norm(squeeze(reshape(Vydn-Vys,ss^2*(ss+1),1,1)),2);
rVx = norm(squeeze(reshape(Vxdn-Vxs,ss^2*(ss+1),1,1)),2);
paulPlot(Pdn,Vydn,Vxdn,ss,ss,ss)
save('Resid24','rP','rVy','rVx');
function [Xout] = downsample(Xin,scale)
    Xout = Xin(1:scale:end,1:scale:end,1:scale:end);
end
function [] = paulPlot(P,Vy,Vx,nx,ny,nz)
        figure
        clf
        subplot(221)
        title('CUDA Run');

        subplot(222)
        imagesc(squeeze(P(:,:,(nz+1)/2))'),title("Pr")
        xlabel('x')
        ylabel('y')
        set(gca,'YDir','normal')
        colorbar
        
        subplot(223)
        imagesc(squeeze(Vy(:,:,(nz+1)/2))'),title("Vy")
        set(gca,'YDir','normal')
        xlabel('x')
        ylabel('y')
        axis equal
        colorbar

        subplot(224)
        imagesc(squeeze(Vx(:,:,(nz+1)/2))'),title("Vx")
        set(gca,'YDir','normal')
        axis equal
        colorbar
end