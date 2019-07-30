clear
% Load the DATA and infos
cd d32
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
% mat = load('../p_l_v.mat');
% ResPr = max(P-mat.P);
% ResVx = max(Vx-mat.Vx);
% ResVy = max(Vy-mat.Vy);
% ResVz = max(Vz-mat.Vz);
% ResTxx = max(Txx-mat.Txx);
% ResTyy = max(Tyy-mat.Tyy);
% ResTzz = max(Tzz-mat.Tzz);
% ResTxy = max(Txy-mat.Txy);
% ResTxz = max(Txz-mat.Txz);
% ResTyz = max(Tyz-mat.Tyz);

% figure(1)
%         clf
%         subplot(221)
%         semilogy(mat.evol);
%         title('Matlab Run');
% 
%         subplot(222)
%         temp1 = reshape(mat.P,nx,ny,nz);
%         imagesc(squeeze(temp1(:,:,(nz+1)/2))'),title("Pr")
%         xlabel('x')
%         ylabel('y')
%         set(gca,'YDir','normal')
%         colorbar
%         
%         subplot(223)
%         temp2 = reshape(mat.Vy,nx,ny+1,nz);
%         imagesc(squeeze(temp2(:,:,(nz+1)/2))'),title("Vy")
%         set(gca,'YDir','normal')
%         xlabel('x')
%         ylabel('y')
%         axis equal
%         colorbar
% 
%         subplot(224)
%         temp3 = reshape(mat.Vx,nx+1,ny,nz);
%         imagesc(squeeze(temp3(:,:,(nz+1)/2))'),title("Vx")
%         set(gca,'YDir','normal')
%         axis equal
%         colorbar
%         
figure(2)
        clf
        subplot(221)
        title('CUDA Run');

        subplot(222)
        temp1 = reshape(P,nx,ny,nz);
        imagesc(squeeze(temp1(:,:,(nz+1)/2))'),title("Pr")
        xlabel('x')
        ylabel('y')
        set(gca,'YDir','normal')
        colorbar
        
        subplot(223)
        temp2 = reshape(Vy,nx,ny+1,nz);
        imagesc(squeeze(temp2(:,:,(nz+1)/2))'),title("Vy")
        set(gca,'YDir','normal')
        xlabel('x')
        ylabel('y')
        axis equal
        colorbar

        subplot(224)
        temp3 = reshape(Vx,nx+1,ny,nz);
        imagesc(squeeze(temp3(:,:,(nz+1)/2))'),title("Vx")
        set(gca,'YDir','normal')
        axis equal
        colorbar
%%
 f = figure(3)
    subplot(221)
        plot3sections(nx,ny,nz,Ps,'Pressure')
       % view(2)
    subplot(222)
        plot3sections(nx,ny,nz,.5*(Vys(:,2:end,:)+Vys(:,1:end-1,:)),'Vy')
        % view(2)
    subplot(223)
        plot3sections(nx,ny,nz,.5*(Vxs(2:end,:,:)+Vxs(1:end-1,:,:)),'Vx')
         %view(2)
    subplot(224)
        plot3sections(nx,ny,nz,Tyys,'Tyy')
         %view(2)
saveas(f,'plots.png')  

 