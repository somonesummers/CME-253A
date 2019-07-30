clear
% Load the DATA and infos

nxy = load('0_nxy.inf');  PRECIS=nxy(1); nx=nxy(2); ny=nxy(3);
if (PRECIS==8), DAT = 'double';  elseif (PRECIS==4), DAT = 'single';  end 
id = fopen('0_P.res' ); P  = fread(id,DAT); fclose(id); Ps  = reshape(P ,nx  ,ny  );
id = fopen('0_Vx.res'); Vx = fread(id,DAT); fclose(id); Vxs = reshape(Vx,nx+1,ny  );
id = fopen('0_Vy.res'); Vy = fread(id,DAT); fclose(id); Vys = reshape(Vy,nx  ,ny+1);
id = fopen('0_Txx.res'); Txx = fread(id,DAT); fclose(id); Txxs = reshape(Txx,nx  ,ny  );
id = fopen('0_Tyy.res'); Tyy = fread(id,DAT); fclose(id); Tyys = reshape(Tyy,nx  ,ny  );
id = fopen('0_Txy.res'); Txy = fread(id,DAT); fclose(id); Txys = reshape(Txy,nx+1,ny+1);
% Plot it
figure(2),clf,
subplot(231),imagesc(flipud(Ps' )),axis image,colorbar,title('P C')
subplot(232),imagesc(flipud(Vxs')),axis image,colorbar,title('Vx C')
subplot(233),imagesc(flipud(Vys')),axis image,colorbar,title('Vy C')
subplot(234),imagesc(flipud(Txxs')),axis image,colorbar,title('Txx C')
subplot(235),imagesc(flipud(Tyys')),axis image,colorbar,title('Tyy C')
subplot(236),imagesc(flipud(Txys')),axis image,colorbar,title('Txy C')

% mat = load('p_l_v.mat');
% PMs   = reshape(mat.P   ,nx  ,ny  );
% VxMs  = reshape(mat.Vx  ,nx+1,ny  );
% VyMs  = reshape(mat.Vy  ,nx  ,ny+1);
% TxxMs = reshape(mat.Txx ,nx  ,ny  );
% TyyMs = reshape(mat.Tyy ,nx  ,ny  );
% TxyMs = reshape(mat.Txy ,nx+1,ny+1);
% 
% figure(1),clf,
% subplot(231),imagesc(flipud(PMs' )),axis image,colorbar,title('P Ml')
% subplot(232),imagesc(flipud(VxMs')),axis image,colorbar,title('Vx Ml')
% subplot(233),imagesc(flipud(VyMs')),axis image,colorbar,title('Vy Ml')
% subplot(234),imagesc(flipud(TxxMs')),axis image,colorbar,title('Txx Ml')
% subplot(235),imagesc(flipud(TyyMs')),axis image,colorbar,title('Tyy Ml')
% subplot(236),imagesc(flipud(TxyMs')),axis image,colorbar,title('Txy Ml')
% 
% figure(3),clf,
% subplot(231),imagesc(flipud((Ps - PMs)' )),axis image,colorbar,title('Diff P')
% subplot(232),imagesc(flipud((Vxs - VxMs)')),axis image,colorbar,title('Diff Vx')
% subplot(233),imagesc(flipud((Vys - VyMs)')),axis image,colorbar,title('Diff Vy')
% subplot(234),imagesc(flipud((Txxs - TxxMs)')),axis image,colorbar,title('Diff Txx')
% subplot(235),imagesc(flipud((Tyys - TyyMs)')),axis image,colorbar,title('Diff Tyy')
% subplot(236),imagesc(flipud((Txys - TxyMs)')),axis image,colorbar,title('Diff Txy')
% 
% 
% 
% ResPr = max(P-mat.P);
% ResVx = max(Vx-mat.Vx);
% ResVy = max(Vy-mat.Vy);