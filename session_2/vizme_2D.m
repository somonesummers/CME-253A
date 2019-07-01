clear
% Load the DATA and infos
mat = load('p_l_v.mat');
nxy = load('0_nxy.inf');  PRECIS=nxy(1); nx=nxy(2); ny=nxy(3);
if (PRECIS==8), DAT = 'double';  elseif (PRECIS==4), DAT = 'single';  end 
id = fopen('0_P.res' ); P  = fread(id,DAT); fclose(id); Ps  = reshape(P ,nx  ,ny  );
id = fopen('0_Vx.res'); Vx = fread(id,DAT); fclose(id); Vxs = reshape(Vx,nx+1,ny  );
id = fopen('0_Vy.res'); Vy = fread(id,DAT); fclose(id); Vys = reshape(Vy,nx  ,ny+1);
% Plot it
figure(2),clf,
subplot(311),imagesc(flipud(Ps' )),axis image,colorbar,title('P')
subplot(312),imagesc(flipud(Vxs')),axis image,colorbar,title('Vx')
subplot(313),imagesc(flipud(Vys')),axis image,colorbar,title('Vy')

PMs  = reshape(mat.P  ,nx  ,ny  );
VxMs = reshape(mat.Vx ,nx+1,ny  );
VyMs = reshape(mat.Vy ,nx  ,ny+1);
figure(1),clf,
subplot(311),imagesc(flipud(PMs' )),axis image,colorbar,title('P')
subplot(312),imagesc(flipud(VxMs')),axis image,colorbar,title('Vx')
subplot(313),imagesc(flipud(VyMs')),axis image,colorbar,title('Vy')

figure(3),clf,
subplot(311),imagesc(flipud((Ps-PMs)' )),axis image,colorbar,title('Diff P')
subplot(312),imagesc(flipud((Vxs-VxMs)')),axis image,colorbar,title('Diff Vx')
subplot(313),imagesc(flipud((Vys-VyMs)')),axis image,colorbar,title('Diff Vy')



ResPr = max(P-mat.P);
ResVx = max(Vx-mat.Vx);
ResVy = max(Vy-mat.Vy);