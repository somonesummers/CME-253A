function [] = plot3sections(nx,ny,nz,S,str)
    surf(1:nx,1:ny,(nz+1)/2*ones(nx,ny),squeeze(S(:,:,(nz+1)/2))')
    shading interp
    title(str);
    hold on
    [x,z] = meshgrid(1:nx, 1:nz);
    y = (ny+1)/2 * ones(ny);
    surf(x,y,z,squeeze(S(:,(ny+1)/2,:)));
    shading interp
    [y,z] = meshgrid(1:ny, 1:nz);
    x = (nx+1)/2 * ones(nx);
    surf(x,y,z,squeeze(S((ny+1)/2,:,:))');
    shading interp
    xlabel('X')
    ylabel('Y')
    zlabel('Z')
    colorbar
    hold off
    