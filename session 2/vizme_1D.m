% vizme wave 1D
fname = 'P_c.dat';
id = fopen(fname); P = fread(id,'double'); fclose(id);
figure(3),clf,plot(P,'-d'),title('Wave 1D'),drawnow
% !rm a.out Pc.dat