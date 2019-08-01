load('timeRuns.mat');
MTP_max = 733; %[Gb/s]   
f1 = figure;
plot(edged,speedd/MTP_max,'b.-')
hold on
%plot(edge,speeds/MTP_max,'rx-')
legend('Double Precision','Location','southeast');
xlabel('^3\surd (n_x n_y n_z)')
ylabel('MTP_{eff}/MTP_{max}');
saveas(f1,'runTimes.png');

f2 = figure;
plot(edged,itd,'b.-')
title('Global Iterations by Domain Size')
legend('Double Precision', 'Single Precision','Location','southeast');
xlabel('^3\surd (n_x n_y n_z)')
ylabel('# iterations');
saveas(f2,'iterations.png');

d04 = load('Resid04.mat');
d08 = load('Resid08.mat');
d16 = load('Resid16.mat');
d24 = load('Resid24.mat');
d32 = load('Resid32.mat');

rP = [d04.rP,d08.rP,d16.rP,d24.rP,d32.rP]./edged.^3;
rVx = [d04.rVx,d08.rVx,d16.rVx,d24.rVx,d32.rVx]./edged.^3;
rVy = [d04.rVy,d08.rVy,d16.rVy,d24.rVy,d32.rVy]./edged.^3;

f3 = figure;
plot(edged,rP,'b.-')
hold on
plot(edged,rVx,'gx-')
plot(edged,rVy,'rx-')
legend('Pressue Residual', 'Vx Residual','Vy Residual','Location','northeast');
xlabel('^3\surd (n_x n_y n_z)')
ylabel('L2 Norm of Residual/# grid points');
saveas(f3,'resids.png');