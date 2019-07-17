load('timeRuns.mat');
MTP_max = 733; %[Gb/s]   
f1 = figure;
plot(edge,speedd/MTP_max,'b.-')
hold on
plot(edge,speeds/MTP_max,'rx-')
legend('Double Precision', 'Single Precision','Location','southeast');
xlabel('^3\surd (n_x n_y n_z)')
ylabel('MTP_{eff}/MTP_{max}');
saveas(f1,'runTimes.png');