load('timeRuns.mat');
MTP_max = 733; %[Gb/s]   
f1 = figure;
plot(grid,speedd/MTP_max,'b.-')
hold on
plot(grid,speeds/MTP_max,'rx-')
legend('Double Precision', 'Single Precision','Location','southeast');
xlabel('\surd (n_x n_y)')
ylabel('MTP_{eff}/MTP_{max}');
saveas(f1,'runTimes.png');