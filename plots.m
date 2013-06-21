%%
nparticle = [1024 2048 4096 6144 8192 12288 16384 16384];
group = [16 32 64 128 256 512 1024];
bbox = [100 100 100 100 100 100 100 200];

%%
for i = 1:length(nparticle)
    figure(i);
    hold on;
    title([int2str(nparticle(i)) ' Particles, ' int2str(bbox(i)) ' box']);
    xlabel('Group Size');
    ylabel('ms/update');
    plot(group, fn(i,:), '--ob', 'MarkerEdgeColor', 'black', ...
        'MarkerFaceColor', 'b');
    plot(group, fnc(i,:), '--og', 'MarkerEdgeColor', 'black', ...
        'MarkerFaceColor', 'g');
    plot(group, ft(i,:), '--or', 'MarkerEdgeColor', 'black', ...
        'MarkerFaceColor', 'r');
    plot(group, ftc(i,:), '--oy', 'MarkerEdgeColor', 'black', ...
        'MarkerFaceColor', 'y');
    legend('force\_naive', 'force\_naive\_clip', 'force\_tile', ...
        'force\_tile\_clip', 'Location', 'SouthOutside');
end