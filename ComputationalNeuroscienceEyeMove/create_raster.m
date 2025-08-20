function create_raster()
fnm = dir('data\');
for findx = 3:length(fnm)
    fname = fnm(findx).name;
    load(['data\' fname]);
    spikes = full(data.spikes);
    directions = data.target_direction;
    motion = data.target_motion;
    dirs = unique(directions);
    t = tiledlayout(2, 4);
    title(t, fname(1:8));
    for i=1:length(dirs)
        mini = (spikes(motion(1):motion(2), directions == dirs(i)))';
        nexttile;
        spy(sparse(mini), '|');  set(gca, 'PlotBoxAspectRatio', [1 1 1]);
        set(gca, 'XTickLabel', [motion(1):150:motion(2)]);
        title(['angle ' mat2str(dirs(i))]);
        saveas(t, ['raster_plot_' fname(1:8) '.fig']);
    end
end