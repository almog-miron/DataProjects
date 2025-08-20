function fitting_funcs()
    load 'data\cell3196.mat';
    spikes = full(data.spikes);
    directions = data.target_direction;
    motion = data.target_motion;
    dirs = unique(directions);
    for i=1:length(dirs)
        mini = (spikes(motion(1):motion(2), directions == dirs(i)))';
        avg_act = sum(mini)/((motion(2)-motion(1))/100);
        direction = dirs(i);
        cosfit = fittype('a0 + (a1*cos(x - a2))');
        efit = fittype('a0 * (exp ^ (a1*(cos(direction-a2))))'); 
        [fit1, gof, firinfo] = fit(DIR', fr_per_dir, f, 'StartPoint', [0 0 0]);
        fitobj_cos = fit(motion, avg_act, cosfit);
        fitobj_e = fit(motion, avg_act, efit);
        figure;
        plot(fitobj_cos, motion, avg_act);
        figure;
        plot(fitobj_e, motion, avg_act);
    end
end
