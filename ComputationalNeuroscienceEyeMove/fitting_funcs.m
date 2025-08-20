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



function my_angle = get_angle(frs_dirs, to_test_row)
    mini = to_test_row;
    tot_ev = size(mini,1);
    avg = smooth(((sum(mini)*1000)/tot_ev), 30);
    
    prev_gap = abs(frs_dirs(2, 1) - avg);
    my_angle = frs_dirs(1, 1);
    for j=1:size(frs_dirs, 2)
        gap = abs(frs_dirs(2, j) - avg);
        if gap < prev_gap
            my_angle = frs_dirs(1, j);
        end
    end
end

function frs_dirs = get_frs_dirs(directions, dir_unq, spikes, psth)
    for i=1:length(dir_unq)
        mini = spikes(find(directions == dir_unq(i)), :);
        tot_ev = size(mini,1);
        avg_act = smooth(((sum(mini)*1000)/tot_ev), 30);
        psth(i, :) = avg_act;
    end
    frs_dirs = [dir_unq;col_fr];
end



%%kmeans is a vector that contains the min distance of the angle from the
%%given neuron data. 
function kmean = find_similar(v1, to_train, directions, dir_unq)
    sums = zeros(1, size(dir_unq, 1));
    for i=1:length(dir_unq)
        train_by_dir = to_train(find(directions == dir_unq(i)), :);
        v1_mult = repmat(v1, size(train_by_dir, 1), 1);
        sp_vs_v1 = (train_by_dir - v1_mult);
        sum_sp_vs_v1 = sum(sp_vs_v1'== 0);
        sums(1, i) = max(sum_sp_vs_v1);
    end
    [~, indx] = max(sums);
    kmean = dir_unq(indx);
end