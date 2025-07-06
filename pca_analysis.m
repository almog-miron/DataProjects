% Loads pre-aligned LFP data centered around go/cue/movement signals.
% Combines data across brain areas per signal type, then applies PCA per subject
% to assess variance structure and identify functional clustering of areas.
function pca_analysis()
    floc = 'lfp\lfp_';
    subjects = ['c', 'm', 'p'];
    relAreas = {'pm', 'm1', 'ss'};

    for a = 1:length(relAreas)
        areasCode.(char(relAreas(a))) = a;
    end

    for i=1:length(subjects)
        curSubject = subjects(i);
        curFname = [floc curSubject];
        curFile = load(curFname);
        [signals, splicing] = preprocessing(curFile, areasCode);
        run_pca(relAreas, signals, splicing, curSubject);
    end

end

% Combine signals by event type and across areas.
function [signals, splicing] = preprocessing(curFile, areasCode)
    signals = struct(); %contain LFP data per signal   
    splicing = []; %indicator of brain areas
    [signals.go, signals.cue, signals.mov] = deal([]);
    areaNevent = fieldnames(curFile);

    % extract brain area and event type
    for j = 1:length(areaNevent)
        curAE = char(areaNevent(j)); 
        splitAW = split(curAE, '_');
        if length(splitAW) < 2, continue; end
        
        area = char(splitAW(1));
        eventType = char(splitAW(2));

        % Add data to signals and update splicing
        if isfield(signals, eventType) && isfield(curFile, curAE) && ...
                            any(contains(fieldnames(areasCode), area))
            data = curFile.(curAE);
            if ~isempty(data)
                signals.(eventType) = [signals.(eventType); data];
                if isempty(splicing) || max(splicing) < areasCode.(area)
                    splicing = [splicing; repmat(areasCode.(area), ...
                                                    size(data, 1), 1)];
                end
            end
        end
    end
end

% Runs PCA per event signal (go, cue and mov) per subject and plots both 
% scatter and scree plots.
function run_pca(relAreas, signals, splicing, subject)
    dimensions = [2, 3]; % 2D & 3D for PCA visualizations
    smoothWindow = 10; % change as needed
    signalTypes = fieldnames(signals);

    for d = dimensions

        for r = 1:size(signalTypes, 1)
            curSignal = char(signalTypes(r));
            curData = signals.(curSignal);
            if isempty(curData), continue; end

            % Normalize LFP signals and smooth each LFP trace
            curData = edit_signal(curData, smoothWindow);

            % Perform PCA
            [~, score, ~, ~, explained] = pca(curData);

            % Visualize PCA
            visualize_pca(score, explained, d, splicing, ...
                                        curSignal, subject, relAreas);
        end
    end
end

function editedData = edit_signal(curData, smoothWindow)
    editedData = zeros(size(curData));
    for i = 1:size(curData, 1)
        tmp = curData(i,:);
        % option 1 normalize by z-score (keep amplitude info)
%         tmp = (tmp-mean(tmp)) / std(tmp); 
        % option 2 : shape-based normalization
        tmp = (tmp-mean(tmp)) / norm(tmp); 
        editedData(i,:) = smooth(tmp, smoothWindow);
    end
end

function visualize_pca(score, explained, d, splicing, curSignal, ...
                                                    subject, relAreas)
    scatter_plot(score, explained, d, splicing, curSignal, ...
                                                subject, relAreas);
    if d == 2  
        scree_plot(explained, subject, curSignal);  
         
        %run classification
        totAreas = size(unique(splicing), 1);
        brainLabels = get_brain_labels(splicing, relAreas);
        % According to the scree plot, PC1–10 explain most of the variance.
        c = decoder(score(:, 1:10), brainLabels, totAreas);
        plot_conf_chart(c, subject, curSignal, relAreas);
        plot_conf_perc(c, subject, curSignal, relAreas);
    end
end

% Creates a plot of type "scatter" from all the selected score columns 
% (1&2|3) which mostly explain the variance. "splicing" used to mark the
% brain areas as pre-defined. 
function scatter_plot(score, explained, d, splicing, curSignal, ...
                                                    subject, brainAreas)
    f = figure; 
    for k = 1:length(brainAreas)
       xaxis = score(splicing == k, 1);
       yaxis = score(splicing == k, 2);

       if d == 3
          zaxis = score(splicing == k, 3);
          scatter3(xaxis, yaxis, zaxis);
       else
           scatter(xaxis, yaxis);
       end
       hold on;
    end

    title(curSignal);
    legend(brainAreas, 'Location', 'best');
    xlabel(['PCO1 (', sprintf('%.2f', explained(1)), '%)']);
    ylabel(['PCO2 (', sprintf('%.2f', explained(2)), '%)']);
    if d == 3
        zlabel(['PCO3 (', sprintf('%.2f', explained(3)), '%)']);
    end
    saveas(f, ['scatter_', mat2str(d), '_', subject, curSignal]);
end


% Creates a graph plot of the explained variance (extracted by PCA) 
% as a function of the total number of PC's. 
function scree_plot(explained, subject, curSignal)
    f = figure;
    plot(explained(1:min(50, end), 1));
    title(['Scree Plot: ', curSignal, '- Subject ',subject]);
    xlabel('Principal Component #');
    ylabel('Explained Variance (%)');
    saveas(f, ['scree_' curSignal, '_',subject, '.fig']);
end

% Converts numeric splicing vector to a cell array of brain area labels
function brain_labels = get_brain_labels(splicing, relAreas)    
    brain_labels = cell(length(splicing), 1);
    for i = 1:size(relAreas, 2)
        brain_labels(splicing == i) = relAreas(i);
    end
end


function c = decoder(score, brainLabels, totGroups)
    c = zeros(totGroups,totGroups);
    relAreas = unique(brainLabels, 'stable');

   for i = 1:1000
        cv = cvpartition(brainLabels, 'HoldOut', 0.10);
        trainIndxs = training(cv); 
        testIndxs = test(cv); 

        sampleData = score(testIndxs, :);
        trainingData = score(trainIndxs, :);

        mdl = fitcdiscr(trainingData, brainLabels(trainIndxs));
        class = predict(mdl, sampleData);

        c = c + confusionmat(brainLabels(testIndxs), ...
                                            class, 'order', relAreas);
    end
end

% plot confusion chart
function plot_conf_chart(c, subject, curSignal, relAreas)
    figure;
    cm = confusionchart(c, relAreas);
    cm.RowSummary = 'row-normalized';
    cm.ColumnSummary = 'column-normalized';
    cm.Title = ['Prediction results by nums - ' subject ' ' curSignal];

    saveas(gcf, [subject, '_confusion_', curSignal '.png']);
end

% plot confusion prediction by percentage
function plot_conf_perc(c, subject, curSignal, relAreas)
    totGroups = size(relAreas, 2);
    perc = zeros(totGroups, totGroups);
    labels= strings(totGroups,totGroups);
    for i=1:totGroups
        perc(i, :) = round((c(i, :) / sum(c(i, :))) * 100);
        for j=1:totGroups
            labels(i, j) = [mat2str(c(i,j)), ...
                                sprintf('\n %.2f', perc(i,j)), '%'];
        end
    end
    figure;
    confusionchart(perc, relAreas);
    title(['Prediction by Percentages - ' subject ' ' curSignal]);
    saveas(gcf, [subject, '_confusionP_', curSignal '.png']);
end



