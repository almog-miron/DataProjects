function [thetas, freq, spectra, figs] = analyse_video(filename, Fs)
% ANALYSE_VIDEO - Analyze wing motion from high-speed video.
%   Inputs:
%       filename - video file name (string)
%       Fs       - frame rate (Hz)
%   Outputs:
%       thetas   - [Nx2] matrix of wing angles (θL, θR)
%       freq     - frequency vector corresponding to spectra
%       spectra  - amplitude spectra [θL, θR, symmetric, antisymmetric]
%       figs     - handles to 4 output plots

    if nargin == 0
        filename = 'FlappingWings.mp4';
        Fs = 5000;
    end

    % Read Video
    obj = VideoReader(filename);

    % Extracting wing angles from frames
    thetas = extract_wing_angles(obj, obj.NumFrames);

    % Symmetric and anti-symmetric decomposition
    symetric = 0.5 * (thetas(:, 1) + thetas(:, 2));
    antiSymetric = 0.5 * (thetas(:, 2) - thetas(:, 1));

    % Compute amplitude spectra
    spectra = [compute_spectrum(thetas(:, 1)), compute_spectrum(thetas(:, 2)), ...
                compute_spectrum(symetric), compute_spectrum(antiSymetric)];

    % Frequency vector
    N = size(thetas, 1);
    freq = (0:(N+1)/2 - 1) * Fs / N;


    plot_results(thetas, symetric, antiSymetric, spectra, freq);
end

function thetas = extract_wing_angles(obj, totFrames)
    thetas = zeros(totFrames, 2);

    % Process each frame
    for i = 1:totFrames
        frame = readFrame(obj); 
        BW = im2bw(frame, 0.6);
        Il = BW(400:600, 300:650); %Left wing ROI
        Ir = BW(400:600, 800:1200); %Right wing ROI

        p1 = fit_wing_line(Il); %Fit left wing
        p2 = fit_wing_line(Ir); %Fit right wing
        thetas = compute_angles(p1, p2, i, thetas); %Store angles
    end
    
    % Trim transient segment
    thetas = thetas(1200:end, :);

end

function p1 = fit_wing_line(I)
    % Fits a line to the segmented wing blob
    %   I - binary image of wing
    %   Output: p - polynomial [slope, intercept] from polyfit

    rp = [regionprops(I, 'PixelList')];
    x_lst=[]; y_lst=[];
    for j = 1:length(rp)
        vals = rp(j).PixelList;
        x_lst = [x_lst; vals(:, 1)];
        y_lst = [y_lst; vals(:, 2)];
    end
    p1 = polyfit(x_lst, -y_lst, 1); % Flip y-axis to match angle definition
end

function thetas = compute_angles(p1, p2, i, thetas)
    %   Computes wing angles from slopes
    %   p1, p2 - polyfit outputs for left and right wings
    %   i - current frame index
    %   thetas - running storage matrix

    thetas(i, 1) = - atand(p1(1)); % Left wing angle
    thetas(i, 2) = atand(p2(1)); % Right wing angle

    % adjust for flipped slopes
    if p1(1) > 0
        thetas(i, 1) = - atand(p1(1));
    end
   if p2(1) < 0  
       thetas(i, 2) = - atand(p2(1));
   end
end

function output = compute_spectrum(data)
    %   Computes one-sided amplitude spectrum
    %   data - time-domain signal
    %   output - amplitude spectrum

    y = fft(data);
    N = size(y, 1);
    p2 = abs(y / N);
    ampSpec = p2(1:((N+1)/2));
    is_odd = mod(N, 2)==1;

    % Double non-DC bins except Nyquist
    ampSpec(2:end-is_odd) = 2 * ampSpec(2:end-is_odd);
    output = ampSpec;
end

function plot_results(thetas, symetric, antiSymetric, spectra, freq)
%   Plots angles and spectra
%   Returns vector of figure handles

    plot_wing_degrees(thetas(:, 1), thetas(:, 2), 'left', 'right',...
            'left and right wings degrees', 'Left_Right_Wing_Degrees');

    plot_wing_degrees(symetric, antiSymetric, 'sym', 'anti-sym',...
    'symetric and anti-symetric wings degrees', 'Sym_Anti_Wing_Degrees');

    plot_spectra_move(freq, spectra(:, 1), spectra(:, 2), 'left', 'right',...
        'spectra of left and right movements', 'Spectra_Left_Right');
    
    plot_spectra_move(freq, spectra(:, 3), spectra(:, 4), 'sym', 'anti-sym',...
    'spectra of symetric and antisymetric movements', 'Spectra_Sym_Anti');
end

function plot_wing_degrees(data1, data2, l1, l2, s1, s2)
    f = figure;
    plot(data1 - mean(data1)); hold on;
    plot(data2 - mean(data2)); 
    legend({l1, l2});
    xlabel('Time(ms)'); ylabel('Direction(degrees)');
    title(s1);
    saveas(f, s2, 'png');
end

function plot_spectra_move(freq, data1, data2, l1, l2, s1, s2)
    f = figure;
    plot(freq, data1); hold on; 
    plot(freq, data2);
    legend({l1, l2}); 
    xlabel('Freq[Hz]'); ylabel('Amplitude');
    title(s1);
    saveas(f, s2, 'png');
end