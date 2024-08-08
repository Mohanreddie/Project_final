clc
clear all
close all

inp=input('ENTER IMAGE :')

originalImage = imread(inp); 

input_image = im2double(originalImage);

figure,
imshow(input_image);
title('input')

% Convert the color input image to grayscale
grayscale_image = rgb2gray(input_image);

% Calculate the average pixel value (Lm) in the grayscale image
Lm = mean(grayscale_image(:));

% Calculate the maximum intensity in the grayscale image (Lmax)
Lmax = max(grayscale_image(:));

% Calculate the contrast (Cin) of the image
Cin = -((Lmax - Lm) / Lmax) + 1.0000 ;

if Cin <= 0.5 
    exposure_classification = 'Underexposed';
elseif Cin >= 0.5
    exposure_classification = 'Overexposed';
else
    exposure_classification = 'Properly Exposed';
end

disp(['Average Pixel Value (Lm): ', num2str(Lm)]);
disp(['Exposure Classification: ', exposure_classification]);
disp(['Contrast (Cin): ', num2str(Cin)]);



% Define the parameters
alpha = 13; % Example value for alpha parameter
beta = 1e4; % Example value for beta parameter
G = 10; % Minimum number of optimization iterations to update categories
Min = 50; % Minimum number of optimization iterations
Max = 1000; % Maximum number of optimization iterations

% Define the objective function
objective_function = @(histogram) compute_objective_function(histogram, alpha, beta);


re=input_image;
% Optimize histogram using ICSO algorithm
best_histograms = zeros(256, 3); % Initialize best histograms
for channel = 1:3
    histogram_input = imhist(input_image(:,:,channel));
    best_histograms(:, channel) = optimize_with_icso(histogram_input, objective_function, G, Min, Max);
end

% Get the histogram for each color channel separately
redHistogram = best_histograms(:, 1);
greenHistogram = best_histograms(:, 2);
blueHistogram = best_histograms(:, 3);

% Combine histograms into a single matrix
optimizedHistograms = [redHistogram, greenHistogram, blueHistogram];

% Get image dimensions
[rows, cols, ~] = size(input_image);

% Initialize the reconstructed image
reconstructedImage = zeros(rows, cols, 3);

% Iterate over each color channel
for k = 1:3
    % Compute PDF
    Np = sum(optimizedHistograms(:, k)); % Total number of pixels
    p = optimizedHistograms(:, k) / Np; % Probability density function

    % Compute CDF
    C = cumsum(p); % Cumulative distribution function

    % Compute modified transform function
    L = 256; % Assuming 8-bit images
    T = (L - 1) * C ;

    % Map intensity values to reconstructed image
    for i = 1:rows
        for j = 1:cols
            % Ensure intensity value is within valid range
            intensity = round(input_image(i, j, k) * (L - 1) + 0.5);
            intensity = max(1, min(intensity, L));
            
            % Map intensity value to reconstructed image
            reconstructedImage(i, j, k) = T(intensity) / (L - 1);
        end
    end
end
reconstructedImage=zeros(1,1).*(reconstructedImage)+re
% Compute mean intensity of the reconstructed image
meanIntensity = mean(reconstructedImage(:));

% Define gamma range based on brightness level
if meanIntensity < 0.5 % Low brightness
    gamma_range = [0, 1];
    gamma1=2;
else % High brightness
    gamma_range = [1, 2];
    gamma1=0.9;
end

% Adjust gamma value to prevent over-enhancement
gamma = min(max(gamma_range), gamma_range(1));

% Apply gamma correction
reconstructedImages = reconstructedImage * gamma1;


% Display the computed histogram and output image
figure;

subplot(1,2,1)
imshow(input_image);
title('input')

subplot(1, 2, 2);
imshow(reconstructedImages,[]);
title('Enhanced Image');
result_image = psychovisual(reconstructedImages)
[ssimValue, fsimValue, gsimValue, qcolorValue] = evaluateEnhancementMetrics(reconstructedImages, input_image);

    % Display results in message box
    msg = sprintf('PROPOSED SSIM: %.4f\nFSIM: %.4f\nGSIM: %.4f\nQCOLOR: %.4f', ssimValue, fsimValue, gsimValue, qcolorValue);
    msgbox(msg, 'Enhancement Metrics');
    
% Example usage:
originalImage = input_image;
threshold = 0.2;
enhancedImage(:,:,1) = DOTHE(originalImage(:,:,1), threshold);
enhancedImage(:,:,2) = DOTHE(originalImage(:,:,2), threshold);
enhancedImage(:,:,3) = DOTHE(originalImage(:,:,3), threshold);


[ssimValue1, fsimValue1, gsimValue1, qcolorValue1] = evaluateEnhancementMetrics(enhancedImage, input_image);

 msg = sprintf('DOTHE SSIM: %.4f\nFSIM: %.4f\nGSIM: %.4f\nQCOLOR: %.4f', ssimValue1, fsimValue1, gsimValue1, qcolorValue1);
    msgbox(msg, 'Enhancement Metrics');

% Example usage:
originalImage1 = input_image;
alpha = 13; 
enhancedImage1(:,:,1) = DHECI(originalImage1(:,:,1) , alpha);
enhancedImage1(:,:,2) = DHECI(originalImage1(:,:,2) , alpha);
enhancedImage1(:,:,3) = DHECI(originalImage1(:,:,3) , alpha);

[ssimValue2, fsimValue2, gsimValue2, qcolorValue2] = evaluateEnhancementMetrics(enhancedImage1, input_image);

 msg = sprintf('DHECI SSIM: %.4f\nFSIM: %.4f\nGSIM: %.4f\nQCOLOR: %.4f', ssimValue2, fsimValue2, gsimValue2, qcolorValue2);
    msgbox(msg, 'Enhancement Metrics');
figure;

subplot(2,2,1)
imshow(input_image);
title('input')

subplot(2, 2, 2);
imshow(reconstructedImages,[]);
title('proposed Image');
subplot(2, 2, 3);
imshow(enhancedImage,[]);
title('DOTHE Image');
subplot(2, 2, 4);
imshow(enhancedImage1,[]);
title('DHECI Image');