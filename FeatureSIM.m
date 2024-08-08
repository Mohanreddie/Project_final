function score = FeatureSIM(im1, im2)
    % Function to calculate FSIM
    % This function calculates the Feature Similarity Index (FSIM)
    % between two images.
    % Input: im1, im2 - input images (assumed to be in the same size)
    % Output: score - the FSIM score
    
    % Convert images to grayscale
    im1_gray = rgb2gray(im1);
    im2_gray = rgb2gray(im2);
    
    % Calculate gradient magnitudes
    Gx1 = imgradientxy(im1_gray, 'Sobel');
    Gy1 = imgradientxy(im1_gray', 'Sobel')';
    GradientMap1 = sqrt(Gx1.^2 + Gy1.^2);
    
    Gx2 = imgradientxy(im2_gray, 'Sobel');
    Gy2 = imgradientxy(im2_gray', 'Sobel')';
    GradientMap2 = sqrt(Gx2.^2 + Gy2.^2);
    
    % Calculate contrast similarity
    contrast_sim = corr2(GradientMap1, GradientMap2);
    
    % Calculate structural similarity
    struct_sim = sum(sum(GradientMap1 .* GradientMap2)) / sum(sum(GradientMap1.^2));
    
    % Calculate FSIM
    score = contrast_sim * struct_sim;
end