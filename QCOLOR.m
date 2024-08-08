function score = QCOLOR(im1, im2)
    % Function to calculate QCOLOR
    % This function calculates the Quality through Colorfulness (QCOLOR)
    % between two images.
    % Input: im1, im2 - input images (assumed to be in the same size)
    % Output: score - the QCOLOR score
    
    % Convert images to Lab color space
    im1_lab = rgb2lab(im1);
    im2_lab = rgb2lab(im2);
    
    % Calculate standard deviations of Lab channels
    std_dev1 = std(reshape(im1_lab, [], 3));
    std_dev2 = std(reshape(im2_lab, [], 3));
    
    % Calculate mean of standard deviations
    mean_std_dev1 = mean(std_dev1);
    mean_std_dev2 = mean(std_dev2);
    
    % Calculate QCOLOR score
    score = mean_std_dev2 / mean_std_dev1;
end
