function score = GSIM(im1, im2)
    % Function to calculate GSIM
    % This function calculates the Gradient Similarity Index (GSIM)
    % between two images.
    % Input: im1, im2 - input images (assumed to be in the same size)
    % Output: score - the GSIM score
    
    % Convert images to grayscale
    im1_gray = rgb2gray(im1);
    im2_gray = rgb2gray(im2);
    
    % Calculate gradients
    [Gx1, Gy1] = imgradientxy(im1_gray);
    [Gx2, Gy2] = imgradientxy(im2_gray);
    
    % Compute gradient similarity
    sim_map = (2 * Gx1 .* Gx2 + eps) ./ (Gx1.^2 + Gx2.^2 + eps) + ...
              (2 * Gy1 .* Gy2 + eps) ./ (Gy1.^2 + Gy2.^2 + eps);
    score = mean(sim_map(:));
end