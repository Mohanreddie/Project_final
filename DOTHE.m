function enhancedImage = DOTHE(originalImage, threshold)
    % Convert the image to grayscale
    if size(originalImage, 3) == 3
        originalImage = rgb2gray(originalImage);
    end
    
    % Perform histogram equalization
    histEqualizedImage = histeq(originalImage);
    
    % Compute the difference image
    diffImage = double(histEqualizedImage) - double(originalImage);
    
    % Apply enhancement based on threshold
    enhancedImage = originalImage;
    enhancedImage(diffImage > threshold) = histEqualizedImage(diffImage > threshold);
end