function enhancedImage = DHECI(originalImage, alpha)
    % Convert the image to grayscale
    if size(originalImage, 3) == 3
        originalImage = rgb2gray(originalImage);
    end
    
    % Perform histogram equalization
    histEqualizedImage = histeq(originalImage);
    
    % Compute the difference image
    diffImage = double(histEqualizedImage) - double(originalImage);
    
    % Enhance contrast
    enhancedImage = originalImage + alpha * diffImage;
    
    % Clip values to [0, 1]
    enhancedImage(enhancedImage < 0) = 0;
    enhancedImage(enhancedImage > 1) = 1;
end