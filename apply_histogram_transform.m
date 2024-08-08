function output_image = apply_histogram_transform(input_image, optimized_histogram)
    % Compute Probability Density Function (PDF)
    p = optimized_histogram / sum(optimized_histogram);
    % Compute Cumulative Distribution Function (CDF)
    C = cumsum(p);
    % Compute modified transform function
    T = round((255 * C) + 0.5);
    % Apply transform to input image
    output_image = zeros(size(input_image));
    for i = 1:size(input_image, 1)
        for j = 1:size(input_image, 2)
            intensity = input_image(i, j) + 1; % Add 1 to match MATLAB indexing
            output_image(i, j) = T(intensity);
        end
    end
    % Apply gamma correction
    output_image = imadjust(output_image, [], [], 1.5); %
end