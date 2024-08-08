function result_image = psychovisual(input_image)
%b=imsize(input_image)
% Convert the input image to double precision
input_image_double = im2double(input_image);

% Convert the input image to grayscale
gray_image = rgb2gray(input_image_double);

% Convert the grayscale image to binary
threshold = graythresh(gray_image); % Automatic thresholding using Otsu's method
binary_image = imbinarize(gray_image, threshold);

% Invert the binary image
inverted_binary_image = ~binary_image;

% Define the size and standard deviation of the Gaussian filter kernel
kernel_size = 5; % Adjust kernel size as needed
sigma = 2; % Adjust standard deviation as needed

% Create Gaussian filter kernel
kernel = fspecial('gaussian', [kernel_size kernel_size], sigma);

% Apply Gaussian filtering to each color channel separately
filtered_image = zeros(size(input_image_double)); % Initialize filtered image

for i = 1:3
    filtered_image(:,:,i) = conv2(input_image_double(:,:,i), kernel, 'same');
end

% Convert binary images to the same data type as filtered_image (double)
binary_image = double(binary_image);
inverted_binary_image = double(inverted_binary_image);

% Perform element-wise multiplication between original and filtered images based on binary masks
result_image = input_image_double .* repmat(binary_image, [1, 1, 3]) + ...
    filtered_image .* repmat(inverted_binary_image, [1, 1, 3]);



figure,
imshow(input_image_double);
title('input Image');

% Display the result
figure,
imshow(result_image);
title('Result Image');

