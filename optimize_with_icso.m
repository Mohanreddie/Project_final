function best_histogram = optimize_with_icso(initial_histograms, objective_function, G, Min, Max)
    % Initialize histograms with input histogram for the first 75 chickens
    histograms = initial_histograms;
    % Initialize optimization parameters
    t = 0;
    
    N = 150;    % Total number of initial states (chickens)
    RN = 23;    % Number of roosters
    HN = 105;   % Number of hens
    CN = 22;    % Number of chicks
    MN = 53;    % Number of mother hens
    w_min = 0.4;
    w_max = 0.9;
 
    
        % Update categories every G iterations
        if mod(t, G) == 0
            % Ordering fitness
            fitness_values = zeros(1, size(histograms, 2));
            for i = 1:size(histograms, 2)
                % Compute fitness value for histogram i
                fitness_values(i) = objective_function(histograms(:, i));
            end
            % Sort histograms based on fitness values
            [~, sorted_indices] = sort(fitness_values);
            sorted_histograms = histograms(:, sorted_indices);
            
            % Ensure sorted_histograms has at least RN columns
            if size(sorted_histograms, 2) >= RN
                % Classify histograms into roosters, hens, and chicks
                roosters = sorted_histograms(:, 1:RN);
                hens = sorted_histograms(:, RN+1:RN+HN);
                chicks = sorted_histograms(:, RN+HN+1:end);
                
                % Update positions of roosters
                for i = 1:RN
                    roosters(:, i) = update_position(roosters(:, i), fitness_values(i), fitness_values, RN);
                end
                
                % Update positions of hens
                for i = 1:HN
                    rooster_index = mod(i, RN) + 1; % Select rooster from the same group
                    other_chicken_index = datasample([1:RN+HN], 1); % Select random chicken index
                    hens(:, i) = update_position(hens(:, i), fitness_values(RN+i), fitness_values([rooster_index, other_chicken_index]), HN);
                end
                
                % Update positions of chicks
                for i = 1:CN
                    mother_hen_index = mod(i, HN) + 1; % Select mother hen from the same group
                    rooster_index = mod(i, RN) + 1; % Select rooster from the same group
                    chicks(:, i) = update_position(chicks(:, i), fitness_values(RN+HN+i), fitness_values([rooster_index, mother_hen_index]), CN);
                end
                
                % Combine updated categories
                histograms = [roosters hens chicks];
                
                histograms
            end
            
            % Increment optimization iteration count
            t = t + 1;
        end
        
        
        % Perform other optimization steps if necessary
   
    % Return the best histogram
    [~, best_index] = min(fitness_values);
  
    best_histogram = histograms(:, best_index);
end