function obj = compute_objective_function(histogram, alpha, beta)
    % Compute the terms of the objective function
    term1 = norm(ones(size(histogram)) - histogram)^2;
    term2 = beta * norm(diff(histogram))^2; % Assuming histogram is a column vector
    % Combine terms to compute the objective function
    obj = term1 + alpha * term2;
end