function updated_position = update_position(position, fitness_value, fitness_values, num_elements)
    other_element_index = datasample(1:num_elements, 1); % Select random index from the same group
    epsilon = 1e-6;
    if fitness_value <= fitness_values(other_element_index)
        sigma_squared = 1;
    else
        sigma_squared = exp((fitness_values(other_element_index) - fitness_value) / abs(fitness_value) + epsilon);
    end
    updated_position = position .* (1 + randn(1, 1) * sqrt(sigma_squared));
end