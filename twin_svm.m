function [w_1, b_1, w_2, b_2] = twin_svm(data_1, data_2, C_1, C_2)
% INPUT, i=1,2
%  data_i:   dim_i-by-num_i matrix. num_i is the number of data_i points,
%  dim_i is the dimension of a point
%  C_i:      the tuning parameter
% OUTPUT, i=1,2
%  w_i:      dim_i-bu-1 vector, the normal direction of hyperplane
%  b_i:      a scalar, the bias
    [num_1, dim_1] = size(data_1');
    [num_2, dim_2] = size(data_2');
    
    cvx_begin
        variables w_1(dim_1) b_1 xi_1(num_1);
        minimize(sum((data_1' * w_1 + b_1).^2) / 2 + C_1 * sum(xi_1));
        subject to
            - (data_2' * w_1 + b_1) + xi_1 >= 1;
            xi_1 >= 0;
    cvx_end
    
    cvx_begin
        variables w_2(dim_2) b_2 xi_2(num_2);
        minimize(sum((data_2' * w_2 + b_2).^2) / 2 + C_2 * sum(xi_2));
        subject to
            - (data_1' * w_2 + b_2) + xi_2 >= 1;
            xi_2 >= 0;
    cvx_end
end