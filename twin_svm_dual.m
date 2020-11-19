function [w_1, b_1, alpha_1, w_2, b_2, alpha_2] = twin_svm_dual(data_1, data_2, C_1, C_2)
% INPUT, i=1,2
%  data_i:   dim_i-by-num_i matrix. num_i is the number of data_i points,
%  dim_i is the dimension of a point
%  C_i:      the tuning parameter
% OUTPUT, i=1,2
%  w_i:      dim_i-by-1 vector, the normal direction of hyperplane
%  b_i:      a scalar, the bias
%  alpha_i:  num-by-1 vector, dual variables

    [num_1, dim_1] = size(data_1');
    [num_2, dim_2] = size(data_2');
    
    A = [data_1', ones(num_1, 1)];
    B = [data_2', ones(num_2, 1)];
    H_1 = B * (A' * A)^(-1) * B';
    H_2 = A * (B' * B)^(-1) * A';
    
    cvx_begin
        variable alpha_1(num_2);
        maximize(sum(alpha_1) - alpha_1' * H_1 * alpha_1 / 2);
        subject to
            alpha_1 >= 0;
            alpha_1 <= C_1;
    cvx_end
    
    u_1 = - (A' * A)^(-1) * B' * alpha_1;
    w_1 = u_1(1:dim_1);
    b_1 = u_1(dim_1+1);
    
    cvx_begin
        variable alpha_2(num_1);
        maximize(sum(alpha_2) - alpha_2' * H_2 * alpha_2 / 2);
        subject to
            alpha_2 >= 0;
            alpha_2 <= C_2;
    cvx_end
    
    u_2 = - (B' * B)^(-1) * A' * alpha_2;
    w_2 = u_2(1:dim_2);
    b_2 = u_2(dim_2+1);
end