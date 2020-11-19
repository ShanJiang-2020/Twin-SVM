%% Description
% INPUT, i=1,2
%  data_i:   dim_i-by-num_i matrix. num_i is the number of data_i points,
%  dim_i is the dimension of a point
%  C_i:      the tuning parameter
% OUTPUT
%  w_i:      dim_i-bu-1 vector, the normal direction of hyperplane of prim
%  problem
%  b_i:      a scalar, the bias of prim problem
%  w_i_d:    dim_i-by-1 vector, the normal direction of hyperplane of dual
%  problem
%  b_i_d:    a scalar, the bias of dual problem
%  alpha_i:  num-by-1 vector, dual variables
% If you want to draw a graph, please change dim_1 and dim_2 to 2 or 3 by
% yourself, and the dimension in 'Plot' section needs to be changed
% accordingly.
%% Data generation
% Generate training data_1 and data_2, also set the value of C_1 and C_2.
dim_1 = 2; % set dimension to 2 so as to plot a 2-dimensional graph
dim_2 = 2;
num_1 = 1000;
num_2 = 1000;
upper = 200; % range of random numbers U(0, upper)
% set the 'gradient', 'intercept' of the train data, also introduce
% Gaussian noise
a_1 = 0.9; b_1 = 50;
data_1 = zeros(dim_1, num_1); % assign space
data_2 = zeros(dim_2, num_2);
data_1(1,:) = upper * rand(1, num_1); % generate first column
data_1(2,:) = a_1 * data_1(1,:) + b_1 + normrnd(0, 20, 1, num_1); % the second column is an affine mapping of the first column, plus a Gaussian noise
a_2 = 1.1; b_2 = -50;
data_2(1,:) = upper * rand(1, num_2); % generate first column
data_2(2,:) = a_2 * data_2(1,:) + b_2 + normrnd(0, 20, 1, num_2); % the second column is an affine mapping of the first column, plus a Gaussian noise
C_1 = 0.05;
C_2 = 0.05;
% testing data - same generation mode with the training data
dim_t_1 = dim_1;
dim_t_2 = dim_2;
num_t_1 = 100;
num_t_2 = 100;
data_t_1 = zeros(dim_t_1, num_t_1); % assign space
data_t_2 = zeros(dim_t_2, num_t_2);
data_t_1(1,:) = upper * rand(1, num_t_1); % generate first column
data_t_1(2,:) = a_1 * data_t_1(1,:) + b_1 + normrnd(0, 20, 1, num_t_1); % the second column is an affine mapping of the first column, plus a Gaussian noise
a_2 = 1.1; b_2 = -50;
data_t_2(1,:) = upper * rand(1, num_t_2); % generate first column
data_t_2(2,:) = a_2 * data_t_2(1,:) + b_2 + normrnd(0, 20, 1, num_t_2); % the second column is an affine mapping of the first column, plus a Gaussian noise
% labels for testing data
labels_t = [ones(num_t_1, 1); -ones(num_t_2, 1)];
%% Calculation
[w_1, b_1, w_2, b_2] = twin_svm(data_1, data_2, C_1, C_2); % prim problem
[w_1_d, b_1_d, alpha_1, w_2_d, b_2_d, alpha_2] = twin_svm_dual(data_1, data_2, C_1, C_2); % dual problem

%% Add labels for testing data, calculate classification accuracy
distance_1 = [data_t_1' * w_1 + b_1; data_t_2' * w_1 + b_1]; % distance to first hyperplane, prim problem
distance_2 = [data_t_1' * w_2 + b_2; data_t_2' * w_2 + b_2]; % distance to second hyperplane, prim problem
distance_1_dual = [data_t_1' * w_1_d + b_1_d; data_t_2' * w_1_d + b_1_d]; % distance to first hyperplane, dual problem
distance_2_dual = [data_t_1' * w_2_d + b_2_d; data_t_2' * w_2_d + b_2_d]; % distance to second hyperplane, dual problem
len = length(distance_1);
labels_prim = zeros(len, 1);
labels_dual = zeros(len, 1);
for i=1:len
    if abs(distance_1(i)) <= abs(distance_2(i))
        labels_prim(i) = 1; % set label=1 if the data point belongs to class 1
    else
        labels_prim(i) = -1; % set label=-1 if the data point belongs to class 2
    end
    if abs(distance_1_dual(i)) <= abs(distance_2_dual(i))
        labels_dual(i) = 1; % set label=1 if the data point belongs to class 1
    else
        labels_dual(i) = -1; % set label=-1 if the data point belongs to class 2
    end
end

accuracy_prim = sum(labels_prim == labels_t) / len;
accuracy_dual = sum(labels_dual == labels_t) / len;
%% Plot
class_1 = []; % array to store the the first class point
len_1 = 0; % length of class_1
class_2 = []; % array to store the the second class point
len_2 = 0; % length of class_2
class_1_dual = []; % array to store the the first class point
len_1_dual = 0; % length of class_1
class_2_dual = []; % array to store the the second class point
len_2_dual = 0; % length of class_2
% determine parameters of two lines
x = [0, upper];
a_prim_1 = - w_1(2) / w_1(1);
b_prim_1 = b_1 / w_1(1);
y_prim_1 = a_prim_1 * x + b_prim_1;
a_prim_2 = - w_2(2) / w_2(1);
b_prim_2 = b_2 / w_2(1);
y_prim_2 = a_prim_2 * x + b_prim_2;
a_dual_1 = - w_1_d(2) / w_1_d(1);
b_dual_1 = b_1_d / w_1_d(1);
y_dual_1 = a_dual_1 * x + b_dual_1;
a_dual_2 = - w_2_d(2) / w_2_d(1);
b_dual_2 = b_2_d / w_2_d(1);
y_dual_2 = a_dual_2 * x + b_dual_2;
% Plot - tesing data, original data and two lines
for i=1:len
    if i <= num_t_1 % data_t_1
        if labels_prim(i)==1
            len_1 = len_1 + 1;
            class_1(len_1, 1) = data_t_1(1, i);
            class_1(len_1, 2) = data_t_1(2, i);
        else
            len_2 = len_2 + 1;
            class_2(len_2, 1) = data_t_1(1, i);
            class_2(len_2, 2) = data_t_1(2, i);
        end
        if labels_dual(i)==1
            len_1 = len_1 + 1;
            class_1_dual(len_1, 1) = data_t_1(1, i);
            class_1_dual(len_1, 2) = data_t_1(2, i);
        else
            len_2 = len_2 + 1;
            class_2_dual(len_2, 1) = data_t_1(1, i);
            class_2_dual(len_2, 2) = data_t_1(2, i);
        end
    else % data_t_2
        if labels_prim(i)==1
            len_1 = len_1 + 1;
            class_1(len_1, 1) = data_t_2(1, i-num_t_1);
            class_1(len_1, 2) = data_t_2(2, i-num_t_1);
        else
            len_2 = len_2 + 1;
            class_2(len_2, 1) = data_t_2(1, i-num_t_1);
            class_2(len_2, 2) = data_t_2(2, i-num_t_1);
        end
        if labels_dual(i)==1
            len_1 = len_1 + 1;
            class_1_dual(len_1, 1) = data_t_2(1, i-num_t_1);
            class_1_dual(len_1, 2) = data_t_2(2, i-num_t_1);
        else
            len_2 = len_2 + 1;
            class_2_dual(len_2, 1) = data_t_2(1, i-num_t_1);
            class_2_dual(len_2, 2) = data_t_2(2, i-num_t_1);
        end
    end
end

scatter(data_t_1(1,:), data_t_1(2,:), 'b', '*');
hold on
scatter(data_t_2(1,:), data_t_2(2,:), 'r', '+');
hold off
legend('class 1', 'class 2');
title('Pre-labelled Test Data');

subplot(1,2,1)
scatter(data_1(1,:), data_1(2,:), 'b', '*');
hold on
scatter(data_2(1,:), data_2(2,:), 'r', '+');
hold on
plot(x, y_prim_1, 'k');
hold on
plot(x, y_prim_2, 'k', 'linestyle' ,'-.');
hold off
legend('class 1', 'class 2');
title('Twin SVM Prim Problem - train');

subplot(1,2,2)
scatter(data_1(1,:), data_1(2,:), 'b', '*');
hold on
scatter(data_2(1,:), data_2(2,:), 'r', '+');
hold on
plot(x, y_dual_1, 'k');
hold on
plot(x, y_dual_2, 'k', 'linestyle' ,'-.');
hold off
legend('class 1', 'class 2');
title('Twin SVM Dual Problem - train');

subplot(1,2,1)
scatter(class_1(:,1), class_1(:,2), 'b', '*');
hold on
scatter(class_2(:,1), class_2(:,2), 'r', '+');
hold on
plot(x, y_prim_1, 'k');
hold on
plot(x, y_prim_2, 'k', 'linestyle' ,'-.');
hold off
legend('class 1', 'class 2');
title('Twin SVM Prim Problem - test result');

subplot(1,2,2)
scatter(class_1_dual(:,1), class_1_dual(:,2), 'b', '*');
hold on
scatter(class_2_dual(:,1), class_2_dual(:,2), 'r', '+');
hold on
plot(x, y_dual_1, 'k');
hold on
plot(x, y_dual_2, 'k', 'linestyle' ,'-.');
hold off
legend('class 1', 'class 2');
title('Twin SVM Dual Problem - test result');