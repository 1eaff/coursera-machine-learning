function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
logs = zeros(length(vec) * length(vec), 3);
idx = 1;

for c_idx=1:length(vec)
    for sigma_idx=1:length(vec)
        c = vec(c_idx);
        sigma = vec(sigma_idx);
        model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
        pred = svmPredict(model, Xval);
        loss = mean(double(pred ~= yval));
        logs(idx, :) = [c, sigma, loss];
        idx = idx + 1;
    end
end

[_, idx] = min(logs(:, 3));
C = logs(idx, 1);
sigma = logs(idx, 2);

% =========================================================================

end
