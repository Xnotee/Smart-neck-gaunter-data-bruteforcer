function [params, table] = tune_svm_model(X_train, labels_train, dev_set, labels_dev, kernels, kernelscales, boxconstraints)
% function params = tune_svm_model ()
%
% Inputs: 
%   X_train         = training subjects
%   labels_train    = ground truth of training samples
%   dev_set         = vector of subjects' samples used for hyperparameter tuning
%   labels_dev    = ground truth of dev_set samples
%   kernels         = kernels to test
%                       Default:  ['gaussian', 'linear', 'polynomial']
%   kernelscales    = scales of kernel to test
%                       Default: [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, auto]
%   boxcontstraint  = boxcontraints to test
%                       Default: [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
% Outputs:
%   params          = vector of optimal kernel, kernel scaling and boxcontstraint
%   table           = matrix of runs' f1 scores
if nargin < 7
    boxc = [0.001 0.01 0.1 1 10 100 1000 10000];
    %boxc = [1.7 1.8 1.9 2 2.1 2.2 2.3 2.4 2.5];
else
    boxc = boxconstraints;
end
if nargin < 6
    kscl = [0.001 0.01, 0.1 1 10 100 1000 10000];
    %kscl = [865 870 880 890 900 910 920 930 940];
else
    kscl = kernelscales;
end
if nargin < 5
    krnls = ["gaussian" "linear" "polynomial"];
else
    krnls = kernels;
end
f1_scores = zeros(size(krnls,2)*size(kscl,2),size(boxc,2));
table = zeros(size(krnls,2)*(size(kscl,2))+1,size(boxc,2)+1);

krl = 0;
row = 1;
best_f1 = 0;
best_hypers = [krnls(1) kscl(1) boxc(1)  ""];
tic

% loop trough different kernels, scales and boxsc
% creates: 
% f1_scores: with only each runs score
% table: has row and column topics to spot which score is with what
% best_hypers: best hypermeter and the f1 score for each kernel
for ker =  krnls
    krl = krl + 1;
    table(row,1) = krl;
    table(row,2:end) = boxc;
    table(row+1:row+size(kscl,2),1) = kscl';
    for scale = kscl
        col = 0;
        for box = boxc
            col = col + 1;
            SVMModel = fitcsvm(X_train, labels_train, 'KernelFunction', ker, ...
                'KernelScale', scale, 'BoxConstraint', box);
            pred_dev = predict(SVMModel, dev_set);

            [~,~,f1,~,~] = evaluate_detector(labels_dev, pred_dev);
            if f1 > best_f1
                best_f1 = f1;
                if ker ~= best_hypers(1)
                    best_hypers = [best_hypers;ker scale box best_f1];
                else
                    best_hypers(krl,:) = [ker scale box best_f1];
                end
            end
            f1_scores(row,col) = f1;
            table(row+1,col+1) = f1;
            disp(row)
            disp(col)
            disp(best_hypers)
        end
        row = row + 1;
    end
end
toc
% find best hyperparameters 
M = max(f1_scores,[],'all','linear');
[best_row, best_col] = find(f1_scores == M);

params = f1_scores;
end