function [recall,precision,f1,confmat,best_thr] = evaluate_detector_optimal(target,pred,thresholds)
% function [recall,precision,f1,confmat,best_thr] = evaluate_detector(target,pred,thresholds)
%
% Inputs: 
%   target       = target labels (-1, 1)
%   pred         = predicted scalar values
%   thresholds   = set of detection thresholds to test (optional).
%                  Default:  [0.00:0.001:0.1 0.2 0.3 0.4 0.5 0.75 1].
%
% Outputs:
%   recall      = recall 
%   precision   = precision
%   f1          = harmonic mean of precision and recall (primary metric)
%   confmat     = confusion matrix of the classes
%   best_thr    = optimal detection threshold in terms of f1 
%                 (among values in "thresholds").

pred_orig = pred;

if nargin <3
    thrvals = [0.00:0.001:0.1 0.2 0.3 0.4 0.5 0.75 1];
else
    thrvals = thresholds;
end

f1 = zeros(length(thrvals),1);
recall = zeros(length(thrvals),1);
precision = zeros(length(thrvals),1);
confmat = zeros(2,2,length(thrvals));

% Convert target labels to {0,1} for convenience of indexing
target(target == -1) = 0;

x = 1;
% Iterate across thresholds
for thr = thrvals
    pred = pred_orig;
    
    % Threshold predictions to {0,1} based on current threshold
    pred(pred >= thr) = 1;
    pred(pred < thr) = 0;

    % Calculate confusion matrix    
    for k = 1:length(pred)
       confmat(target(k)+1,pred(k)+1,x) = confmat(target(k)+1,pred(k)+1,x)+1;
    end

    % Normalize for recall (proportion of cases detected)
    confmat_recall = confmat(:,:,x)./repmat(eps+sum(confmat(:,:,x),2),1,2);

    % Normalize for precision (proportion of true positives of all positives)
    confmat_precision = confmat(:,:,x)./repmat(eps+sum(confmat(:,:,x),1),2,1);

    % Overall recall and precision as their mean across individual
    % classes (diagonal of the confusion matrices). 
    recall(x) = mean(diag(confmat_recall));
    precision(x) = mean(diag(confmat_precision));
  
    % F1 score as harmonic mean of precision and recall
    f1(x) = 2*recall(x)*precision(x)/(precision(x)+recall(x));
    x = x+1;
end



% Select best threshold based on f1 score
[f1,i] = max(f1);

% Determine outputs based on the optimal threshold
best_thr = thrvals(i);
recall = recall(i);
precision = precision(i);
confmat = confmat(:,:,i);