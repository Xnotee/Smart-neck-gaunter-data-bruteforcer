% A baseline detector/classifier using SVM and dev set for hyperparameter
% tuning for swallow detection from "älykauluri". 
% Requires measurements pre-processed with prepare_dataset.m

datadir = '/Users/OMISTAJA/OneDrive - TUNI.fi/kandityö/prosessoitu_mittausdata/';

load([datadir 'alldata.mat']); % Load data

X_all = X;
rng(123,'twister'); % reset random seed to enable repeatable experiments 
                    % if random numbers are used (e.g., data subsampling).

                    % Data explanation
% X = input accelerometer and gyro featrures,
%
% Y = binary signal for swallowing events ( 1= swallow, 0 = no swallow).
%     Note: the events have been marked with a button prss after the 
%     swallowing has been identified, and therefore the labels lag approx. 
%     1–3 s behind the actual measurements.
%
% chan_name = name of each sensor channel


channelname={'leuan alla','kieliluu, vasen','kieliluu, oikea','kurkunpää', 'ala vasen','ala oikea','niska','merkki'};
sensorname = {'accX','accY','accZ','gyroX','gyroY','gyroZ'};
sensors_to_use={{'accX','accY','accZ','gyroX','gyroY','gyroZ'}};
%position_to_use = {{'leuan alla'},{'kieliluu, vasen'},{'kieliluu, oikea'},{'kurkunpää'},{'ala vasen'},{'ala oikea'},{'niska'},{'kieliluu, vasen','kieliluu, oikea'}, ...
   % {'kieliluu, vasen','kieliluu, oikea','kurkunpää'},{'leuan alla','kieliluu, vasen','kieliluu, oikea'},{'leuan alla','kurkunpää'},{'leuan alla','kieliluu, vasen','kieliluu, oikea','kurkunpää', 'ala vasen','ala oikea','niska'}};
position_to_use = {{'leuan alla','kieliluu, vasen','kieliluu, oikea','kurkunpää', 'ala vasen','ala oikea','niska'}};
%rand_nums = [123,124,554,667,1111];
%rand_nums = [123,554];
rand_nums = [124,667,1111];
result_cell={};

i=0;
 for rn = rand_nums
    i=i+1;
    j=0;
    jj=0;
    result_cell{i,1} = rn;
    result_cell{i,2} = {};
    for use = position_to_use
    
       
        j=j+1;
        result_cell{i,2}{1,j} = string(use{:});
        result_cell{i,2}{2,j} = {};
        
        for use2 = sensors_to_use
        jj=jj+1;
        result_cell{i,2}{2,j}{jj,1} = string(use2{:});
     
    
        rng(rn,'twister');
        disp(rn)
        tic
                        % OPTIONS
        fs = 100;                               % Data is sampled at 100 Hz
        model_to_use = 2;                       % 1 for linear regression and 2 for SVM
        n_subj = size(X,3);                     % how many subjects in dataset
        n_dev_subj = 2;                         % how many subjects in devset
        n_test_subj = 1;                        % how many subjects in test set, usually LOSO with dataset of this size
        tune = 0;                               % optimize hyperparameters for SVM
        use_PCA = 1;                            % PCA on = 1, off 0
        use_highpass = 1;                       % when zero mean values are deleted from feature vectors, one using hp cutoff 0.1
        dev_subj = randi(n_subj,1,n_dev_subj);  % randomly pick n_dev_subj subjects for dev set
        feat_domain = 2;                        % time domain features for 1 and frequency domain for 2
        subsample_balanced_dataset = 1;         % balance classes to equal amount of samples

        % Select sensor channels of interest
        channels_to_use = (contains(chan_name,use{:}) & contains(chan_name,use2{:}));

        X = X_all(:,channels_to_use,:); 

        if (~use_highpass)
            for chan = 1:size(X,2)
                for subject = 1:size(X,3)
                    X(:,chan,subject) = X(:,chan,subject)-nanmean(X(:,chan,subject));
                end
            end
        end
        % Allocate variables for performance measures on train and test sets
        recall_train = zeros(n_subj,1);
        precision_train = zeros(n_subj,1);
        f1_train = zeros(n_subj,1);
        confmat_train = zeros(2,2,n_subj);
        recall_test = zeros(n_subj,1);
        precision_test = zeros(n_subj,1);
        f1_test = zeros(n_subj,1);
        confmat_test = zeros(2,2,n_subj);

        % Iterate through all n_subj speakers as test subjects (leave-one-subject-out; LOSO)
        for test_subject = 1:n_subj
         % Form a training set from N_of_subjects-devset-1 subjects
            X_train = zeros(size(X,1)*(size(X,3)-n_dev_subj-n_test_subj),size(X,2));
            Y_train = zeros(size(Y,1)*(size(Y,2)-n_dev_subj-n_test_subj),1);
         % Form a dev set from n_dev_subj subjects
            X_dev = zeros(size(X,1)*n_dev_subj,size(X,2));
            Y_dev = zeros(size(Y,1)*n_dev_subj,1);
            % Concatenate data from all training subjects and dev subjects into 
            % seperated long multivariate time-series (time x channel)

            loc_train = 1;
            loc_dev= 1;
            for k = setxor(1:n_subj,test_subject)
                if ~(ismember(dev_subj,k))
                X_train(loc_train:loc_train+size(X,1)-1,:) = X(:,:,k);
                Y_train(loc_train:loc_train+size(Y,1)-1) = Y(:,k);
                loc_train = loc_train+size(X,1);
                else
                    X_dev(loc_dev:loc_dev+size(X,1)-1,:) = X(:,:,k);
                    Y_dev(loc_dev:loc_dev+size(Y,1)-1) = Y(:,k);
                    loc_dev = loc_dev+size(X,1);
                end
            end

            % Form a test set from the held-out subject
            X_test = X(:,:,test_subject);
            Y_test = Y(:,test_subject);
            % Remove empty time steps (marked as NaN measurements) as all subjects
            % did not have the same number of samples.

            [row,~] = find(isnan(X_train));
            X_train(row,:) = [];
            Y_train(row) = [];
            [row,~] = find(isnan(X_test));
            X_test(row,:) = [];
            Y_test(row) = [];
            [row,~] = find(isnan(X_dev));
            X_dev(row,:) = [];
            Y_dev(row) = [];

            % highpass filter the datasets
            if (use_highpass)
                X_train = highpass(X_train,0.1,fs);
                X_test = highpass(X_test,0.1,fs);
                X_dev = highpass(X_dev,0.1,fs);
            end

            % Split data to 2.5 s windows with 1 s step size
            [X_train,labels_train] = splitSlidingWindow(X_train,Y_train,250);
            [X_test,labels_test] = splitSlidingWindow(X_test,Y_test,250);
            [X_dev,labels_dev] = splitSlidingWindow(X_dev,Y_dev,250);
            % time domain features
            if (feat_domain==1)
                X_train_temp = zeros(size(X_train,1),numel(use{:})*numel(use2{:})*9);
                X_dev_temp = zeros(size(X_dev,1),numel(use{:})*numel(use2{:})*9);
                X_test_temp = zeros(size(X_test,1),numel(use{:})*numel(use2{:})*9);
                feat_vector = 0;
                idx=1;
                % extract features from X_train
                for samples = 1:size(X_train,1)
                    % calculate mean
                    for channels = 1:size(X_train,3)
                        signal = squeeze(X_train(samples,:,channels)); % current sample (time x chan)
                        feat_vector(idx) = mean(signal);
                        idx=idx+1;
                    end
                    % calculate std
                    for channels = 1:size(X_train,3)
                        signal = squeeze(X_train(samples,:,channels)); 
                        feat_vector(idx) = std(signal);
                        idx=idx+1;
                    end
                    % calculate median
                    for channels = 1:size(X_train,3)
                        signal = squeeze(X_train(samples,:,channels)); 
                        feat_vector(idx) = median(signal);
                        idx=idx+1;
                    end
                    % calculate median absolute deviation
                    for channels = 1:size(X_train,3)
                        signal = squeeze(X_train(samples,:,channels)); 
                        feat_vector(idx) = mad(signal,1);
                        idx=idx+1;
                    end
                    % calculate kurtosis
                    for channels = 1:size(X_train,3)
                        signal = squeeze(X_train(samples,:,channels)); 
                        feat_vector(idx) = kurtosis(signal);
                        idx=idx+1;
                    end
                    % calculate skewness
                     for channels = 1:size(X_train,3)
                        signal = squeeze(X_train(samples,:,channels));
                        feat_vector(idx) = skewness(signal);
                        idx=idx+1;
                     end
                    % calculate min
                    for channels = 1:size(X_train,3)
                        signal = squeeze(X_train(samples,:,channels)); 
                        feat_vector(idx) = min(signal);
                        idx=idx+1;
                    end
                    % calculate max
                    for channels = 1:size(X_train,3)
                        signal = squeeze(X_train(samples,:,channels)); 
                        feat_vector(idx) = max(signal);
                        idx=idx+1;
                    end
                    % difference of maxinum and mininum
                    for channels = 1:size(X_train,3)
                        signal = squeeze(X_train(samples,:,channels)); 
                        feat_vector(idx) = max(signal)-min(signal);
                        idx=idx+1;
                    end
                    X_train_temp(samples,:) = feat_vector;
                    idx=1;
                end
                X_train_freq = X_train_temp;
                
                % extract features from X_dev
                for samples = 1:size(X_dev,1)
                    % calculate mean
                    for channels = 1:size(X_dev,3)
                        signal = squeeze(X_dev(samples,:,channels)); % current sample (time x chan)
                        feat_vector(idx) = mean(signal);
                        idx=idx+1;
                    end
                    % calculate std
                    for channels = 1:size(X_dev,3)
                        signal = squeeze(X_dev(samples,:,channels)); 
                        feat_vector(idx) = std(signal);
                        idx=idx+1;
                    end
                    % calculate median
                    for channels = 1:size(X_dev,3)
                        signal = squeeze(X_dev(samples,:,channels)); 
                        feat_vector(idx) = median(signal);
                        idx=idx+1;
                    end
                    % calculate median absolute deviation
                    for channels = 1:size(X_dev,3)
                        signal = squeeze(X_dev(samples,:,channels)); 
                        feat_vector(idx) = mad(signal,1);
                        idx=idx+1;
                    end
                    % calculate kurtosis
                    for channels = 1:size(X_dev,3)
                        signal = squeeze(X_dev(samples,:,channels)); 
                        feat_vector(idx) = kurtosis(signal);
                        idx=idx+1;
                    end
                    % calculate skewness
                     for channels = 1:size(X_dev,3)
                        signal = squeeze(X_dev(samples,:,channels));
                        feat_vector(idx) = skewness(signal);
                        idx=idx+1;
                    end
                    % calculate min
                    for channels = 1:size(X_dev,3)
                        signal = squeeze(X_dev(samples,:,channels)); 
                        feat_vector(idx) = min(signal);
                        idx=idx+1;
                    end
                    % calculate max
                    for channels = 1:size(X_dev,3)
                        signal = squeeze(X_dev(samples,:,channels)); 
                        feat_vector(idx) = max(signal);
                        idx=idx+1;
                    end
                    % difference of maxinum and mininum
                    for channels = 1:size(X_dev,3)
                        signal = squeeze(X_dev(samples,:,channels)); 
                        feat_vector(idx) = max(signal)-min(signal);
                        idx=idx+1;
                    end
                    X_dev_temp(samples,:) = feat_vector;
                    idx=1;
                end
                X_dev_freq = X_dev_temp;
                % extract features from X_test
                for samples = 1:size(X_test,1)
                    % calculate mean
                    for channels = 1:size(X_test,3)
                        signal = squeeze(X_test(samples,:,channels)); % current sample (time x chan)
                        feat_vector(idx) = mean(signal);
                        idx=idx+1;
                    end
                    % calculate std
                    for channels = 1:size(X_test,3)
                        signal = squeeze(X_test(samples,:,channels)); 
                        feat_vector(idx) = std(signal);
                        idx=idx+1;
                    end
                    % calculate median
                    for channels = 1:size(X_test,3)
                        signal = squeeze(X_test(samples,:,channels)); 
                        feat_vector(idx) = median(signal);
                        idx=idx+1;
                    end
                    % calculate median absolute deviation
                    for channels = 1:size(X_test,3)
                        signal = squeeze(X_test(samples,:,channels)); 
                        feat_vector(idx) = mad(signal,1);
                        idx=idx+1;
                    end
                    % calculate kurtosis
                    for channels = 1:size(X_test,3)
                        signal = squeeze(X_test(samples,:,channels)); 
                        feat_vector(idx) = kurtosis(signal);
                        idx=idx+1;
                    end
                    % calculate skewness
                    for channels = 1:size(X_test,3)
                        signal = squeeze(X_test(samples,:,channels));
                        feat_vector(idx) = skewness(signal);
                        idx=idx+1;
                     end
                    % calculate min
                    for channels = 1:size(X_test,3)
                        signal = squeeze(X_test(samples,:,channels)); 
                        feat_vector(idx) = min(signal);
                        idx=idx+1;
                    end
                    % calculate max
                    for channels = 1:size(X_test,3)
                        signal = squeeze(X_test(samples,:,channels)); 
                        feat_vector(idx) = max(signal);
                        idx=idx+1;
                    end
                    % difference of maxinum and mininum
                    for channels = 1:size(X_test,3)
                        signal = squeeze(X_test(samples,:,channels)); 
                        feat_vector(idx) = max(signal)-min(signal);
                        idx=idx+1;
                    end
                X_test_temp(samples,:) = feat_vector;
                idx=1;
                end
                X_test_freq = X_test_temp;
            end
            
            % frequency domain
            if (feat_domain==2)
                % Get a feature vector for each training/testing sample using
                % FFT log-magnitude spectrum
                F_train = getAccFFT(X_train);
                F_test = getAccFFT(X_test);
                F_dev = getAccFFT(X_dev);   
            end
            % Use principal component analysis (PCA) to reduce data dimensionality:
            % See: https://en.wikipedia.org/wiki/Principal_component_analysis
            if(use_PCA && feat_domain==1)
                var_to_explain = 0.85; % Default: 0.85
                [coeff, score, latent, tsquared, explained] = pca(X_train_freq);
                explained = explained./sum(explained);
                cumul_explained = cumsum(explained);
                how_many_pca_coeffs = find(cumul_explained >= var_to_explain,1);
                X_train_freq = X_train_freq*coeff(:,1:how_many_pca_coeffs);
                X_test_freq = X_test_freq*coeff(:,1:how_many_pca_coeffs);
                X_dev_freq = X_dev_freq*coeff(:,1:how_many_pca_coeffs);
            end
            if(use_PCA && feat_domain==2)
                var_to_explain = 0.85; % Default: 0.85
                [coeff, score, latent, tsquared, explained] = pca(F_train);
                explained = explained./sum(explained);
                cumul_explained = cumsum(explained);
                how_many_pca_coeffs = find(cumul_explained >= var_to_explain,1);
                F_train = F_train*coeff(:,1:how_many_pca_coeffs);
                F_test = F_test*coeff(:,1:how_many_pca_coeffs);
                F_dev = F_dev*coeff(:,1:how_many_pca_coeffs);
            end

            % Sample an equal number of swallow and non-swallow time frames to
            % avoid classifier predicting everything as dominant class (non-swallow).
            
            if(subsample_balanced_dataset)
                if (feat_domain==1)
                    [X_train_freq,labels_train,X_test_freq,labels_test,X_dev_freq,labels_dev] = subsampleSet2(X_train_freq,labels_train,X_test_freq,labels_test,X_dev_freq,labels_dev);
                end
                if (feat_domain==2)
                    [F_train,labels_train,F_test,labels_test,F_dev,labels_dev] = subsampleSet2(F_train,labels_train,F_test,labels_test,F_dev,labels_dev);
                end
            end
            
            if ((tune && model_to_use==2) && (test_subject == 1) && feat_domain==1)
                [f1_scores, table] = tune_svm_model(X_train_freq,labels_train,X_dev_freq,labels_dev);
                tune = 0;
            end
            if ((tune && model_to_use==2) && (test_subject == 1) && feat_domain==2)
                [f1_scores, table] = tune_svm_model(F_train,labels_train,F_dev,labels_dev);
                tune = 0;
            end
            if (model_to_use==1)
                if ( feat_domain==1)
                    X_train_freq = [X_train_freq;X_dev_freq];
                    labels_train=[labels_train;labels_dev];
                    X_train_freq(isnan(X_train_freq))=0;
                    X_test_freq(isnan(X_test_freq))=0;
                    W = pinv(X_train_freq)*labels_train;
                    pred_train = X_train_freq*W;
                    [recall_train(test_subject),precision_train(test_subject),f1_train(test_subject),confmat_train(:,:,test_subject),best_thr] = evaluate_detector(labels_train,pred_train);
                    fprintf('Train fold %d/%d: f1-score: %0.2f%%.\n',test_subject,n_subj,f1_train(test_subject)*100); 
                    pred_test = X_test_freq*W;
                    [recall_test(test_subject),precision_test(test_subject),f1_test(test_subject),confmat_test(:,:,test_subject),dah] = evaluate_detector(labels_test,pred_test,best_thr);
                    fprintf('Test fold %d/%d: f1-score: %0.2f%%.\n',test_subject,n_subj,f1_test(test_subject)*100);
                end
                if ( feat_domain==2)
                    F_train = [F_train;F_dev];
                    labels_train=[labels_train;labels_dev];
                    
                    W = pinv(F_train)*labels_train;
                    pred_train = F_train*W;
                    [recall_train(test_subject),precision_train(test_subject),f1_train(test_subject),confmat_train(:,:,test_subject),best_thr] = evaluate_detector(labels_train,pred_train);
                    fprintf('Train fold %d/%d: f1-score: %0.2f%%.\n',test_subject,n_subj,f1_train(test_subject)*100); 
                    pred_test = F_test*W;
                    [recall_test(test_subject),precision_test(test_subject),f1_test(test_subject),confmat_test(:,:,test_subject),dah] = evaluate_detector(labels_test,pred_test,best_thr);
                    fprintf('Test fold %d/%d: f1-score: %0.2f%%.\n',test_subject,n_subj,f1_test(test_subject)*100);
                end
            end
            if (model_to_use==2)
                if ( feat_domain==1)
                    X_train_freq = [X_train_freq;X_dev_freq];
                    labels_train=[labels_train;labels_dev];
                    SVMModel = fitcsvm(X_train_freq, labels_train, 'KernelFunction', 'linear', ...
                    'KernelScale', 0.1, 'BoxConstraint', 0.01);
                    pred_train = predict(SVMModel,X_train_freq);
                    [recall_train(test_subject),precision_train(test_subject),f1_train(test_subject),confmat_train(:,:,test_subject),best_thr] = evaluate_detector(labels_train,pred_train);
                    fprintf('Train fold %d/%d: f1-score: %0.2f%%.\n',test_subject,n_subj,f1_train(test_subject)*100); 
                    pred_test = predict(SVMModel,X_test_freq);
                    [recall_test(test_subject),precision_test(test_subject),f1_test(test_subject),confmat_test(:,:,test_subject),dah] = evaluate_detector(labels_test,pred_test,best_thr);
                    fprintf('Test fold %d/%d: f1-score: %0.2f%%.\n',test_subject,n_subj,f1_test(test_subject)*100);
                end
                if ( feat_domain==2)
                    F_train = [F_train;F_dev];
                    labels_train=[labels_train;labels_dev];
                    SVMModel = fitcsvm(F_train, labels_train, 'KernelFunction', 'gaussian', ...
                    'KernelScale', 900, 'BoxConstraint', 2.1);
                    pred_train = predict(SVMModel,F_train);
                    [recall_train(test_subject),precision_train(test_subject),f1_train(test_subject),confmat_train(:,:,test_subject),best_thr] = evaluate_detector(labels_train,pred_train);
                    fprintf('Train fold %d/%d: f1-score: %0.2f%%.\n',test_subject,n_subj,f1_train(test_subject)*100); 
                    pred_test = predict(SVMModel,F_test);
                    [recall_test(test_subject),precision_test(test_subject),f1_test(test_subject),confmat_test(:,:,test_subject),dah] = evaluate_detector(labels_test,pred_test,best_thr);
                    fprintf('Test fold %d/%d: f1-score: %0.2f%%.\n',test_subject,n_subj,f1_test(test_subject)*100);
                end
            end
    
        end

            % Overall scoring across all LOSO folds.
        disp(rn)
        disp(use{:})
        disp(use2{:})
        fprintf('-------------\nOverall scores:\n');
        fprintf('Train f1: %0.2f%% (+- %0.2f%%).\n',mean(f1_train)*100,std(f1_train)*100);
        fprintf('Test f1: %0.2f%% (+- %0.2f%%).\n',mean(f1_test)*100,std(f1_test)*100);
        toc
        result_cell{i,2}{2,j}{jj,2}=[mean(f1_train)*100;std(f1_train)*100;mean(f1_test)*100;std(f1_test)*100];
        result_cell{i,2}{2,j}{jj,3}=toc;
        end
        jj=0;
    end
    
end