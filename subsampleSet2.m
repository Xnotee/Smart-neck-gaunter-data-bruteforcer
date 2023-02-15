function [F_train,labels_train,F_test,labels_test,F_dev,labels_dev] = subsampleSet2(F_train,labels_train,F_test,labels_test,F_dev,labels_dev)
    

i1 = find(labels_train == 1);
i2 = find(labels_train == -1); % non-swallow moments
i2 = i2(randperm(length(i2))); % randomize index order
i2 = i2(1:length(i1)); % take same N as swallow frames
i_train = [i1;i2]; % combine indices

i1 = find(labels_test == 1);
i2 = find(labels_test == -1); % non-swallow moments
i2 = i2(randperm(length(i2))); % randomize index order
i2 = i2(1:length(i1)); % take same N as swallow frames
i_test = [i1;i2]; % combine indices

i1 = find(labels_dev == 1);
i2 = find(labels_dev == -1); % non-swallow moments
i2 = i2(randperm(length(i2))); % randomize index order
i2 = i2(1:length(i1)); % take same N as swallow frames
i_dev = [i1;i2]; % combine indices

F_train = F_train(i_train,:);
labels_train = labels_train(i_train);
F_test = F_test(i_test,:);
labels_test = labels_test(i_test);
F_dev = F_dev(i_dev,:);
labels_dev = labels_dev(i_dev);
