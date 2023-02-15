function [X,labels] = splitSlidingWindow(X_train,Y_train,wl)
% function [X,labels] = splitSlidingWindow(X_train,Y_train,wl)
% 
% Splits data in X_train into fixed-length segments and labels them
% accordingly based on Y_train (+1 for swallow, -1 for non-swallow).

if nargin <3
    wl = 250; % window length (samples)
end
ws = 100; % window step size (samples)


X = zeros(round(size(X_train,1)/ws)+1,wl,size(X_train,2));
labels = zeros(round(size(X_train,1)/ws)+1,1);
j = 1;
for wloc = 1:ws:size(X_train,1)-wl+1-100
   X(j,:,:) = X_train(wloc:wloc+wl-1,:);    
   
   % Check if swallowing was identified within 1s from window onset to 1s
   % of window offset)
   if(sum(Y_train(wloc+100:wloc+wl-1+100)) == 0)
       labels(j) = -1;
   else
       labels(j) = 1;
   end   
   j = j+1;
end

labels = labels(1:j-1);
X = X(1:j-1,:,:);





