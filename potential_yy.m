function [I_y, f] = potential_yy(y, W_mu)
% % % % % % % % % % % % % % % % % % % % % % % % % % % 
% \Phi_{yy}(y)
% R is #of regions(instances), D is #of features, K is #of class labels.
% inputs: 
%   W_mu are the weight matrices; is K by K by 4.
%   y denotes image labels, is of length K at most.
% outputs:
%   I_y: will store the output
% % % % % % % % % % % % % % % % % % % % % % % % % % % 

    K = length(y);

    % all possible label pair combinations
%     indices = combnk(1:K,2);

    indices = permn(1:K, 2);
    % removing the same label indices. (1,1), (2,2), ...
    indices(indices(:,1) == indices(:,2), :) = []; 
    
    % since there are only 4 types of correlations;
    %     -> negative - negative ([-1 -1] = 1)
    %     -> negative - positive ([-1  1] = 2)
    %     -> positive - negative ([ 1 -1] = 3)
    %     -> positive - positive ([ 1  1] = 4)
    corr_idx = zeros(size(indices,1),1);
    corr_idx(all(y(indices) == repmat([-1 -1], size(indices,1), 1), 2)) = 1;
    corr_idx(all(y(indices) == repmat([-1  1], size(indices,1), 1), 2)) = 2;
    corr_idx(all(y(indices) == repmat([ 1 -1], size(indices,1), 1), 2)) = 3;
    corr_idx(all(y(indices) == repmat([ 1  1], size(indices,1), 1), 2)) = 4;
    
    % find the weights corresponding to the label pairs (indices) and 
    % their values (corr_idx).
    idx = sub2ind(size(W_mu), indices(:,1), indices(:,2), corr_idx);
    f   = zeros(size(W_mu));
    f(idx) = 1;
    I_y = sum(W_mu(idx));
    
    
end
