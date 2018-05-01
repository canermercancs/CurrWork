function [I_a, post] = potential_xh_svm(X, H, classifier, post)
% % % % % % % % % % % % % % % % % % % % % % % % % % % 
% \Phi_s(h,x)
% computes association between a region and its label given SVM classifier
% 
% R is #of regions(instances), D is #of features, K is #of class labels.
% inputs:
%   X denotes feature matrix, is R by D; row corresponds to a region feature vector. 
%   H is R by C matrix, row corresponds to region label vector, one-hot decoded; all -1, except for one.
%   model is a pretrained SVM model.
% outputs:
%   I_a: will store the association potential for X and H.
% 
% % % % % % % % % % % % % % % % % % % % % % % % % % % 

    if ~exist('post', 'var')
        post = [];
    end

    if isvector(H)
        h = H;
    else
        h = onehotMat2labelVec(H);
    end  

    if isempty(post)
        [prds,~,~,post] = predict(classifier, X);   
    end
    
    % get the posterior probs from the classifier.
    h_idx = sub2ind(size(post), 1:size(post,1), h);
    I_a = sum(log(post(h_idx)));

end