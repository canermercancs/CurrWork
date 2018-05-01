function [I_a, post] = potential_xh(region_classifier, X, H, model, post)
% % % % % % % % % % % % % % % % % % % % % % % % % % % 
% \Phi_s(h,x)
% computes association between a region and its label
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


    if strcmp(region_classifier, 'nnet')
        [I_a, post] = potential_xh_nnet(X', H, model, post);
    elseif strcmp(region_classifier, 'svm')
        [I_a, post] = potential_xh_svm(X, H, model, post);
    else
        error('wrong classification type') 
    end

end