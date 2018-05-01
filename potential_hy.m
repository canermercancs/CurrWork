function [I_o, vTy] = potential_hy(y, H, W_gamma)
% % % % % % % % % % % % % % % % % % % % % % % % % % % 
% \Phi_{hy}(h,y)
% computes coherence between region label and image label
% 
% R is #of regions(instances), D is #of features, K is #of class labels.
% inputs: 
%   H denotes region labels, is K by R; row corresponds to region label ...
%       vector, one-hot decoded; all -1, except for one.
%   y denotes image labels, is of length K at most
%   G denotes the weighting parameter gamma; is scalar.
% 
% outputs:
%   I_o: will store the output
% 
% % % % % % % % % % % % % % % % % % % % % % % % % % % 
        
    % transform decimal labels into binary representation.
    if isvector(H)
        H = labelVec2onehotMat(H, length(y));
    end        

    % basically an OR operation on the rows of H.
    v = double(sum(H>0,2)>0); 
    v(v==0) = -1;
    vTy = dot(v,y);
    I_o = W_gamma * vTy;
    
end