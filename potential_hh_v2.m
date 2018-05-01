function [I_s, f] = potential_hh_v2(H, U, W_alpha)
% % % % % % % % % % % % % % % % % % % % % % % % % % % 
% \Phi_s(h,x)
% computes spatial association of two region labels
% 
% R is #of regions(instances), D is #of features, K is #of class labels.
% inputs:
%   H denotes region labels, is K by R; column corresponds to region label
%       vector, one-hot decoded; all -1, except for one.
%   U denotes neighborhood matrix, is R by R; U_{ij} = 1 if X_i and X_j
%       are neighbors, U_{ij} = 0 otherwise.
%   W denotes the weighting parameter of each label pair; K by K
% 
% outputs:
%   I_s: will store the spatial potential for a given sequence of H.
% 
% % % % % % % % % % % % % % % % % % % % % % % % % % % 

    % make the diagonals of U zeros. (i am not a neighbor of myself)
%     U(logical(eye(size(U)))) = 0;
    assert(sum(diag(U)) == 0);
    % if no need to compute f, only compute I_s and return.
%     returnPotentialOnly = false; 
%     if nargout < 2 
%         returnPotentialOnly = true;
%     end
    if isvector(H)
        h = H;
    else
        h = onehotMat2labelVec(H);
    end     
    
    % find neighbor regions
    [nei_x, nei_y] = find(U);       
    neighborpair_labelidx = sub2ind(size(W_alpha), h(nei_x), h(nei_y));
       
% if returnPotentialOnly 
%     I_s = sum(W_alpha(idx));
%     return
% else
    % f = reshape(histcounts(idx), size(W_alpha)); % wont work 
    % so using the following loop (almost the same speed with above)
    
    U_aux = U(U>0); % neighborhood is based on proximity; between 0 and 1.
    f = zeros(size(W_alpha));
    for u = unique(neighborpair_labelidx)
%         f(u) = sum( U_aux(neighborpair_labelidx==u) ); 
        f(u) = mean( U_aux(neighborpair_labelidx==u) ); 
    end
    I_s = sum(sum( W_alpha .* f ));
%     f = f - diag(diag(f)/2); % corrects the problem of counting the same label pairs twice by dividing the diagonal by 2.
%     f = f - diag(diag(f)/2); % corrects the problem of counting the same label pairs twice by dividing the diagonal by 2.
%     I_s = sum(sum( W_alpha .* f ));
% end
    
end