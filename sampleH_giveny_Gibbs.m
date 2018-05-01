function H_samp = sampleH_giveny_Gibbs(y, H_ref, X, U, region_classifier, model, params)
% does sampling by computing the probability of a new random state.
% does not care about the current state/configuration of the hidden unit.
% not recommended.

    K = length(y);
    [W_alpha, W_gamma, ~] = unzip_params(params);
    [~, post] = potential_xh(region_classifier, X, H_ref, model, []);    
    if isvector(H_ref)
        H_ref = labelVec2onehotMat(H_ref, K);
    end
    
    H_samp = H_ref;
	for j = randperm(size(H_samp, 2))
        H_aux       = H_samp;           
        H_aux(:,j)  = perturb(H_samp(:,j));

        % potentials with the perturbed region labels (H_aux)
        pot_aux     = computePotentials(H_aux);             
        % potentials for all possible values of region
        pot_all     = zeros(1,K); 
        H_all       = H_samp;
        for l=1:K
            H_all(:,j) = labelVec2onehotMat(l, K);
            pot_all(l) = computePotentials(H_all);
        end
        % ratio of 'perturbed' potentials to 'all';
        tau     = exp(pot_aux) ./ (sum(exp(pot_all)));
        flag    = min(1, tau) > rand; 

        if flag 
            H_samp = H_aux;
        end
    end
    
    function pot = computePotentials(H)
    % potentials with the perturbed region labels (H_aux)
        I_a     = potential_xh(region_classifier, X, H, model, post);
        I_s     = potential_hh_v2(H, U, W_alpha);
        I_hy    = potential_hy(y, H, W_gamma);
        pot     = (I_a + I_s + I_hy);
    end
end

function h = perturb(h)
    [~, idx] = max(rand(length(h),1));
    h = ones(length(h),1) * -1;
    h(idx) = 1;
end

% function h = perturb(h)
% 
%     rng(1234); 
% 
%     mask = 2 * (rand(length(h),1) < 0.5 ) - 1;
%     h = h .* mask;
% 
% end