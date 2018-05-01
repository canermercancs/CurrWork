function H_samp = sampleH_giveny(y, H_ref, X, U, clsf_type, model, params)
% does sampling by comparing a new random state of a hidden unit 
% to its current configuration.

    R = size(H_ref,2); % number of regions.
    K = length(y); % number of labels.
    
    [W_alpha, W_gamma, ~] = unzip_params(params);
    [~, post] = potential_xh(clsf_type, X, H_ref, model, []);    
    if isvector(H_ref)
        H_ref = labelVec2onehotMat(H_ref, length(y));
    end
        
    H_samp = H_ref;
    j_all = randperm(size(H_samp, 2));
    for j = j_all
%         for i = 1:length(y)
            H_aux       = H_samp;           
            H_aux(:,j)  = perturb(H_samp(:,j));

            % potentials with the perturbed region labels (H_aux)
            pot_aux     = computePotentials_norm(H_aux);            
            % potentials with the nonperturbed region labels (H_samp) 
            pot_samp    = computePotentials_norm(H_samp);
            % ratio of 'perturbed' potentials to 'nonperturbed';
            tau     = exp(pot_aux - pot_samp);
    %         flag    = min(1, tau) > rand; 
%             flag    = min(1, tau) > max(rand, 0.5); 
            flag    = min(1, tau) > 0.75+rand; 
            if flag   
                H_samp = H_aux;
            end
%         end
    end 
     
%     function pot = computePotentials(H)
%     % potentials with the perturbed region labels (H_aux)
%         I_a     = potential_xh(clsf_type, X, H, model, post);
%         I_s     = potential_hh_v2(H, U, W_alpha);
%         I_hy    = potential_hy(y, H, W_gamma);
%         
% %         I_a  = I_a / length(H);
% %         I_s  = I_s / (numel(U) - size(U,1));
% %         I_hy = I_hy / length(y);
%         pot  = (I_a + I_s + I_hy);
%     end

    function pot = computePotentials_norm(H)
       
    % potentials with the perturbed region labels (H_aux)
        I_a     = potential_xh(clsf_type, X, H, model, post) / R;
        I_s     = potential_hh_v2(H, U, W_alpha); % this is already sorta normalized.
        I_hy    = potential_hy(y, H, W_gamma) / K;
        
%         I_a  = I_a / length(H);
%         I_s  = I_s / (numel(U) - size(U,1));
%         I_hy = I_hy / length(y);
        pot  = (I_a + I_s + I_hy);
    end
end

function h = perturb(h)
    aux = find(h == -1);
    idx = aux(randi(length(aux)));
    h = ones(length(h),1) * -1;
    h(idx) = 1;
%     [~, idx] = max(rand(length(h),1));
%     h = ones(length(h),1) * -1;
%     h(idx) = 1;
end

% function h = perturb(h)
% 
%     rng(1234); 
% 
%     mask = 2 * (rand(length(h),1) < 0.5 ) - 1;
%     h = h .* mask;
% 
% end