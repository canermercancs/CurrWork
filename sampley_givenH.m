function y_samp = sampley_givenH(H, y_ref, params)

    [~, W_gamma, W_mu] = unzip_params(params);
%     [~, post] = potential_xh(X, H_ref, classifier, []);
    
    if isvector(H)
        H = labelVec2onehotMat(H, length(y_ref));
    end
    
    y_samp = y_ref;
    for i = randperm(length(y_samp))
        y_aux    = y_samp; 
        y_aux(i) = perturb(y_samp(i));

        % potentials with the perturbed bag labels (y_aux)
        pot_aux  = computePotentials(y_aux);
        % potentials with the nonperturbed bag labels (y_samp)
        pot_samp = computePotentials(y_samp);
        % ratio of 'perturbed' potentials to 'nonperturbed'
        tau =  exp(pot_aux - pot_samp);
        flag = min(1, tau) > rand; 

        if flag 
            y_samp = y_aux;
        end        
    end
    
    function pot = computePotentials(y)
    % potentials with the perturbed region labels (H_aux)
        I_hy = potential_hy(y, H, W_gamma);
        I_yy = potential_yy(y, W_mu);
        
%         I_hy = I_hy / length(y);
        pot  = (I_hy + I_yy);
    end
end

% might need to change since y(i) cannot be 1 if H(r,i)=0 for r=1,...,R
function y = perturb(y)
    if rand > 0.5
        y = y*-1;
    end
end