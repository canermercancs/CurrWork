function [yc, Hc, yf, Hf] = inference(X, U, params, model, K, CD_step, gibbs_walk, clsf_type)


    [~, Hc] = max(model(X', 'UseGPU', 'no'));
%     yc = 2*randi([0 1], [K 1])-1;
%     yc = walk_on_y(gibbs_walk, Hc, yc, params); 
    yc = -1*ones(K,1);
    yc(unique(Hc)) = 1;
%     yc = walk_on_y(gibbs_walk, Hc, yc, params); 
     
    yf = yc; 
    Hf = Hc;
    for c=1:CD_step
        Hf = walk_on_H(gibbs_walk, yf, Hf, X, U, clsf_type, model, params);
        yf = walk_on_y(gibbs_walk, Hf, yf, params);
    end
end


% function y = walk_on_y(gibbs_walk, H, y, params)
%     
%     %%# sample from visible units
%     y_all = [y zeros(length(y), gibbs_walk)];
%     for k=1:gibbs_walk   
%         y_all(:,k+1) = sampley_givenH(H, y_all(:,k), params);
%     end
%     y = y_all(:,end);
% end
function y = walk_on_y(num_walks, H, y, params)
    
    %%# sample from visible units
    for k=1:num_walks   
        y(:,end+1) = sampley_givenH(H, y(:,end), params);
    end
    % take the last N of the samples and take their mode.
    y = mode(y(:,min(ceil(size(y,2)/2),10):size(y,2)), 2);
end

% function H = walk_on_H(gibbs_walk, y, H, X, U, clsf_type, model, params)
% 
%     H_all = cell(1, gibbs_walk+1);
%     H_all{1} = H;
%     %%# sample from hidden units
%     for k=1:gibbs_walk
%         aux = sampleH_giveny(y, H_all{k}, X, U, clsf_type, model, params);
%         H_all{k+1} = onehotMat2labelVec(aux);
%     end    
%     H = H_all{end}; 
% end
function H = walk_on_H(num_walks, y, H, X, U, clsf_type, model, params)

    H = {H};
    %%# sample from hidden units
    for k=1:num_walks
        H{end+1} = sampleH_giveny(y, H{end}, X, U, clsf_type, model, params);
    end    
    % take the last N of the samples and take their mode.
    H = cell2mat(cellfun(@onehotMat2labelVec, H, 'UniformOutput', false)');
    H = mode(H( min(ceil(size(H,1)/2),5) : size(H,1), :));
end
    
% function [y, H] = inference(X, U, params, model, K, CD_step, gibbs_walk, clsf_type)


%     [~, H_init] = max(model(X', 'UseGPU', 'no'));
    
%     Hfree = {H_init}; 
%     %Hfree = sampleH_giveny(yclamped, H_recon, X, U, clsf_type, model, param);
% %     Hfree = labelVec2onehotMat(randi(K, [1,R]), K);
% 
% %     yfree = 2*(rand(K,1) > 0.5) -1;
%     yfree = -1 .* ones(K, 1);
%     yfree(unique(H_init)) = 1;
% 
%     for j = 1:CD_step
%         %%# sample from visible units
%         for k=1:gibbs_walk   
%             yfree(:,end+1) = sampley_givenH(Hfree{1}, yfree(:,end), param);
%         end
%         % take the last N of the samples and take their mode.
%         yfree = mode(yfree(:,min((size(yfree,2)/2),10):size(yfree,2)), 2);
% 
%         %%# sample from hidden units
%         for k=1:gibbs_walk
%             Hfree{end+1} = sampleH_giveny(yfree, Hfree{end}, X, U, 'nnet', model, param);
%         end    
%         % take the last N of the samples and take their mode.
%         Hfree = cell2mat(cellfun(@onehotMat2labelVec, Hfree, 'UniformOutput', false)');
%         Hfree = mode(Hfree( min(ceil(size(Hfree,1)/2),5) : size(Hfree,1), :));
%         Hfree = {labelVec2onehotMat(Hfree, K)};
%     end                
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Hfree = Hfree{1};

%     [~, H_init] = max(model(X', 'UseGPU', 'no'));
%     Hfree       = {H_init};
%     yfree       = -1 .* ones(K, 1);
%     yfree(unique(H_init)) = 1;
%     
%     %%# sample from hidden units
%     for k=1:gibbs_walk
%         Hfree{end+1} = sampleH_giveny(yfree, Hfree{end}, X, U, clsf_type, model, params);
%     end    
%     Hfree = Hfree(end);
% 
%     %# run the markov chain for CD_step times (generally 1 is enough)
%     for j = 1:CD_step        
%         
%         %%# sample from visible units
%         for k=1:gibbs_walk   
%             yfree(:,end+1) = sampley_givenH(Hfree{:}, yfree(:,end), params);
%         end
%         % take the last N of the samples and take their mode.
%         yfree = yfree(:,end);
%         
%         %%# sample from hidden units
%         for k=1:gibbs_walk
%             Hfree{end+1} = sampleH_giveny(yfree, Hfree{end}, X, U, clsf_type, model, params);
%         end    
%         Hfree = Hfree(end);
%     end       
%     Hfree = Hfree{:};

    
    
%     [~, H_init] = max(model(X', 'UseGPU', 'no'));
%     Hfree       = {H_init};
%     yfree       = -1 .* ones(K, 1);
%     yfree(unique(H_init)) = 1;    
%         
%     %%# sample from visible units
%     for k=1:gibbs_walk   
%         yfree(:,end+1) = sampley_givenH(Hfree{:}, yfree(:,end), params);
%     end
%     % take the last N of the samples and take their mode.
%     yfree = yfree(:,end);
% 
%     %# run the markov chain for CD_step times (generally 1 is enough)
%     for j = 1:CD_step        
%         
%         %%# sample from hidden units
%         for k=1:gibbs_walk
%             Hfree{end+1} = sampleH_giveny(yfree, Hfree{end}, X, U, clsf_type, model, params);
%         end    
%         Hfree = Hfree(end);
%         
%         %%# sample from visible units
%         for k=1:gibbs_walk   
%             yfree(:,end+1) = sampley_givenH(Hfree{:}, yfree(:,end), params);
%         end
%         % take the last N of the samples and take their mode.
%         yfree = yfree(:,end);
%         
%     end       
%     Hfree = Hfree{:};

% end

