function [params, y_orig, H_orig, H_clampeds, H_frees, y_frees] = trainHCRF_epoch2(curr_epoch, ...
            train_bags, train_targets, train_proxy, train_ccmap, train_cases, ...
            H_init, params, model, hyperparams, props)

% let everything from now on be random. Important for sampling! 
rng('shuffle'); 

% hyperparameters...
eta         = hyperparams.etas(curr_epoch);          % 1./(1:num_epochs)    % learning rate
gauss_prior = hyperparams.gauss_prior;   % needs to be tuned?
CD_step     = hyperparams.CD_steps(curr_epoch);      % number of CD steps
gibbs_walk  = hyperparams.gibbs_walk;    % number of sampling steps
clsf_type   = hyperparams.clsf_type;     % 'svm' or 'nnet'
num_batch   = props.num_batch;     % how big will be the batch?

K           = size(train_targets, 1);
%%# initial y are null. 
y_orig      = zeros(size(train_targets));
H_orig      = cell(size(train_bags)); 
H_clampeds  = cell(size(train_bags)); 
y_frees     = zeros(size(train_targets));
H_frees     = cell(size(train_bags)); 

% set the parameters of the HCRF
[W_alpha, W_gamma, W_mu] = unzip_params(params);

% initialize the sufficient stats to zeros/null.
f_hy_clamped = zeros(1, num_batch);
f_hy_free    = zeros(1, num_batch);
f_hh_clamped = cell(1, num_batch);
f_hh_free    = cell(1, num_batch);
f_yy_clamped = cell(1, num_batch);
f_yy_free    = cell(1, num_batch);

% randomly select batch data
numbags     = length(train_bags);
bag_batch   = randperm(numbags);
bag_batch   = bag_batch(1:num_batch);
for i = 1:num_batch

    s = sprintf('Processing batch #%i out of %i', i, num_batch);
    fprintf(s);
    
    bag_no      = bag_batch(i);         % the randomly selected bag
    H           = H_init{bag_no};     % initial(or updated) hidden states
    y           = train_targets(:, bag_no); 
    X           = train_bags{bag_no};      
    U           = train_proxy{bag_no};  % neighborhood matrix.
    R           = size(X,1);    
%     cc          = train_cc{bag_no};
%     mycase      = train_cases(bag_no);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% CONTRASTIVE DIVERGENCE
    %%# sample N times, and then get the most sampled ones among those.
    %%# clamped state 
    yclamped = y;
    Hclamped = walk_on_H(gibbs_walk, yclamped, H, X, U, clsf_type, model, params);
    %%# free state         
    Hfree = H; 
    yfree = yclamped;  
    %# run the markov chain for CD_step times (generally 1 is enough)
    for j = 1:CD_step        
        %%# sample from visible units
        yfree = walk_on_y(gibbs_walk, Hfree, yfree, params);
        %%# sample from hidden units
        Hfree = walk_on_H(gibbs_walk, yfree, Hfree, X, U, clsf_type, model, params);
    end  
   	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%# compute p(h|y,x;\theta) for each h to find H_recons{bag_no}        
%     H_recons{bag_no} = updateRegionLabels(yclamped, X, U, cc, ...
%                 clsf_type, model, params, K, mycase); 
%     H_recons{bag_no}    = onehotMat2labelVec(Hclamped); % These return bad results.
%     H_recons{bag_no}   = Hfree;
%     H_clamped{bag_no}  = doCorrection(Hclamped); % approximation to updateRegionLabels

    y_orig(:,bag_no)    = y;
    H_orig{bag_no}      = H;
    H_clampeds{bag_no}  = Hclamped; %walk_on_H(gibbs_walk, yclamped, H, X, U, clsf_type, model, params);
    H_frees{bag_no}     = Hfree;
    y_frees(:,bag_no)   = yfree; 
    
    [~, f_hy_clamped(i)]    = potential_hy(yclamped, H, W_gamma);
    [~, f_hy_free(i)]       = potential_hy(yfree, Hfree, W_gamma);
    [~, f_hh_clamped{i}]    = potential_hh_v2(H, U, W_alpha);
    [~, f_hh_free{i}]       = potential_hh_v2(Hfree, U, W_alpha);
    [~, f_yy_clamped{i}]    = potential_yy(yclamped, W_mu);
    [~, f_yy_free{i}]       = potential_yy(yfree, W_mu);
  

    % delete the current command window line.
    fprintf(repmat('\b', 1, numel(s)));    
end 

% COMPUTING THE GRADIENTS0
% delta_gamma 
delta_gamma = mean(f_hy_clamped) - mean(f_hy_free) - (gauss_prior^-2) .* W_gamma;
% delta_alpha
aux_clamped = cat(3, f_hh_clamped{:});
aux_free    = cat(3, f_hh_free{:});
delta_alpha = mean(aux_clamped,3) - mean(aux_free,3) - (gauss_prior^-2) .* W_alpha;
% delta_mu
aux_clamped = cat(4, f_yy_clamped{:});
aux_free    = cat(4, f_yy_free{:});
delta_mu    = mean(aux_clamped,4) - mean(aux_free,4) - (gauss_prior^-2) .* W_mu;

% % COMPUTING THE GRADIENTS
% % delta_gamma 
% delta_gamma = mean(f_hy_free) - mean(f_hy_clamped) - (gauss_prior^-2) .* W_gamma;
% % delta_alpha
% aux_clamped = cat(3, f_hh_clamped{:});
% aux_free    = cat(3, f_hh_free{:});
% delta_alpha = mean(aux_free,3) - mean(aux_clamped,3) - (gauss_prior^-2) .* W_alpha;
% % delta_mu
% aux_clamped = cat(4, f_yy_clamped{:});
% aux_free    = cat(4, f_yy_free{:});
% delta_mu    = mean(aux_free,4) - mean(aux_clamped,4) - (gauss_prior^-2) .* W_mu;


% UPDATING THE PARAMETERS
W_gamma     = W_gamma + eta .* delta_gamma;
W_alpha     = W_alpha + eta .* delta_alpha;
W_mu        = W_mu    + eta .* delta_mu;    
params      = zip_params(W_alpha, W_gamma, W_mu);

end

function y = walk_on_y(num_walks, H, y, params)
    
    %%# sample from visible units
    y_all = [y zeros(length(y), num_walks)];
    for k=1:num_walks   
        y_all(:,k+1) = sampley_givenH(H, y_all(:,k), params);
    end
    y = y_all(:,end);
end
% function y = walk_on_y(num_walks, H, y, params)
%     
%     %%# sample from visible units
%     for k=1:num_walks   
%         y(:,end+1) = sampley_givenH(H, y(:,end), params);
%     end
%     % take the last N of the samples and take their mode.
%     y = mode(y(:,min((size(y,2)/2),10):size(y,2)), 2);
% end

function H = walk_on_H(num_walks, y, H, X, U, clsf_type, model, params)

    H_all = cell(1, num_walks+1);
    H_all{1} = H;
    %%# sample from hidden units
    for k=1:num_walks
        aux = sampleH_giveny(y, H_all{k}, X, U, clsf_type, model, params);
        H_all{k+1} = onehotMat2labelVec(aux);
    end    
    H = H_all{end}; 
end
% function H = walk_on_H(num_walks, y, H, X, U, clsf_type, model, params)
% 
%     H = {H};
%     %%# sample from hidden units
%     for k=1:num_walks
%         H{end+1} = sampleH_giveny(y, H{end}, X, U, clsf_type, model, params);
%     end    
%     % take the last N of the samples and take their mode.
%     H = cell2mat(cellfun(@onehotMat2labelVec, H, 'UniformOutput', false)');
%     H = mode(H( min(ceil(size(H,1)/2),5) : size(H,1), :));
% end

% function h = doCorrection(y, h)
% 
%     % % doing correction when there are predicted labels that are more
%     % % severe than the correct most severe label for the case.
%     mst_svr = find(y==1,1,'last');
%     h(h > mst_svr) = mst_svr;         
%     % % doing correction when there are predicted labels that are less
%     % % severe than the correct least severe label for the case.
%     lst_svr = find(y==1,1,'first'); 
%     h(h < lst_svr) = lst_svr;
%     % % doing correction when all predicted labels do not cover all the
%     % % correct labels.
% end