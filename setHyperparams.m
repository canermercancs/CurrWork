
% update the filenname for each different setting!
props.filename              = 'aux_lclNOtrain_v1_2'; %'aux_2(byH)_v5'; %'NOTtrainClassifier_plusHclmp_v#4'; % 'aux';
props.getPast               = false; % if you want to continue from a previous check point.
props.getPastHyperparams    = false; % to use the previous hyperparams, make this true. to override; make this true.
props.getPastProps          = false; % to use the previous hyperparams, make this true. to override; make this true.
props.inferencePerEpoch     = 10;
props.num_batch             = length(bags) * .5;                                % number of finetuning epochs.        
props.trainClassifier       = false;

%%# hyperparameters
hyperparams.num_epochs   = 5000;%1e+4;                               % how big will be the batch?
hyperparams.num_fineTune = 1000;                                                     % # of EM iterations (num of epochs)
hyperparams.etas        = 1e-1 ./ (1:0.02:((hyperparams.num_epochs)/50)+1);      % ones(1,1e+4)*0.1    % learning rate
hyperparams.etas        = hyperparams.etas(1:end-1);
hyperparams.etas        = [hyperparams.etas, 5e-2 ./ (1:0.01:((hyperparams.num_fineTune)/100+1))];      % ones(1,1e+4)*0.1    % learning rate
hyperparams.gauss_prior = 3;                                                % needs to be tuned?
hyperparams.CD_steps    = 1*ones(1, hyperparams.num_epochs);                % number of CD steps 
hyperparams.CD_steps    = [hyperparams.CD_steps 1*ones(1, hyperparams.num_fineTune)];
hyperparams.gibbs_walk  = 50;                                               % number of sampling steps
hyperparams.CD_steps_inference      = 4;
hyperparams.gibbs_walk_inference    = 100;
% which local classifier to use (nnet or SVM) is based on hyperparams.clsf_type
hyperparams.clsf_type   = 'nnet'; % 'svm' or 'nnet'        
hyperparams.nnet_params.hidden = [20];
hyperparams.nnet_params.useGPU = false;
hyperparams.SVM.kernel = 'linear'; % 'rbf', 'linear'
% The region(polygon) proximity/neighborhood selection
proxy_type = 1;

%%
if proxy_type == 1
    proxy = Proximities_CCmin;
else
    proxy = Proximities;
end

%%
if props.trainClassifier
    lclnote = 'local classifier is trained at every epoch now!';
else
    lclnote = 'local classifier is not trained';
end
props.notes2self = { lclnote, ...
               'W_gamma learning rate is eta', ...
               'potential_hh_v2 uses MEAN not SUM', ...
               'sampleH_giveny flips w.r.t 0.75+rand, not rand', ...
               ['trainHCRF_epoch2 is used; sufficient statistics for ' ...
                'clamped states are computed from Hclassifier']};
% , ...                
% 'local classifier is trained at every epoch now!' ...
  