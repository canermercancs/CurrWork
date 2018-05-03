
function runHCRFonPIM()

clc, clear, warning('off', 'all');

cv = [];
addPaths
loadData
setHyperparams 
setDistance

% do your magic!
% runHCRFonCV(hyperparams, props, bags, labels, caseID, cv, CCMaps, ... 
%             proxy, proxy_type, params, model);
runHCRFonCV(hyperparams, props, bags, labels, caseID, cv, CCMaps, ... 
            proxy, proxy_type, [], model);

end

function runHCRFonCV(hyperparams, props, bags, labels, caseID, cv, CCMaps, ...
                        proxy, proxy_type, params, model)

% DATASET CROSSVALIDATION ADJUSTMENTS
cvs = 1:size(cv,2);
K = size(labels,1);
% 4 fold CV for training and testing
% parpool(2)
for curr_cv = cvs
    best_AP = 0;
    tr_cvMask = true(1,length(cvs)); % training CV mask
    tr_cvMask(curr_cv) = false; % setting current test set idx to false for training.
     
    % 3 fold CV for training and validation
    for curr_vd = cvs(tr_cvMask)
        tr_cvMask(curr_vd) = false; % setting current validation set idx to false.
        
        train_filt = sum(cv(:, tr_cvMask), 2) > 0; % train set filter.
        valid_filt = cv(:, curr_vd); % validation set filter.
                
        [curr_params, curr_model, curr_AP] = runHCRF(1, hyperparams.num_epochs, ...
                    curr_cv, curr_vd, train_filt, valid_filt, params, model); 
                           
        if curr_AP > best_AP
            best_params = curr_params; 
            best_model  = curr_model;
            best_AP     = curr_AP;
        end
        
        tr_cvMask(curr_vd) = true; % setting current validation set idx to true.
    end
    
	fprintf('Finetuning... \n');   
    trVd_filt = sum(cv(:, tr_cvMask), 2) > 0; % train+validation set filter
    test_filt = cv(:, curr_cv); % test set filter.
    runHCRF(hyperparams.num_epochs+1, hyperparams.num_fineTune, ...
        curr_cv, 0, trVd_filt, test_filt, best_params, best_model);

end


%% TRAINING
function [bestParams, bestModel, bestAP] = runHCRF(prev_epoch, num_epochs, ...
                curr_cv, curr_vd, train_filt, valid_filt, params, model)

AUX_VAR = []; % a variable to use during debug; otherwise cannot add a variable to the static workspace

params_all       = cell(1, num_epochs);
models_all       = cell(1, num_epochs);
sampling_perf    = zeros(num_epochs, 4);
infer_perf.train = zeros(num_epochs, 11); % cause there are 5 performance criteria
infer_perf.valid = zeros(num_epochs, 11); % cause there are 5 performance criteria
H_classifiers_all= cell(1, num_epochs);
H_clampeds_all   = cell(1, num_epochs);
H_frees_all      = cell(1, num_epochs);
y_frees_all      = cell(1, num_epochs);      

% set the full filename.
setFname()
% To continue from the previously saved check point
getPastSettings();


[train_bags, train_labels, train_proxy, train_ccmap, train_cases] = getDataBatch(train_filt);
[valid_bags, valid_labels, valid_proxy, valid_ccmap, valid_cases] = getDataBatch(valid_filt); %#ok<*ASGLU>

%%# initial H (hidden region labels) are determined by a neural net.
if ~exist('H_classifier', 'var')
    [H_classifier, model] = initializeHiddenVarsbyNNet(train_bags, train_labels, model, hyperparams.nnet_params);
end
%%# init params from data; by counting from bag labels and region labels.
if isempty(params)
%     params = initializeParamsFromData(H_classifier, train_labels, train_proxy);
%     params = initializeParams(K); % randomly initialize the parameters.
    params = initializeParamsSmart(train_labels);
end

time_all = zeros(1, num_epochs);
for curr_epoch = prev_epoch:prev_epoch+num_epochs-1
    time_epoch = tic; 
    
    printEpochInfo();

    % train one epoch of HCRF
    [params_next, y_target, H_classifier_next, H_clampeds, H_frees, y_frees] = ...
        trainHCRF_epoch(curr_epoch, train_bags, train_labels, ...
                        train_proxy, train_ccmap, train_cases, ...
                        H_classifier, params, model, hyperparams, props); 
      
	% looking at the results of the CD sampling...
    samp_perf = observeSampling();
                    
    % looking at the results of Inference on training and validation sets.
    [perfclsf_train, perfclsf_valid, perfsamp_train, perfsamp_valid] = observeInference(props.inferencePerEpoch);
        
    % provide epoch running time and remaining estimated time...
    printRemainingTime();    
    
    % write the model params etc. to a file at each epoch.
    writeEpoch2File();
    
    % update the params and the model (and H_classifier)
    updateParams();
    
end

[bestAP, bestIdx] = max(infer_perf.valid(:,1)); % the best model is the one with the largets validation AP
bestParams = params_all{bestIdx};
bestModel  = models_all{bestIdx};



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% HELPER FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function setFname()
    props.fullname = sprintf('hcrfPIM_(%s)_testcv(%i)_validcv(%i)_proxy(%i).mat', ...
            props.filename, curr_cv, curr_vd, proxy_type);
end

function [bags_, labels_, proxy_, maps_, cases_] = getDataBatch(filt)
% getData by the filter
    bags_   = bags(filt);
    labels_ = labels(:, filt);
    proxy_  = proxy(filt);
    maps_   = CCMaps(filt);
    cases_  = caseID(filt);
end

function getPastSettings()
    if ~exist(props.fullname, 'file')
        return
%         error('no file named %s', props.fullname);
    end       
    previous = load(props.fullname);    
    if props.getPast
        params = previous.params_all{previous.curr_epoch};
        %     params  = previous.params_next;
        model        = previous.models_all{previous.curr_epoch};
        H_classifier = previous.H_classifiers_all{previous.curr_epoch};
        %     model   = previous.model_next;
        prev_epoch = previous.curr_epoch+1;
    end      
    if props.getPastHyperparams
        % overrides current configuration of hyperparams.
        hyperparams = previous.hyperparams; 
    end  
    if props.getPastProps
        props = previous.props;
    end
end

function yfromH = getYfromH(Hs)
% labels y(s) combined from the labels of H(s).
    yfromH = -1.*ones(K, length(Hs));
    for i=1:length(Hs) 
        if ~isvector(Hs{i})
            Hs{i} = onehotMat2labelVec(Hs{i});
        end
        Hs_uniq = unique(Hs{i});
        yfromH(sub2ind(size(yfromH), Hs_uniq, repmat(i, size(Hs_uniq)) )) = 1;
    end    
end

function printEpochInfo()
    fprintf('\n@epoch: %i/%i for %s\n', curr_epoch, num_epochs, props.filename);
end
function printRemainingTime()
    time_all(curr_epoch) = toc(time_epoch);
    fprintf('estimated total remaining time; %.1f minutes \n', ...
            (sum(time_all)/curr_epoch * (num_epochs-curr_epoch)) / 60);
end

function samp_perf = observeSampling()
    y_frees_perf        = Average_precision(y_frees, y_target);
    H_classifier_perf   = Average_precision(getYfromH(H_classifier), y_target);    
    H_frees_perf        = Average_precision(getYfromH(H_frees), y_target);
    H_clampeds_perf     = Average_precision(getYfromH(H_clampeds), y_target);
    samp_perf           = [H_classifier_perf, H_clampeds_perf, y_frees_perf, H_frees_perf];
        
    fprintf('\n') 
	fprintf('@validationCV(%i) \n', curr_vd);
    fprintf('Average Precision with y from Hclassifier: %4.2f\n', H_classifier_perf);
    fprintf('Average Precision with y from yfree: %4.2f\n', y_frees_perf);
    fprintf('Average Precision with y from Hfree: %4.2f\n', H_frees_perf);
    fprintf('Average Precision with y from Hclamped: %4.2f\n', H_clampeds_perf);
end

% evaluate train and validation performances at (freq) epochs ().
function [perfclsf_train, perfclsf_valid, perfsamp_train, perfsamp_valid] = observeInference(freq)
    
    perfclsf_train = zeros(1,5); perfclsf_valid = zeros(1,5); 
    perfsamp_train = zeros(1,5); perfsamp_valid = zeros(1,5);
    if mod(curr_epoch, freq) == 0
        [perfclsf_train, perfsamp_train] = validateModel(train_bags, train_labels, train_proxy);
        [perfclsf_valid, perfsamp_valid] = validateModel(valid_bags, valid_labels, valid_proxy); 
        
        fprintf('\n') 
        fprintf('@validationCV(%i) \n', curr_vd);
        fprintf('Average Precision with inference on train data: %4.2f, %4.2f \n', ...
            perfclsf_train(1), perfsamp_train(2));   
        fprintf('Average Precision with inference on validation data: %4.2f, %4.2f \n', ...
            perfclsf_valid(1), perfsamp_valid(2));
    end
end

function updateParams() 
    if props.trainClassifier 
        H_classifier        = H_clampeds;
        [feats_, labels_]   = bag2dataset(train_bags, H_classifier);
        % Neural net training on bags and hiddenVars (H_classifer)
        model = trainClassifier(hyperparams.clsf_type, feats_, labels_, K, hyperparams.nnet_params);
    else        
        H_classifier = H_classifier_next;    
    end
    params = params_next;
end
    
function collectParams()
	params_all{curr_epoch}          = params;
    models_all{curr_epoch}          = model;
    H_classifiers_all{curr_epoch}   = H_classifier;
    H_clampeds_all{curr_epoch}      = H_clampeds;
    H_frees_all{curr_epoch}         = H_frees;
    y_frees_all{curr_epoch}         = y_frees;    
    infer_perf.train(curr_epoch, :) = [perfclsf_train, nan, perfsamp_train];
	infer_perf.valid(curr_epoch, :) = [perfclsf_valid, nan, perfsamp_valid];
    sampling_perf(curr_epoch, :)    = samp_perf;
%     y_targets_all{curr_epoch}       = y_target;
end

function writeEpoch2File()
    if exist(props.fullname, 'file')
        my_epoch = curr_epoch;
        load(props.fullname);
        curr_epoch = my_epoch;
    end
    collectParams()
    % save current epoch props to file...    
    save(props.fullname, 'curr_cv', 'curr_vd', 'curr_epoch', 'hyperparams', 'props', 'y_target', ...
                'params_all', 'models_all', 'sampling_perf', 'infer_perf',  ...
                'H_classifiers_all', 'H_clampeds_all', 'H_frees_all', 'y_frees_all');
end

function [perf_clsf, perf_samp] = validateModel(bags, labels, proxy)
% Performs validation of a model on a given data; returns Average Precision

    cd_step = hyperparams.CD_steps_inference;
    gibbs_step = hyperparams.gibbs_walk_inference;
    
    pred_clsf = zeros(size(labels));
    pred_samp = zeros(size(labels));
    for b = 1:length(bags)
        X = bags{b};
        U = proxy{b};
        [pred_clsf(:,b), ~, pred_samp(:,b), ~] = ...
            inference(X, U, params, model, K, cd_step, gibbs_step, hyperparams.clsf_type);
    end
    perf_clsf = [Average_precision(pred_clsf, labels), ...
                 Hamming_loss(pred_clsf,labels), ...
                 Ranking_loss(pred_clsf,labels), ...
                 One_error(pred_clsf,labels), ...
                 coverage(pred_clsf,labels)];
    perf_samp = [Average_precision(pred_samp, labels), ...
                 Hamming_loss(pred_samp,labels), ...
                 Ranking_loss(pred_samp,labels), ...
                 One_error(pred_samp,labels), ...
                 coverage(pred_samp,labels)];                 
end


end

    
end
        
