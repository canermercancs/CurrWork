function model = trainClassifier(region_classifier, feats, labels, K, model_params)

    if strcmp(region_classifier, 'nnet')
        model = trainNNet(feats', labelVec2onehotMat(labels,K), model_params);
    else    
    % SVM params
        model = trainSVM(feats, labels, model_params);
    end
    
%     


