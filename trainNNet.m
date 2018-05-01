% function [nnet_model, preds, conf_mat] = trainNNet(X, Y, params)
function nnet_model = trainNNet(X, Y, params)
% % % % % % % % % % % % % % % % % % % % % % % % % % % 
% \Phi_{yy}(y)
% N is the , D is #of features, K is #of class labels.
% inputs: 
%   params contains;
%       hidden: the sizes of the hidden layers
%       useGPU: the switch to use GPU or not.
%   X is the dataset; D by N
%   Y is the labels;  K by N
% outputs:
%   nnet_model; neural net model
%   preds is the predictions of the model
%   conf_mat 
% % % % % % % % % % % % % % % % % % % % % % % % % % % 
% 
    H = params.hidden;
    useGPU = 'no';
    if params.useGPU
        useGPU = 'yes'; 
    end
 
%     if isvector(Y)       
%         Y = labelVec2onehotMat(Y, length(unique(Y)));
%     end    
%     if size(X,1) ~= D, X = X'; end
%     if size(Y,1) ~= K, Y = Y'; end
            
    nnet = [];
    nnet = feedforwardnet(H);
    nnet = configure(nnet, X, Y);
%     nnet.trainFcn = 'trainscg'; % to supress warning.
    nnet.trainParam.showWindow = 0; % supress the neural net window.

    nnet_model = [];
    nnet_model = train(nnet, X, Y, 'UseGPU', useGPU);
    %%# to get the performance of the network;
%     predict = nnet_model(X);
%     perform(nnet_model, Y, predict)
%     %%# another way for performance evaluation.
%     [~, preds]  = max(nnet_model(X, 'UseGPU', useGPU));
%     conf_mat    = confusionmat(onehotMat2labelVec(Y), preds); 
    
    
end