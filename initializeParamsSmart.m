function params = initializeParamsSmart(Y)
% K is number of class labels.
% alpha is K by K matrix; weighting params of spatial relation between regions.
% gamma is a scalar value; weighting param of coherence between region and
% image labels.
% mu is K by K by 4 matrix; weighting param of correlations of image
% labels.


K = size(Y,1);
% randomly set some region label weights
% W_alpha = [0.5 .25 0.2 .05;
%            .25 0.5 0.4 0.1
%            0.2 0.4 0.5 0.2
%            .05 0.1 0.2 0.6];
% W_alpha = W_alpha .* 0.05;
W_alpha = [0.0426    0.0160    0.0098   -0.0182;
           0.0160    0.0414   -0.0023   -0.0148;
           0.0098   -0.0023    0.0438    0.0044;
          -0.0182   -0.0148    0.0044    0.0422];
W_alpha = W_alpha .* 10; % when using MEAN(not SUM) in potential_hh_v2.
W_gamma = 0.5 * K;

W_mu = zeros(K,K,4);
for j = 1:size(Y,2)
    y = Y(:,j);
    [~, W_aux] = potential_yy(y, W_mu);
    W_mu = W_mu + W_aux;
end
W_mu = W_mu ./ sum(sum(W_mu(:)));
W_mu = W_mu .* (K^2);


params = zip_params(W_alpha, W_gamma, W_mu);    
