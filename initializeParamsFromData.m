function params = initializeParamsFromData(H, Y, proxy)

% K is number of class labels.
% alpha is K by K matrix; weighting params of spatial relation between regions.
% gamma is a scalar value; weighting param of coherence between region and
% image labels.
% mu is K by K by 4 matrix; weighting param of correlations of image
% labels.

K = size(Y,1);

% region label-region label weights.
W_alpha = zeros(K,K);
for h = 1:length(H) 
    if sum(Y(:,h)>0) > 1 
    %     R = length(H{h});
    %     U = getNeighborhood([],R);
        U = proxy{h};
        W = ones(K,K);
        [~, W_aux] = potential_hh_v2(H{h}, U, W);  
        
%         W_aux = W_aux ./ size(U,1);
%         W_aux = (W_aux - min(min(W_aux))) ./ (max(max(W_aux))- min(min(W_aux)));
        W_alpha = W_alpha + W_aux;
    end
end
counts = zeros(K,K);
for i=1:K
    for j=i:K
        counts(i,j) = sum(all(Y([i j],:)>0));
    end
end
W_alpha = W_alpha ./ (counts + triu(counts,1)');
if any(W_alpha(:) ~= 0)
    W_alpha = W_alpha ./ sum(sum(W_alpha));
end

% region label-image label weight. 
yfromH  = getYfromH(H, K);
aux     = diag(yfromH' * Y);
W_gamma = (mean(aux) - (-K)) / (2*K); % normalizing; (val - min) / (max - min)
% % (random?)
% W_gamma = K*rand; % multiplying by K to have it more weight. 
% image label-image label weights.
W_mu = zeros(K,K,4);
for j = 1:size(Y,2)
    y = Y(:,j);
    [~, W_aux] = potential_yy(y, W_mu);
    W_mu = W_mu + W_aux;
end
W_mu = W_mu ./ sum(sum(W_mu(:)));
W_mu = W_mu .* 4; % multiplying by 4 because there are 4 correlations, need to normalize w.r.t other params.
params = zip_params(W_alpha, W_gamma, W_mu);    

end
