clear
load hcrfPIM_(aux_lclNOtrain_v0)_testcv(1)_validcv(2)_proxy(1).mat

%%

allparams = [params_all{1:curr_epoch}];
allparams = reshape(allparams, 3, length(allparams)/3);

figure, 
for i = 1:curr_epoch
    imagesc(params_all{i}{1});
    title(sprintf('epoch #%i', i));
    colorbar
    hold on
    pause(0.2)
end

%%
fig_ttl = {'-vs-','-vs+', '+vs-', '+vs+'};
figure,
for i=1:curr_epoch
    for j=1:4
        subplot(1,4,j);
        imagesc(params_all{i}{3}(:,:,j));
        title(fig_ttl{j});
    end
%     title(sprintf('epoch #%i', i));
    hold on;
    pause(0.2);
end


%%
if ~exist('curr_epoch', 'var')
    curr_epoch = find(cellfun(@isempty, params_all), 1) - 1;
end

figure,
plot(sampling_perf(1:curr_epoch,:))

for i = 1:curr_epoch
    t_c = tabulate(cell2mat(H_classifiers_all{i}'));
    TC(i,:) = t_c(:,2);
    
    t_s = tabulate(cell2mat(H_clampeds_all{i}'));
    TS(i,:) = t_s(:,2);
end

clr = ['b','r','g','m'];
figure, hold on
% yyaxis left
for j = 1:4
    plot(TC(:,j), sprintf('%s', clr(j)) ); 
    plot(TS(:,j), sprintf('%s.-.', clr(j)) );
end


%%

tabulate(cell2mat(H_classifiers_all{1}'))
tabulate(cell2mat(H_clampeds_all{1}'))
tabulate(cell2mat(H_clampeds_all{curr_epoch}'))