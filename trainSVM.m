function [myModel, mySVM] = trainSVM(X,Y, params)
% C is the cost parameter of the SVM
% lambda is the gamma parameter of rbf kernel SVM.

    rng(1234); 
    kernel = params.kernel;
    mySVM = templateSVM('KernelFunction', kernel );
    myModel = fitcecoc(X, Y, 'Learners', mySVM, 'FitPosterior', 1, ...
                        'ClassNames', unique(Y));
             
end