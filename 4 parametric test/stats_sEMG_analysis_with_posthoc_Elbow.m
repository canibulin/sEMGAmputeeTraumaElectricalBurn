% repeated_cv_balancedAccuracy.m
% Repeated stratified k-fold evaluation using balanced accuracy for multiclass data.
% Recommend starting a parallel pool before running: parpool('local');
%
% Usage:
% 1) Place this file in the same folder as 'WristFlexoExtension6CrossValidation3.csv'
% 2) Edit testOption as desired (1,2,3)
% 3) Run in MATLAB. Uncomment parpool lines to use parallel workers.

clearvars; close all; clc;

%% --------------------- User settings ---------------------
csvFile = 'ElbowBicepsTricepsProcessed6.csv';  % data file
labelColName = 'Label';                                % name of label column in CSV

testOption = 1;    % 1 = all features except CCI; 2 = all features including CCI; 3 = select specific 11 features
k = 5;             % folds
repeats = 10;      % repeated CV
rngSeed = 1;       % RNG for reproducibility

useParallel = true;   % set true to run repeats in parallel (requires Parallel Toolbox)
% If useParallel is true and no pool exists, the script will start a local pool automatically.

%% --------------------- Load data ---------------------
if ~isfile(csvFile)
    error('CSV file not found: %s', csvFile);
end
data = readtable(csvFile, 'ReadVariableNames', true);

% Extract label column
if ismember(labelColName, data.Properties.VariableNames)
    Y = data.(labelColName);
    data.(labelColName) = [];  % remove label column
else
    error('Label column "%s" not found in the data', labelColName);
end

% Ensure categorical labels
if ~iscategorical(Y)
    Y = categorical(Y);
end

%% --------------------- Feature selection per testOption ---------------------
switch testOption
    case 1
        % Remove CCI if present
        if ismember('CCI', data.Properties.VariableNames)
            data.CCI = [];
        else
            warning('CCI column not found; proceeding with all features.');
        end
        X = table2array(data);
    case 2
        % Use all features including CCI
        X = table2array(data);
    case 3
        % Select specific reduced feature set
        selectedFeatures = {'LOGCH1','ACCCH1','FRCH1','PKFCH1', ...
                            'LOGCH2','FRCH2','MNFCH2','MDFCH2', ...
                            'PKFCH2','CCI','ACCCH2'};
        missingFeatures = setdiff(selectedFeatures, data.Properties.VariableNames);
        if ~isempty(missingFeatures)
            error('Missing selected feature columns: %s', strjoin(missingFeatures, ', '));
        end
        X = table2array(data(:, selectedFeatures));
    otherwise
        error('Invalid testOption. Use 1, 2, or 3.');
end

%% --------------------- Prepare for repeated stratified k-fold ---------------------
rng(rngSeed);
numClasses = numel(categories(Y));

% Preallocate results
balAccSVM = zeros(repeats, k);
balAccKNN = zeros(repeats, k);
balAccEns = zeros(repeats, k);

% Manage parallel pool if requested
poolObj = [];
if useParallel
    poolObj = gcp('nocreate');
    if isempty(poolObj)
        % Start a local pool with default settings
        try
            poolObj = parpool('local'); %#ok<UNRCH> % This may start workers
        catch ME
            %warning('Could not start parallel pool automatically: %s\nProceeding without parallel execution.', ME.message);
            useParallel = false;
        end
    end
end

%% --------------------- Repeated CV loop ---------------------
% We parallelize over repeats (outer loop) using parfor when possible
if useParallel
    parfor r = 1:repeats
        % Create a stratified partition for this repeat (cvpartition uses rng state)
        cv = cvpartition(Y, 'KFold', k);
        balSVM_rep = zeros(1, cv.NumTestSets);
        balKNN_rep = zeros(1, cv.NumTestSets);
        balEns_rep = zeros(1, cv.NumTestSets);
        for fold = 1:cv.NumTestSets
            trainIdx = training(cv, fold);
            testIdx  = test(cv, fold);

            X_train = X(trainIdx, :);
            Y_train = Y(trainIdx);
            X_test  = X(testIdx, :);
            Y_test  = Y(testIdx);

            % Train models
            tSVM = templateSVM('KernelFunction','gaussian','KernelScale','auto','Standardize',true);
            svmMdl = fitcecoc(X_train, Y_train, 'Learners', tSVM, 'Coding', 'onevsall');

            knnMdl = fitcknn(X_train, Y_train, 'Distance', 'cosine', 'NumNeighbors', 5);

            ensMdl = fitcensemble(X_train, Y_train, 'Method', 'Bag', 'NumLearningCycles', 100);

            % Predict
            predSVM = predict(svmMdl, X_test);
            predKNN = predict(knnMdl, X_test);
            predEns = predict(ensMdl, X_test);

            % Compute balanced accuracy for this fold
            balSVM_rep(fold) = balancedAccuracy(Y_test, predSVM);
            balKNN_rep(fold) = balancedAccuracy(Y_test, predKNN);
            balEns_rep(fold) = balancedAccuracy(Y_test, predEns);
        end
        balAccSVM(r, :) = balSVM_rep;
        balAccKNN(r, :) = balKNN_rep;
        balAccEns(r, :) = balEns_rep;
    end
else
    for r = 1:repeats
        cv = cvpartition(Y, 'KFold', k);
        for fold = 1:cv.NumTestSets
            trainIdx = training(cv, fold);
            testIdx  = test(cv, fold);

            X_train = X(trainIdx, :);
            Y_train = Y(trainIdx);
            X_test  = X(testIdx, :);
            Y_test  = Y(testIdx);

            % Train models
            tSVM = templateSVM('KernelFunction','gaussian','KernelScale','auto','Standardize',true);
            svmMdl = fitcecoc(X_train, Y_train, 'Learners', tSVM, 'Coding', 'onevsall');

            knnMdl = fitcknn(X_train, Y_train, 'Distance', 'cosine', 'NumNeighbors', 5);

            ensMdl = fitcensemble(X_train, Y_train, 'Method', 'Bag', 'NumLearningCycles', 100);

            % Predict
            predSVM = predict(svmMdl, X_test);
            predKNN = predict(knnMdl, X_test);
            predEns = predict(ensMdl, X_test);

            % Compute balanced accuracy for this fold
            foldIdx = fold;
            balAccSVM(r, foldIdx) = balancedAccuracy(Y_test, predSVM);
            balAccKNN(r, foldIdx) = balancedAccuracy(Y_test, predKNN);
            balAccEns(r, foldIdx) = balancedAccuracy(Y_test, predEns);
        end
    end
end

%% --------------------- Aggregate results ---------------------
vecSVM = balAccSVM(:);
vecKNN = balAccKNN(:);
vecEns = balAccEns(:);

meanSVM = mean(vecSVM);
meanKNN = mean(vecKNN);
meanEns = mean(vecEns);

seSVM = std(vecSVM) / sqrt(numel(vecSVM));
seKNN = std(vecKNN) / sqrt(numel(vecKNN));
seEns = std(vecEns) / sqrt(numel(vecEns));

CI_SVM = [meanSVM - 1.96*seSVM, meanSVM + 1.96*seSVM];
CI_KNN = [meanKNN - 1.96*seKNN, meanKNN + 1.96*seKNN];
CI_Ens = [meanEns - 1.96*seEns, meanEns + 1.96*seEns];

%% --------------------- Display results ---------------------
fprintf('\nBalanced accuracy results (mean ± 95%% CI)\n');
fprintf('SVM (ECOC Gaussian)     : %.2f%% [%.2f%%, %.2f%%]\n', meanSVM*100, CI_SVM(1)*100, CI_SVM(2)*100);
fprintf('KNN (Cosine, k=5)       : %.2f%% [%.2f%%, %.2f%%]\n', meanKNN*100, CI_KNN(1)*100, CI_KNN(2)*100);
fprintf('Ensemble (Bagged Trees) : %.2f%% [%.2f%%, %.2f%%]\n', meanEns*100, CI_Ens(1)*100, CI_Ens(2)*100);

%% --------------------- Save results ---------------------
results.mean = struct('SVM', meanSVM, 'KNN', meanKNN, 'Ensemble', meanEns);
results.CI = struct('SVM', CI_SVM, 'KNN', CI_KNN, 'Ensemble', CI_Ens);
results.perFold = struct('SVM', balAccSVM, 'KNN', balAccKNN, 'Ensemble', balAccEns);
save('balanced_acc_results.mat', 'results');

fprintf('Results saved to balanced_acc_results.mat\n');

%% --------------------- Local helper function ---------------------
function ba = balancedAccuracy(yTrue, yPred)
% balancedAccuracy  Compute balanced accuracy for multiclass predictions.
%   ba = balancedAccuracy(yTrue, yPred) returns the mean recall (sensitivity)
%   across classes present in yTrue for the given yPred.
%
%   yTrue and yPred can be categorical or convertible to categorical.

    if ~iscategorical(yTrue), yTrue = categorical(yTrue); end
    if ~iscategorical(yPred), yPred = categorical(yPred); end

    classes = categories(yTrue);
    nClasses = numel(classes);
    recalls = nan(nClasses, 1);

    for i = 1:nClasses
        cls = classes{i};
        idxTrue = (yTrue == cls);
        if ~any(idxTrue)
            % Class not present in this fold's true labels: leave NaN
            recalls(i) = NaN;
            continue;
        end
        tp = sum(yPred == cls & idxTrue);
        fn = sum(yPred ~= cls & idxTrue);
        recalls(i) = tp / (tp + fn);  % sensitivity (recall) for this class
    end

    % Average recall across classes present in yTrue (ignore NaNs)
    ba = mean(recalls(~isnan(recalls)));
end