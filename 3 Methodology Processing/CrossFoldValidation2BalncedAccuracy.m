%parallel pool is recommended 
%% Load Data
% Read the CSV file with the first row treated as variable names.
data = readtable('WristFlexoExtension6CrossValidation3.csv', 'ReadVariableNames', true);  

% Extract the label column. Make sure your CSV file has a column named 'Label'.
if ismember('Label', data.Properties.VariableNames)
    Y = data.Label;
    data.Label = [];  % Remove the label column from predictors.
else
    error('Label column not found in the data');
end

%% Select Test Option
% testOption = 1 --> Use all features except CCI.
% testOption = 2 --> Use all features (including CCI).
% testOption = 3 --> Use feature reduction (e.g., PCA) on all features.
testOption = 3;  

switch testOption
   case 1
        % Remove CCI if it exists.
        if ismember('CCI', data.Properties.VariableNames)
            data.CCI = [];
        else
            warning('CCI column not found; proceeding with all features.');
        end
   case 2
        % Use all features (including CCI): do nothing.
   case 3
    % Seleccionar únicamente las características especificadas:
    % LOGCH1, ACCCH1, FRCH1, PKFCH1, LOGCH2, FRCH2, MNFCH2, MDFCH2, PKFCH2, CCI, ACCCH2
    selectedFeatures = {'LOGCH1', 'ACCCH1', 'FRCH1', 'PKFCH1', ...
                        'LOGCH2', 'FRCH2', 'MNFCH2', 'MDFCH2', ...
                        'PKFCH2', 'CCI', 'ACCCH2'};
    
    % Verificar que todas las características seleccionadas existan en la tabla de datos.
    missingFeatures = setdiff(selectedFeatures, data.Properties.VariableNames);
    if ~isempty(missingFeatures)
        error('Faltan las siguientes columnas de características: %s', strjoin(missingFeatures, ', '));
    end
    
    % Extraer solo las características seleccionadas del conjunto de datos.
    X = table2array(data(:, selectedFeatures));


    otherwise
        error('Invalid testOption value. Set it to 1, 2, or 3.');
end

if testOption ~= 3
    X = table2array(data);  % Convert predictors table to a numeric array.
end

% Ensure that labels are categorical.
if ~iscategorical(Y)
    Y = categorical(Y);
end

%% Cross-Validation Parameters
k = 5;          % Number of folds for cross-validation.
repeats = 10;   % Number of repetitions.
accuracySVM = zeros(repeats, k);
accuracyKNN = zeros(repeats, k);
accuracyEns = zeros(repeats, k);

rng(1);  % For reproducibility.

%% Repeated k-Fold Cross-Validation
for r = 1:repeats
    cv = cvpartition(Y, 'KFold', k);  % Create a stratified k-fold partition.
    for fold = 1:cv.NumTestSets
        % Get training and test indices.
        trainIdx = training(cv, fold);
        testIdx  = test(cv, fold);
        
        % Split data into training and test sets.
        X_train = X(trainIdx, :);
        Y_train = Y(trainIdx);
        X_test  = X(testIdx, :);
        Y_test  = Y(testIdx);
        
        %% Train Models
        
        % For multi-class SVM, use ECOC with SVM learners.
        template = templateSVM('KernelFunction', 'gaussian', 'KernelScale', 'auto');
        svmMdl = fitcecoc(X_train, Y_train, 'Learners', template);
        
        % KNN with cosine distance.
        knnMdl = fitcknn(X_train, Y_train, 'Distance', 'cosine');
        
        % Ensemble using Bagged Trees.
        ensMdl = fitcensemble(X_train, Y_train, 'Method', 'Bag');
        
        %% Predictions and Accuracy Calculation
        predSVM = predict(svmMdl, X_test);
        predKNN = predict(knnMdl, X_test);
        predEns = predict(ensMdl, X_test);
        
        accuracySVM(r, fold) = mean(predSVM == Y_test);
        accuracyKNN(r, fold) = mean(predKNN == Y_test);
        accuracyEns(r, fold) = mean(predEns == Y_test);
    end
end

%% Calculate 95% Confidence Intervals
% Flatten the matrices of accuracies into vectors.
accVectorSVM = accuracySVM(:);
accVectorKNN = accuracyKNN(:);
accVectorEns = accuracyEns(:);

% Compute mean accuracies.
meanSVM = mean(accVectorSVM);
meanKNN = mean(accVectorKNN);
meanEns = mean(accVectorEns);

% Compute standard errors.
stdErrSVM = std(accVectorSVM) / sqrt(length(accVectorSVM));
stdErrKNN = std(accVectorKNN) / sqrt(length(accVectorKNN));
stdErrEns = std(accVectorEns) / sqrt(length(accVectorEns));

% Compute 95% confidence intervals (assuming normality of accuracy distributions).
CI_SVM = [meanSVM - 1.96 * stdErrSVM, meanSVM + 1.96 * stdErrSVM];
CI_KNN = [meanKNN - 1.96 * stdErrKNN, meanKNN + 1.96 * stdErrKNN];
CI_Ens = [meanEns - 1.96 * stdErrEns, meanEns + 1.96 * stdErrEns];

%% Display the Results
fprintf('SVM (Fine Gaussian Kernel via ECOC): Mean Accuracy = %.2f%%, 95%% CI = [%.2f%%, %.2f%%]\n', ...
    meanSVM*100, CI_SVM(1)*100, CI_SVM(2)*100);
fprintf('KNN (Cosine): Mean Accuracy = %.2f%%, 95%% CI = [%.2f%%, %.2f%%]\n', ...
    meanKNN*100, CI_KNN(1)*100, CI_KNN(2)*100);
fprintf('Ensemble (Bagged Trees): Mean Accuracy = %.2f%%, 95%% CI = [%.2f%%, %.2f%%]\n', ...
    meanEns*100, CI_Ens(1)*100, CI_Ens(2)*100);

