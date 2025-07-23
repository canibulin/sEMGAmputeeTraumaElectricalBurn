% stats_sEMG_sample_size_validation.m
% -------------------------------------------------------------
% Loads sEMG feature data from CSV, identifies the Co-Contraction Index (CCI),
% and evaluates whether group sample sizes are statistically adequate for
% detecting group differences using effect size and power analysis.
% -------------------------------------------------------------

%% 1. Load data
tbl = readtable('WristFlexoExtension6.csv');
tbl.groupStump = categorical(tbl.groupStump);

%% 2. Summarize group sample sizes
groupLabels = categories(tbl.groupStump);
groupSizes  = countcats(tbl.groupStump);
disp(table(groupLabels, groupSizes, ...
     'VariableNames', {'Group','SampleSize'}));

%% 3. Locate CCI feature
cciIdx = contains(tbl.Properties.VariableNames, {'CCI','indexCoactivations'});
if ~any(cciIdx)
    error('CCI-related feature not found in the dataset.');
end
cciVar = tbl.Properties.VariableNames{find(cciIdx,1)};
cciData = tbl.(cciVar);
groups  = tbl.groupStump;

%% 4. Kruskal–Wallis test
[p_kw,Hstat_raw,~] = kruskalwallis(cciData, groups, 'off');
Hstat = Hstat_raw{1};  % Extract numeric value from cell

k = numel(groupLabels);  % Number of groups
N = numel(cciData);      % Total sample size

% 5. Effect size calculation (eta² and Cohen's f)
eta2 = (Hstat - (k - 1)) / (N - k);             % Kruskal–Wallis effect size
cohen_f = sqrt(eta2 / (1 - eta2));              % Convert to Cohen's f

%% 6. Power analysis: required sample size per group
% Sample size formula for one-way ANOVA:
% n_per_group = [ (lambda / f²) + (k - 1) ] / k

% Parameters
alpha = 0.05;
power = 0.8;
k = numel(groupLabels);    % number of groups
f = cohen_f;               % from earlier
% Approximate required sample size per group
n_required = ceil(( (f^(-2)) * (finv(1 - alpha, k - 1, 1000) + finv(power, k - 1, 1000) ) ) );
lambda = fzero(@(l) 1 - ncfcdf(f^2*(k-1), k-1, k*(n_required - 1), l) - power, 10); % optional refinement



fprintf('\n📐 Estimated required sample size per group (ANOVA approximation): %.1f\n', n_required);