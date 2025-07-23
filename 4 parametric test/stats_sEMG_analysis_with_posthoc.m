% stats_sEMG_analysis_with_posthoc.m
%-------------------------------------------------------------
% Loads sEMG feature data with a grouping variable 'groupStump',
% runs ANOVA or Kruskal–Wallis per feature, interprets p-values,
% and performs post-hoc pairwise comparisons for significant results.
%-------------------------------------------------------------

%% 1. Load data
tbl = readtable('WristFlexoExtension6.csv');
tbl.groupStump = categorical(tbl.groupStump);  

%% 2. Identify feature columns
allVars  = tbl.Properties.VariableNames;
featVars = setdiff(allVars, {'groupStump'});

% Prioritize CCI-like columns
cciIdx     = contains(featVars, {'CCI','CoCon','index coactivations'}); 
featVars = [featVars(cciIdx), featVars(~cciIdx)];

%% 3. Setup results container (fixed pairing)
alpha   = 0.05;
results = struct( ...
    'Feature',       {}, ...
    'TestUsed',      {}, ...
    'pValue',        {}, ...
    'Interpretation',{}, ...   
    'PostHoc',       {} ...
);

grpLevels = categories(tbl.groupStump);

%% 4. Loop over each feature
for i = 1:numel(featVars)
    varName = featVars{i};
    dataVec = tbl.(varName);
    groups  = tbl.groupStump;
    
    % 4.1 Normality per group (Lilliefors)
    isNormal = true;
    for g = 1:numel(grpLevels)
        vals = dataVec(groups==grpLevels{g});
        if numel(vals) >= 5 && lillietest(vals) == 1
            isNormal = false;
            break;
        end
    end
    
    % 4.2 Variance homogeneity (Levene’s)
    pVar = vartestn(dataVec, groups, ...
                   'TestType','LeveneAbsolute', ...
                   'Display','off');
    
    % 4.3 Global test
    if isNormal && pVar > alpha
        [p,~,stats] = anova1(dataVec, groups, 'off');
        testName    = 'ANOVA';
    else
        [p,~,stats] = kruskalwallis(dataVec, groups, 'off');
        testName    = 'Kruskal–Wallis';
    end
    
    % 4.4 Interpret omnibus p-value
    if p < alpha
        interp = sprintf('p=%.3f < %.2f → reject H₀: groups differ', p, alpha);
    else
        interp = sprintf('p=%.3f ≥ %.2f → fail to reject H₀', p, alpha);
    end
    
    % 4.5 Post-hoc pairwise comparisons
    posthocList = {};
    if p < alpha
        switch testName
            case 'ANOVA'
                comp = multcompare(stats, ...
                                   'CType','tukey-kramer', ...
                                   'Display','off');
            case 'Kruskal–Wallis'
                comp = multcompare(stats, ...
                                   'CType','dunn-sidak', ...
                                   'Display','off');
        end
        % comp cols: [grp1, grp2, lowerCI, meanDiff, upperCI, adjP]
        sigIdx = comp(:,6) < alpha;
        for k = find(sigIdx)'
            g1 = grpLevels{ comp(k,1) };
            g2 = grpLevels{ comp(k,2) };
            adjP = comp(k,6);
            posthocList{end+1} = ...
                sprintf('%s vs %s (p=%.3f)', g1, g2, adjP); %#ok<SAGROW>
        end
    end
    
    % 4.6 Store results
    results(end+1).Feature        = varName;       %#ok<SAGROW>
    results(end  ).TestUsed       = testName;
    results(end  ).pValue         = p;
    results(end  ).Interpretation = interp;
    results(end  ).PostHoc        = posthocList;
end

%% 5. Display and save
resultsTable = struct2table(results);
disp(resultsTable);

writetable(resultsTable, 'sEMG_stat_results_with_posthoc.csv');