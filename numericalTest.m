clc, clear

dataSet = 1;
numTrials = 5; % Must be an integer between 1 and 12
% 1 = Abalone
% 2 = Airfoil
% 3 = CCPP
% 4 = Gas Turbine
% 5 = Hill Valley
% 6 = Shill Bid
% 7 = Transfusion
% 8 = FourClass
% 9 = German
% 10 = Space
% 11 = CA
% 12 = Boston Housing

%% Load Data and Parameters
switch dataSet
    case 1
        load('Data/abalone.mat'); p = load('Data/abalone_params.mat');
        Type = 'Classification'; C = p.D.C; % Calculated using 2-fold CV
    case 2
        load('Data/airfoil.mat'); p = load('Data/airfoil_params.mat');
        Type = 'Regression'; C = p.D.C; % Calculated using 2-fold CV
    case 3
        load('Data/CCPP.mat'); p = load('Data/CCPP_params.mat');
        Type = 'Classification'; C = p.D.C; % Calculated using 2-fold CV
    case 4
        load('Data/gasTurbine.mat'); p = load('Data/gasTurbine_params.mat');
        Type = 'Regression'; C = p.D.C; % Calculated using 2-fold CV
    case 5
        load('Data/hillValley.mat'); p = load('Data/hillValley_params.mat');
        Type = 'Classification'; C = p.D.C; % Calculated using 2-fold CV
    case 6
        load('Data/shillBid.mat'); p = load('Data/shillBid_params.mat');
        Type = 'Regression'; C = p.D.C; % Calculated using 2-fold CV
    case 7
        load('Data/transfusion.mat'); p = load('Data/transfusion_params.mat');
        Type = 'Regression'; C = p.D.C; % Calculated using 2-fold CV
    case 8
        load('Data/fourclass.mat'); p = load('Data/fourclass_params.mat');
        Type = 'Regression'; C = p.D.C; % Calculated using 2-fold CV
    case 9
        load('Data/german.mat'); p = load('Data/german_params.mat');
        Type = 'Regression'; C = p.D.C; % Calculated using 2-fold CV
    case 10
        load('Data/space.mat'); p = load('Data/space_params.mat');
        Type = 'Regression'; C = p.D.C; % Calculated using 2-fold CV
    case 11
        load('Data/ca.mat'); p = load('Data/ca_params.mat');
        Type = 'Regression'; C = p.D.C; % Calculated using 2-fold CV
    case 12
        load('Data/bostonHousing.mat'); p = load('Data/bostonHousing_params.mat');
        Type = 'Regression'; C = p.D.C; % Calculated using 2-fold CV
    otherwise
        error('That dataset does not exist.  Please enter an integer between 1 and 12.')
end

params.maxit = 100; params.epsilon = .1; params.tol = .01; params.degree = 1;
for n = 1:numTrials
    x = D{n}.x; y = D{n}.y; xTest = D{n}.xTest; yTest = D{n}.yTest; 
    params.bound = p.D.bound(n);
    
    %% Train the Support Vector Machine
    params = paramsTK(1, p.D.bound(n), .01, 100, p.D.tol);
    [SVM] = PMKL(x,y,Type,C(n),params);
    time(n) = sum(SVM.Opt.T);
    
    yPred = evaluatePMKL(SVM,xTest);
    if strcmp(Type,'Regression')
        mse(n) = sum((yTest-yPred').^2)./length(yTest);
    else
        tsa(n) = sum(yTest == yPred')./length(yTest);
    end
end
    
    