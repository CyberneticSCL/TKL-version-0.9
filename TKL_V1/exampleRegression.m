clc, clear

%% Create Data
rng(17); % For Reproducibility
num = 1000;
x = 10.*rand(num,1);
y = (x)+sin(x)+5.*sign(x-5)+(rand(num,1)-.5);

%% Required Parameters
C = 1; % Penalty term

%% Optional Parameters
degree = 1; % degree of TK kernel function
bound = .1; % Bounds of integration are [0-bound,1+bound]^n
Eps = 1; % Epsilon-loss term
params = paramsTKL(degree,bound,Eps);


%% Training
tic;
SVM = TKL(x,y,'Regression',C); % TKL finds an optimal TK kernel and trains a Support Vector Machine
%SVM = TKL(x,y,'Regression',C,params); % TKL with additional optional parameters.
toc

%% Prediction
xx = linspace(0,10);
[yPred] = evaluateTKL(SVM,xx); % Uses the trained Support Vector Machine to predict the output.

%% Plot prediction
figure(1)
plot(x,y,'ok')
hold on
plot(xx,yPred,':g','LineWidth',3)
plot(xx,(xx)+sin(xx)+5.*sign(xx-5),'r')
hold off
legend({'Data','TKL','Actual'},'Location','NorthWest','FontSize',15)
xlabel('Input (x)','FontSize',20)
ylabel('Output (y)','FontSize',20)
title('Example Regression Problem with TKL','FontSize',23)
