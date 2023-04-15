clc, clear

%% Create Data
rng(17); % For Reproducibility
num = 998;
x = linspace(0, 10, num)';
% x1 = 0:0.1:1;
% x2 = 0:0.1:1;
% [X,Y] = meshgrid(x1,x2);
% xx1 = reshape(X, [121, 1]);
% xx2 = reshape(Y, [121, 1]);
% x = [xx1, xx2]';
% y = 2*xx1(1:end, 1) ;
y = 5*sign(x-5); %(x)+sin(x)+5*sign(x-5)+(rand(num, 1)-.5);
figure(1)
plot(x,y,'.k')
hold on
for C = [1, 5, 10, 100, 1000, 10000]
    %% Required Parameters
%     C = 10; % Penalty term

    %% Optional Parameters
    degree = 1; % degree of TK kernel function
    bound = .1; % Bounds of integration are [0-bound,1+bound]^n
    Eps = 1; % Epsilon-loss term
    params = paramsTK(degree,bound,Eps);

    
    %% Training
    model = svmtrain(y, x, ['-t 2 -s 0 -c ', num2str(C),' -q 1']);
%     tic;
%     SVM = PMKL(x,y,'Regression',C, params); % TKL finds an optimal TK kernel and trains a Support Vector Machine
%     toc

    %% Prediction
    xx =  linspace(0,10, 1000);
    yPred = svmpredict( xx,xx, model);
%     [yPred] = evaluatePMKL(SVM,xx'); % Uses the trained Support Vector Machine to predict the output.

    % Plot prediction
%     figure(1)
%     hold on
    plot(xx,yPred,'.-')
%     hold off
    fprintf('%.2f\n', max(abs(y - yPred)))
end

hold off
legend({'Data','TKL d = 1','TKL C = 2','TKL C = 3','TKL C = 4','TKL C = 6','TKL C = 8','TKL C = 10'},'Location','NorthWest','FontSize',15)
xlabel('Input (x)','FontSize',20)
ylabel('Output (y)','FontSize',20)
title('Example Regression Problem with TKL','FontSize',23)

