clc, clear

%% SVM/Kernel Parameters
degree = 2; % Degree of the TK kernel
C = 10; % Penalty Term
bound = .1; % Bounds of integration are [0-bound,1+bound]^n
params = paramsTKL(degree,bound);

%% Load the Data
load('CircleData.mat');
x = Data.x;
y = Data.y;

%% Train the Support Vector Machine
tic;
[SVM] = TKL(x,y,'Classification',C,params);
toc
%[SVM] = TKL(x,y); % TKL with minimal user input.


%% Plot Prediction boundary if data is 2-Dimensional
[X,Y] = meshgrid(linspace(min(x(:,1))-bound,max(x(:,1))+bound,200),linspace(min(x(:,2))-bound,max(x(:,2))+bound,200));
[Z] = evaluateTKL(SVM,[X(:),Y(:)]'); Z(Z<0) = NaN;
plot(x(find(y==1),1),x(find(y==1),2),'ob')
hold on
plot(x(find(y==-1),1),x(find(y==-1),2),'*r')
surf(X,Y,reshape(Z,size(X)),'EdgeColor','none')
shading interp
axis([min(x(:,1))-bound,max(x(:,1))+bound,min(x(:,2))-bound,max(x(:,2))+bound])
alpha(.4)
view(2)
hold off

legend({'y = 1','y=-1','TKL Classifier'},'Location','NorthWest','FontSize',15)
xlabel('First Input x_1','FontSize',20)
ylabel('Second Input x_2','FontSize',20)
title('Example Classification Problem with TKL','FontSize',23)