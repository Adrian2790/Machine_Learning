Problem 


MLP Regression Classic approach
close all;
clear all;
clc;
diary logFileName_P1_MLP_vCl.txt;
NumHiddenNeurons = 2; % 2, 4, 8, 32
N = 10; % 10, 50, 100, 1000
% prepare training & testing data
XTrain = 0:2*pi/N:2*pi;  % a sample on a column
YTrain = sin(XTrain).^2; % a sample on a column
XTest = 0:2*pi/(N*100):2*pi; % a sample on a column
YTest = sin(XTest).^2; % a sample on a column
%% build the architecture
net = newff(XTrain, YTrain, [NumHiddenNeurons, 1], {'tansig', 'purelin' },'trainlm');
%% config train params
net.trainParam.epochs = 100; % 100, 250, 500
net.trainParam.lr = 0.1; % 0.01, 0.1, 0.3
%net.trainParam.mc = ;
net.trainParam.min_grad = 1e-15;
net.trainParam.max_fail = 200;
%% train the model
net = train(net, XTrain, YTrain);
%% verify the results
	% compute the responses
YNetTrain = sim(net,XTrain);
YNetTest = sim(net,XTest);
% compute the accuracy
AccTrain = mean((YTrain-YNetTrain).^2);
AccTest = mean((YTest-YNetTest).^2);
fprintf('\n AccTrain: %g \n AccTest: %g', AccTest, AccTrain);
% plot the results
figure
subplot(1,2,1), plot(XTrain,YTrain,'or', XTrain,YNetTrain,'b');
title('Training');
subplot(1,2,2), plot(XTest,YTest,'or', XTest,YNetTest,'b');
title('testing');
labels = {'Original data', 'Noise alpha=0.05 percent=25%', 'Noise alpha=0.2 percent=25%', ...
   'Noise alpha=0.5 percent=25%', 'Redundant samples in [0; 0.2]'};
% original data
XTrainNew{1} = XTrain;
XTestNew{1} = XTest;
YTrainNew{1} = YTrain;
YTestNew{1} = YTest;
XTrainNew2 = XTrain;
XTestNew2 = XTest;
XTrainNew3 = XTrain;
XTestNew3 = XTest;
XTrainNew4 = XTrain;
XTestNew4 = XTest;
% noisy data
% XtrainNew=XTrain + alpha*randn(size(XTrain), with alpha = 0.05, 0.2, 0.5.
percent = 0.25;
j = sort(randperm(size(XTrain,2), floor(percent*size(XTrain,2))));
alfa = 0.05;
XTrainNew2(:,j)=XTrain(:,j) + alfa*randn(size(XTrain(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
XTestNew2(:,j)=XTest(:,j) + alfa*randn(size(XTest(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
YTrainNew2 = YTrain;
YTestNew2 = YTest;
alfa = 0.2;
XTrainNew3(:,j)=XTrain(:,j) + alfa*randn(size(XTrain(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
XTestNew3(:,j)=XTest(:,j) + alfa*randn(size(XTest(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
YTrainNew3 = YTrain;
YTestNew3 = YTest;
alfa = 0.5;
XTrainNew4(:,j)=XTrain(:,j) + alfa*randn(size(XTrain(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
XTestNew4(:,j)=XTest(:,j) + alfa*randn(size(XTest(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
YTrainNew4 = YTrain;
YTestNew4 = YTest;
XTrainNew{2} = XTrainNew2;
XTestNew{2} = XTestNew2;
YTrainNew{2} = YTrainNew2;
YTestNew{2} = YTestNew2;
XTrainNew{3} = XTrainNew3;
XTestNew{3} = XTestNew3;
YTrainNew{3} = YTrainNew3;
YTestNew{3} = YTestNew3;
XTrainNew{4} = XTrainNew4;
XTestNew{4} = XTestNew4;
YTrainNew{4} = YTrainNew4;
YTestNew{4} = YTestNew4;
% redundant samples
% an expanded training dataset, including XTrain and supplementary N redundant samples defined in [0, 0.2].
XTrainAdd = 0:2*pi/N:0.2;
XTrainNew{5} = [XTrain, XTrainAdd];
YTrainNew{5} = sin(XTrainNew{5}).^2;
XTestAdd = 0:2*pi/(N*100):0.2;
XTestNew{5} = [XTest, XTestAdd];
YTestNew{5} = sin(XTestNew{5}).^2;
net1 = cell(1, 5);
YNetTrain = cell(1,5);
YNetTest = cell(1,5);
AccTrain = zeros(1,5);
AccTest = zeros(1,5);
fprintf('\n');
for i = 1:1:5
   %% build the architecture
   net1{i} = newff(XTrainNew{i}, YTrainNew{i}, [NumHiddenNeurons, 1], {'tansig', 'purelin' },'trainlm');
   %% config train params
   net1{i}.trainParam.epochs = 100; % 100, 250, 500
   net1{i}.trainParam.lr = 0.1; % 0.01, 0.1, 0.3
   %net.trainParam.mc = ;
   net1{i}.trainParam.min_grad = 1e-15;
   net1{i}.trainParam.max_fail = 200;
  
   %% train the model
   net1{i} = train(net1{i}, XTrainNew{i}, YTrainNew{i});
   %% verify the results
	% compute the responses
   YNTrain = sim(net1{i}, XTrainNew{i});
   YNTest = sim(net1{i}, XTestNew{i});
   YNetTrain{i} = YNTrain;
   YNetTest{i} = YNTest;
  
   % compute the accuracy
   AccTrain(i) = mean((YTrainNew{i}-YNetTrain{i}).^2);
   AccTest(i) = mean((YTestNew{i}-YNetTest{i}).^2);
   fprintf('\n %s: AccTrain = %g AccTest = %g', labels{i}, AccTest(i), AccTrain(i));
end
save finalData_P1_MLP_Regr_classic.mat;
diary off;
 

 AccTrain: 0.125448 
 AccTest: 0.133035

 Original data: AccTrain = 0.152234 AccTest = 0.137412
 Noise alpha=0.05 percent=25%: AccTrain = 0.140605 AccTest = 0.167915
 Noise alpha=0.2 percent=25%: AccTrain = 0.0978391 AccTest = 0.0882073
 Noise alpha=0.5 percent=25%: AccTrain = 0.16729 AccTest = 0.160173
 redundant samples in [0; 0.2]: AccTrain = 0.0966167 AccTest = 0.0821246

/////////////////////////////////////////////////////////////////////////////////////////////

MLP Regression Deep Learning approach

close all;
clear all;
clc;
diary logFileName_P1_MLP_vDL.txt;
NumHiddenNeurons = 10;
maxEpochs = 5000;
N = 10; % 10, 50, 100, 1000
% prepare training & testing data
XTrain = (0:2*pi/N:2*pi)';  % a sample on a row
YTrain = sin(XTrain).^2; % a sample on a row
XTest = (0:2*pi/(N*100):2*pi)'; % a sample on a row
YTest = sin(XTest).^2; % a sample on a row
NumIn = size(XTrain, 2);%the number of inputs
NumSamples = size(XTrain, 1);% the number of samples
NumOut = size(XTrain, 2);%the number of outputs
%% define the architecture
layers = [
   featureInputLayer(NumIn)
   fullyConnectedLayer(NumHiddenNeurons)
   tanhLayer
   fullyConnectedLayer(NumOut)
   regressionLayer];
%% train the model
options = trainingOptions('sgdm', ...
   'MiniBatchSize', NumSamples,...           
   'MaxEpochs', maxEpochs, ...     
   'InitialLearnRate', 1e-1, ... 
   'Verbose',false, ...
   'Plots','training-progress');
net = trainNetwork(XTrain, YTrain, layers, options);
%% verify the model
YNetTrain = predict(net, XTrain);
YNetTest = predict(net, XTest);
% compute the accuracy
AccTrain = mean((YTrain-YNetTrain).^2);
AccTest = mean((YTest-YNetTest).^2);
fprintf('\n AccTrain: %g \n AccTest: %g', AccTest, AccTrain);
% plot the results
figure
subplot(1,2,1), plot(XTrain,YTrain,'or', XTrain,YNetTrain,'b');
title('Training');
subplot(1,2,2), plot(XTest,YTest,'or', XTest,YNetTest,'b');
title('testing');
labels = {'Original data', 'Noise alpha=0.05 percent=25%', 'Noise alpha=0.2 percent=25%', ...
   'Noise alpha=0.5 percent=25%', 'Redundant samples in [0; 0.2]'};
% original data
XTrainNew{1} = XTrain;
XTestNew{1} = XTest;
YTrainNew{1} = YTrain;
YTestNew{1} = YTest;
XTrainNew2 = XTrain;
XTestNew2 = XTest;
XTrainNew3 = XTrain;
XTestNew3 = XTest;
XTrainNew4 = XTrain;
XTestNew4 = XTest;
% noisy data
% XtrainNew=XTrain + alpha*randn(size(XTrain), with alpha = 0.05, 0.2, 0.5.
percent = 0.25;
j = sort(randperm(size(XTrain,2), floor(percent*size(XTrain,2))));
alfa = 0.05;
XTrainNew2(:,j)=XTrain(:,j) + alfa*randn(size(XTrain(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
XTestNew2(:,j)=XTest(:,j) + alfa*randn(size(XTest(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
YTrainNew2 = YTrain;
YTestNew2 = YTest;
alfa = 0.2;
XTrainNew3(:,j)=XTrain(:,j) + alfa*randn(size(XTrain(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
XTestNew3(:,j)=XTest(:,j) + alfa*randn(size(XTest(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
YTrainNew3 = YTrain;
YTestNew3 = YTest;
alfa = 0.5;
XTrainNew4(:,j)=XTrain(:,j) + alfa*randn(size(XTrain(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
XTestNew4(:,j)=XTest(:,j) + alfa*randn(size(XTest(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
YTrainNew4 = YTrain;
YTestNew4 = YTest;
XTrainNew{2} = XTrainNew2;
XTestNew{2} = XTestNew2;
YTrainNew{2} = YTrainNew2;
YTestNew{2} = YTestNew2;
XTrainNew{3} = XTrainNew3;
XTestNew{3} = XTestNew3;
YTrainNew{3} = YTrainNew3;
YTestNew{3} = YTestNew3;
XTrainNew{4} = XTrainNew4;
XTestNew{4} = XTestNew4;
YTrainNew{4} = YTrainNew4;
YTestNew{4} = YTestNew4;
% redundant samples
% an expanded training dataset, including XTrain and supplementary N redundant samples defined in [0, 0.2].
XTrainAdd = (0:2*pi/N:0.2)';
XTrainNew{5} = [XTrain; XTrainAdd];
YTrainNew{5} = sin(XTrainNew{5}).^2;
XTestAdd = (0:2*pi/(N*100):0.2)';
XTestNew{5} = [XTest; XTestAdd];
YTestNew{5} = sin(XTestNew{5}).^2;
net1 = cell(1, 5);
layer1 = cell(1, 5);
options1 = cell(1, 5);
YNetTrain = cell(1,5);
YNetTest = cell(1,5);
AccTrain = zeros(1,5);
AccTest = zeros(1,5);
fprintf('\n');
for i = 1:1:5
   %% define the architecture
   layers1{i} = [
       featureInputLayer(NumIn)
       fullyConnectedLayer(NumHiddenNeurons)
       tanhLayer
       fullyConnectedLayer(NumOut)
       regressionLayer];
  
   %% train the model
   options1{i} = trainingOptions('sgdm', ...
       'MiniBatchSize', NumSamples,...           
       'MaxEpochs', maxEpochs, ...     
       'InitialLearnRate', 1e-1, ... 
       'Verbose',false, ...
       'Plots','training-progress');
  
   net1{i} = trainNetwork(XTrainNew{i}, YTrainNew{i}, layers1{i}, options1{i});
   %% verify the results
	% compute the responses
   YNTrain = predict(net1{i}, XTrainNew{i});
   YNTest = predict(net1{i}, XTestNew{i});
   YNetTrain{i} = YNTrain;
   YNetTest{i} = YNTest;
  
   % compute the accuracy
   AccTrain(i) = mean((YTrainNew{i}-YNetTrain{i}).^2);
   AccTest(i) = mean((YTestNew{i}-YNetTest{i}).^2);
   fprintf('\n %s: AccTrain = %g AccTest = %g', labels{i}, AccTest(i), AccTrain(i));
end
save finalData_P1_MLP_Regr_deeplearning.mat;
diary off;

 
