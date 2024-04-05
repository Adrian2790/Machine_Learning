//Solve the classification problem with RF.

close all;
clear;
clc;
diary logFileName_P2_RF.txt;
% rez = createMyDTRFData('HepatitisC.csv', 0.2, 0, 'myDTRFData.mat'); % se decomenteaza la prima rulare
load 'myDTRFData.mat';
% original data
XTrainNew1 = XTrain;
XTestNew1 = XTest;
YTrainNew1 = YTrain;
YTestNew1 = YTest;
% redundant data
XTrainNew2 = [XTrain, XTrain(:,end) XTrain(:,end)];
XTestNew2 = [XTest, XTest(:,end) XTest(:,end)];
YTrainNew2 = YTrain;
YTestNew2 = YTest;
% redundant samples
[nl1, nc1] = size(XTrain);
[nl2, nc2] = size(XTest);
proc = 0.2; % 20%
i = sort(randperm(min(nl1, nl2), floor(proc*min(nl1,nl2))));
XTrainNew3 = XTrain(i,:);
YTrainNew3 = YTrain(i,:);
XTestNew3 = XTest;
YTestNew3 = YTest;
% normalized data
% normalizare la 0
[XTrainNew4, YTrainNew4] = normalizeData(XTrain, YTrain);
[XTestNew4, YTestNew4] = normalizeData(XTest, YTest);
% normalizare la [2, 10]
a = 2;
b = 10;
[XTrainNew5, YTrainNew5] = normalizeData(XTrain, YTrain, a, b);
[XTestNew5, YTestNew5] = normalizeData(XTest, YTest, a, b);
% outliers
m1=min(XTrainNew5); M1=max(XTrain);
m2=min(XTest); M2=max(XTest);
% % exemple îndepărtate: 2*M, m-2*M, etc.
XTrainNew6 = [XTrain; m1-2*M1; 2*M1];
YTrainNew6 = [YTrain; 2; 2];
XTestNew6 = [XTest; m2-2*M2; 2*M2];
YTestNew6 = [YTest; 2; 2];
% noisy data
percent = 0.25;
j = sort(randperm(size(XTrain,2), floor(percent*size(XTrain,2))));
alfa = 0.05;
XTrainNew7(:,j)=XTrain(:,j) + alfa*randn(size(XTrain(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
XTestNew7(:,j)=XTest(:,j) + alfa*randn(size(XTest(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
YTrainNew7 = YTrain;
YTestNew7 = YTest;
alfa = 0.2;
XTrainNew8(:,j)=XTrain(:,j) + alfa*randn(size(XTrain(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
XTestNew8(:,j)=XTest(:,j) + alfa*randn(size(XTest(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
YTrainNew8 = YTrain;
YTestNew8 = YTest;
alfa = 0.5;
XTrainNew9(:,j)=XTrain(:,j) + alfa*randn(size(XTrain(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
XTestNew9(:,j)=XTest(:,j) + alfa*randn(size(XTest(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
YTrainNew9 = YTrain;
YTestNew9 = YTest;
noTrees = 50;
RFMdl1 = TreeBagger(noTrees, XTrainNew1, YTrainNew1, 'OOBPrediction', 'On', 'Method', 'classification');
RFMdl2 = TreeBagger(noTrees, XTrainNew2, YTrainNew2, 'OOBPrediction', 'On', 'Method', 'classification');
RFMdl3 = TreeBagger(noTrees, XTrainNew3, YTrainNew3, 'OOBPrediction', 'On', 'Method', 'classification');
RFMdl4 = TreeBagger(noTrees, XTrainNew4, YTrainNew4, 'OOBPrediction', 'On', 'Method', 'classification');
RFMdl5 = TreeBagger(noTrees, XTrainNew5, YTrainNew5, 'OOBPrediction', 'On', 'Method', 'classification');
RFMdl6 = TreeBagger(noTrees, XTrainNew6, YTrainNew6, 'OOBPrediction', 'On', 'Method', 'classification');
RFMdl7 = TreeBagger(noTrees, XTrainNew7, YTrainNew7, 'OOBPrediction', 'On', 'Method', 'classification');
RFMdl8 = TreeBagger(noTrees, XTrainNew8, YTrainNew8, 'OOBPrediction', 'On', 'Method', 'classification');
RFMdl9 = TreeBagger(noTrees, XTrainNew9, YTrainNew9, 'OOBPrediction', 'On', 'Method', 'classification');
[YM1, score1] = predict(RFMdl1, XTestNew1);
[YM2, score2] = predict(RFMdl2, XTestNew2);
[YM3, score3] = predict(RFMdl3, XTestNew3);
[YM4, score4] = predict(RFMdl4, XTestNew4);
[YM5, score5] = predict(RFMdl5, XTestNew5);
[YM6, score6] = predict(RFMdl6, XTestNew6);
[YM7, score7] = predict(RFMdl7, XTestNew7);
[YM8, score8] = predict(RFMdl8, XTestNew8);
[YM9, score9] = predict(RFMdl9, XTestNew9);
YM1 = str2double(YM1);
YM2 = str2double(YM2);
YM3 = str2double(YM3);
YM4 = str2double(YM4);
YM5 = str2double(YM5);
YM6 = str2double(YM6);
YM7 = str2double(YM7);
YM8 = str2double(YM8);
YM9 = str2double(YM9);
[classif_err1, idxCorrect, idxIncorrect, samples_errors, samples_correct] = predictPerf(XTestNew1, YTestNew1,
YM1);
[classif_err2, idxCorrect, idxIncorrect, samples_errors, samples_correct] = predictPerf(XTestNew2, YTestNew2,
YM2);
[classif_err3, idxCorrect, idxIncorrect, samples_errors, samples_correct] = predictPerf(XTestNew3, YTestNew3,
YM3);
[classif_err4, idxCorrect, idxIncorrect, samples_errors, samples_correct] = predictPerf(XTestNew4, YTestNew4,
YM4);
[classif_err5, idxCorrect, idxIncorrect, samples_errors, samples_correct] = predictPerf(XTestNew5, YTestNew5,
YM5);
[classif_err6, idxCorrect, idxIncorrect, samples_errors, samples_correct] = predictPerf(XTestNew6, YTestNew6,
YM6);
[classif_err7, idxCorrect, idxIncorrect, samples_errors, samples_correct] = predictPerf(XTestNew7, YTestNew7,
YM7);
[classif_err8, idxCorrect, idxIncorrect, samples_errors, samples_correct] = predictPerf(XTestNew8, YTestNew8,
YM8);
[classif_err9, idxCorrect, idxIncorrect, samples_errors, samples_correct] = predictPerf(XTestNew9, YTestNew9,
YM9);
figure,
plot(oobError(RFMdl1))
xlabel("Number of Grown Trees")
ylabel("Out-of-Bag Classification Error")
fprintf('\nRF model least confident about the following samples in the testing dataset');
maxClassScore = max(score1);
id = cell(1, length(maxClassScore));
for i = 1:length(maxClassScore)
idClass = find(YTestNew1 == i);
id{1,i} = idClass(find(score1(idClass,i)<0.5),:);
fprintf('\n Class %g - %g%%: ', i, length(id{1,i})/length(idClass)*100);
for j = 1:length(id{1,i})
fprintf('%g ', id{1,i}(j));
end
end
fprintf('\n\n');
fprintf('\n Original Data -> RF classif err: %g %%',classif_err1);
fprintf('\n Redundant attributes -> RF classif err: %g %%',classif_err2);
fprintf('\n Redundant samples percent = %g -> RF classif err: %g %%', proc, classif_err3);
fprintf('\n Data normalized to 0 mean -> RF classif err: %g %%',classif_err4);
fprintf('\n Data normalized to [%g; %g] -> RF classif err: %g %%', [a, b], classif_err5);
fprintf('\n Outliers -> RF classif err: %g %%', classif_err6);
fprintf('\n Noise percent = %g, alpha = 0.05 -> RF classif err: %g %%',percent, classif_err7);
fprintf('\n Noise percent = %g, alpha = 0.2 -> RF classif err: %g %%',percent, classif_err8);
fprintf('\n Noise percent = %g, alpha = 0.5 -> RF classif err: %g %%',percent, classif_err9);
[Acc1, Rec1, Spec1, Prec1, F1sc1] = computePerfIndicators(YTestNew1, YM1);
[Acc2, Rec2, Spec2, Prec2, F1sc2] = computePerfIndicators(YTestNew2, YM2);
[Acc3, Rec3, Spec3, Prec3, F1sc3] = computePerfIndicators(YTestNew3, YM3);
[Acc4, Rec4, Spec4, Prec4, F1sc4] = computePerfIndicators(YTestNew4, YM4);
[Acc5, Rec5, Spec5, Prec5, F1sc5] = computePerfIndicators(YTestNew5, YM5);
[Acc6, Rec6, Spec6, Prec6, F1sc6] = computePerfIndicators(YTestNew6, YM6);
[Acc7, Rec7, Spec7, Prec7, F1sc7] = computePerfIndicators(YTestNew7, YM7);
[Acc8, Rec8, Spec8, Prec8, F1sc8] = computePerfIndicators(YTestNew8, YM8);
[Acc9, Rec9, Spec9, Prec9, F1sc9] = computePerfIndicators(YTestNew9, YM9);
indicators2 = [ mean(Acc1), mean(Rec1), mean(Spec1), mean(Prec1), mean(F1sc1);
mean(Acc2), mean(Rec2), mean(Spec2), mean(Prec2), mean(F1sc2);
mean(Acc3), mean(Rec3), mean(Spec3), mean(Prec3), mean(F1sc3);
mean(Acc4), mean(Rec4), mean(Spec4), mean(Prec4), mean(F1sc4);
mean(Acc5), mean(Rec5), mean(Spec5), mean(Prec5), mean(F1sc5);
mean(Acc6), mean(Rec6), mean(Spec6), mean(Prec6), mean(F1sc6);
mean(Acc7), mean(Rec7), mean(Spec7), mean(Prec7), mean(F1sc7);
mean(Acc8), mean(Rec8), mean(Spec8), mean(Prec8), mean(F1sc8);
mean(Acc9), mean(Rec9), mean(Spec9), mean(Prec9), mean(F1sc9);
]
save RFFinalData.mat;
diary off;


function rez = createMyDTRFData(fileNameRawData, pTest, pVal, finalDataName)
rez=0;
%% load the dataset
% Z=readmatrix(fileNameRawData); - only for numerical values
Z=readtable(fileNameRawData);
%% extract X-Y - this step relies on the structure of the csv fie
%% load the dataset
Z=readtable(fileNameRawData);
% extract X-Y - this step relies on the structure of the csv fie
X=[Z(1:end, 3), Z(1:end, 5:end)];
Y=Z(1:end, 2);
% convert to arrays for compatibility with some functions used for data analysis
X=table2array(X);
Y=table2array(Y);
% Y categorical X(:,2) si X(:,4)
[a, ~, C1] = unique(categorical(convertCharsToStrings(table2array(Z(1:end,4)))));
X = [X(:,1), C1, X(:, 2:end)];
[a, ~, Y] = unique(categorical(convertCharsToStrings(Y)));
X(isnan(X))=0;
%% create the trainign and the testing dataset
[XTrain, YTrain, XTest,YTest] = buildDatasets(X,Y,pTest,pVal);
%% save the datasets
feval(@save, finalDataName, 'X', 'Y', 'XTrain', 'YTrain', 'XTest', 'YTest');
rez=1


function [X, Y] = normalizeData(Xs, Ys, a, b)
X = [];
Y = [];
if(nargin == 2),
% normalization to 0
miu = mean(Xs);
sigma = std(Xs);
X = (Xs - repmat(miu,[size(Xs,1) 1]))./ repmat(sigma,[size(Xs,1) 1]);
else
if(nargin == 4)
% normalization to [a, b]
m = min(Xs);
M = max(Xs);
X = a+(b-a)*(Xs-repmat(m,[size(Xs,1) 1]))./repmat(M-m,[size(Xs,1) 1]);
end
end
[X, ia, ib] = unique(X,'rows','stable');
Y=Ys(ia,:);
end



function [classif_err, idxCorrect, idxIncorrect, samples_errors, samples_correct] = predictPerf(X, Y, YM)
idxCorrect = find(YM == Y);%correct responses
idxIncorrect = find(YM~= Y);%incorrect responses
samples_errors = X(idxIncorrect,:); %the samples incorrectly classified
samples_correct = X(idxCorrect,:); %the samples correctly classified
classif_err = numel(find(YM~= Y))/numel(Y)*100;
function [Acc, Rec, Spec, Prec, F1sc] = computePerfIndicators(Y, YM)
labelClasses=unique(Y);
for j=1:numel(labelClasses)
idxTP=((YM == labelClasses(j)) & (Y==labelClasses(j)));
idxTN=((YM ~= labelClasses(j)) & (Y~=labelClasses(j)));
idxFP=((YM == labelClasses(j)) & (Y~=labelClasses(j)));
idxFN=((YM ~= labelClasses(j)) & (Y==labelClasses(j)));
TP=sum(idxTP);
TN=sum(idxTN);
FP=sum(idxFP);
FN=sum(idxFN);
Acc(j)=(TP+TN)/(TP+TN+FN+FP);
Rec(j) = TP / (TP+FN);
Spec(j) = TN/(FP+TN);
Prec(j) = TP/(TP+FP);
F1sc(j) = 2*TP/(2*TP + FN +FP);
%+ other indicators, like recall, precision, F1-score
Acc(isnan(Acc)) = 0;
Rec(isnan(Rec)) = 0;
Prec(isnan(Prec)) = 0;
Spec(isnan(Spec)) = 0;
F1sc(isnan(F1sc)) = 0;
end
end
