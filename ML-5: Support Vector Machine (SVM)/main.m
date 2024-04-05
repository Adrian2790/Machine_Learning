//Select the samples belonging to only two classes (all attributes) and solve a binary classification problem with SVM.

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


close all;
clear all;
clc;
diary logFileName_P2_SVM.txt;
% rez = createMySVMData('Music_origin.csv', 0.2, 0, 'mySVMData.mat'); % se
decomenteaza la prima rulare
load 'mySVMData.mat';
% Xt si Yt
tabulate(Yt)
[Xtrs, Ytrs, Xtes, Ytes] = buildDatasets(Xt,Yt,0.2,0);
Mdl0 = fitcsvm(Xtrs,Ytrs);
Mdl1 = fitcsvm(Xtrs,Ytrs, 'KernelFunction', 'gaussian', 'Standardize', true);
Mdl2 = fitcsvm(Xtrs,Ytrs,'Standardize',true,'KernelFunction','RBF',...
'KernelScale','auto');
[YM0, classif_err0, idxCorrect0, idxIncorrect0, samples_errors0, samples_correct0] =
predictModel(Mdl0, Xtes, Ytes);
[Acc0, Rec0, Spec0, Prec0, F1sc0] = computePerfIndicators(Mdl0, Xtes, Ytes);
fprintf('\nFull columns, 2 classes, simple config on fitcsvm classif err: %g %%',classif_err0);
[YM1, classif_err1, idxCorrect1, idxIncorrect1, samples_errors1, samples_correct1] =
predictModel(Mdl1, Xtes, Ytes);
[Acc1, Rec1, Spec1, Prec1, F1sc1] = computePerfIndicators(Mdl1, Xtes, Ytes);
fprintf('\nFull columns, 2 classes, Gaussian kernel, standardize true, on fitcsvm classif err: %g
%%',classif_err1);
[YM2, classif_err2, idxCorrect2, idxIncorrect2, samples_errors2, samples_correct2] =
predictModel(Mdl2, Xtes, Ytes);
[Acc2, Rec2, Spec2, Prec2, F1sc2] = computePerfIndicators(Mdl2, Xtes, Ytes);
fprintf('\nFull columns, 2 classes, RBF kernel, standardize true, auto on fitcsvm classif err: %g
%%',classif_err2);
[mean(Acc0), mean(Prec0), mean(Rec0), mean(Spec0), mean(F1sc0);
mean(Acc1), mean(Prec1), mean(Rec1), mean(Spec1), mean(F1sc1);
mean(Acc2), mean(Prec2), mean(Rec2), mean(Spec2), mean(F1sc2)
]
save mySVMData2.mat;
XTrainNew=[Xtrs, Xtrs(:,end) Xtrs(:,end)];
XTestNew=[Xtes, Xtes(:,end) Xtes(:,end)];
YTrainNew = Ytrs;
YTestNew = Ytes;
Mdl3 = fitcsvm(XTrainNew, YTrainNew, 'Standardize', true, 'KernelFunction', 'RBF',
'KernelScale', 'auto');
[YM3, classif_err3, idxCorrect3, idxIncorrect3, samples_errors3, samples_correct3] =
predictModel(Mdl3, XTestNew, YTestNew);
[Acc3, Rec3, Spec3, Prec3, F1sc3] = computePerfIndicators(Mdl3, XTestNew, YTestNew);
fprintf('\nRedundant attributes - Full columns, 2 classes, RBF kernel, standardize true, auto on
fitcsvm classif err: %g %%',classif_err3);
j = sort(randperm(size(Xtrs,2), floor(0.25*size(Xtrs,2))));
alfa = 0.05;
XTrainNew(:,j)=Xtrs(:,j) + alfa*randn(size(Xtrs(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
XTestNew(:,j)=Xtes(:,j) + alfa*randn(size(Xtes(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
YTrainNew = Ytrs;
YTestNew = Ytes;
Mdl4 = fitcsvm(XTrainNew, YTrainNew, 'Standardize', true, 'KernelFunction', 'RBF',
'KernelScale', 'auto');
[YM4, classif_err4, idxCorrect4, idxIncorrect4, samples_errors4, samples_correct4] =
predictModel(Mdl4, XTestNew, YTestNew);
[Acc4, Rec4, Spec4, Prec4, F1sc4] = computePerfIndicators(Mdl4, XTestNew, YTestNew);
fprintf('\nNoisy alpha %g attributes - Full columns, 2 classes, RBF kernel, standardize true,
auto on fitcsvm classif err: %g %%', alfa, classif_err4);
alfa = 0.2;
XTrainNew(:,j)=Xtrs(:,j) + alfa*randn(size(Xtrs(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
XTestNew(:,j)=Xtes(:,j) + alfa*randn(size(Xtes(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
YTrainNew = Ytrs;
YTestNew = Ytes;
Mdl5 = fitcsvm(XTrainNew, YTrainNew, 'Standardize', true, 'KernelFunction', 'RBF',
'KernelScale', 'auto');
[YM5, classif_err5, idxCorrect5, idxIncorrect5, samples_errors5, samples_correct5] =
predictModel(Mdl5, XTestNew, YTestNew);
[Acc5, Rec5, Spec5, Prec5, F1sc5] = computePerfIndicators(Mdl5, XTestNew, YTestNew);
fprintf('\nNoisy alpha %g attributes - Full columns, 2 classes, RBF kernel, standardize true,
auto on fitcsvm classif err: %g %%', alfa, classif_err5);
alfa = 0.5;
XTrainNew(:,j)=Xtrs(:,j) + alfa*randn(size(Xtrs(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
XTestNew(:,j)=Xtes(:,j) + alfa*randn(size(Xtes(:,j))); %, cu alfa = 0.05, 0.2, 0.5.
YTrainNew = Ytrs;
YTestNew = Ytes;
Mdl6 = fitcsvm(XTrainNew, YTrainNew, 'Standardize', true, 'KernelFunction', 'RBF',
'KernelScale', 'auto');
[YM6, classif_err6, idxCorrect6, idxIncorrect6, samples_errors6, samples_correct6] =
predictModel(Mdl6, XTestNew, YTestNew);
[Acc6, Rec6, Spec6, Prec6, F1sc6] = computePerfIndicators(Mdl6, XTestNew, YTestNew);
fprintf('\nNoisy alpha %g attributes - Full columns, 2 classes, RBF kernel, standardize true,
auto on fitcsvm classif err: %g %%',alfa, classif_err6);
% normalizare la 0
[XTrainNew, YTrainNew] = normalizeData(Xtrs, Ytrs); % normalized to 0
[XTestNew, YTestNew] = normalizeData(Xtes, Ytes); % normalized to 0
Mdl7 = fitcsvm(XTrainNew, YTrainNew, 'Standardize', true, 'KernelFunction', 'RBF',
'KernelScale', 'auto');
[YM7, classif_err7, idxCorrect7, idxIncorrect7, samples_errors7, samples_correct7] =
predictModel(Mdl7, XTestNew, YTestNew);
[Acc7, Rec7, Spec7, Prec7, F1sc7] = computePerfIndicators(Mdl7, XTestNew, YTestNew);
fprintf('\nNormalized attributes to 0 - Full columns, 2 classes, RBF kernel, standardize true,
auto on fitcsvm classif err: %g %%',classif_err7);
% normalizare la [0, 1]
[XTrainNew, YTrainNew] = normalizeData(Xtrs, Ytrs, 0, 1); % normalized to 0,1
[XTestNew, YTestNew] = normalizeData(Xtes, Ytes, 0, 1); % normalized to 0,1
Mdl8 = fitcsvm(XTrainNew, YTrainNew, 'Standardize', true, 'KernelFunction', 'RBF',
'KernelScale', 'auto');
[YM8, classif_err8, idxCorrect8, idxIncorrect8, samples_errors8, samples_correct8] =
predictModel(Mdl8, XTestNew, YTestNew);
[Acc8, Rec8, Spec8, Prec8, F1sc8] = computePerfIndicators(Mdl8, XTestNew, YTestNew);
fprintf('\nNormalized attributes [0,1] - Full columns, 2 classes, RBF kernel, standardize true,
auto on fitcsvm classif err: %g %%',classif_err8);
% exemple redundante
[nl1, nc1] = size(Xtrs);
[nl2, nc2] = size(Xtes);
proc = 0.2; % 20%
i = sort(randperm(min(nl1, nl2), floor(proc*min(nl1,nl2))));
XTrainNew = Xtrs(i,:);
YTrainNew = Ytrs(i,:);
XTestNew = Xtes;
YTestNew = Ytes;
Mdl9 = fitcsvm(XTrainNew, YTrainNew, 'Standardize', true, 'KernelFunction', 'rbf'); %,
'RemoveDuplicates','on');
[YM9, classif_err9, idxCorrect9, idxIncorrect9, samples_errors9, samples_correct9] =
predictModel(Mdl9, XTestNew, YTestNew);
[Acc9, Rec9, Spec9, Prec9, F1sc9] = computePerfIndicators(Mdl9, XTestNew, YTestNew);
fprintf('\nRedundant samples - Full columns, 2 classes, RBF kernel, standardize true, auto on
fitcsvm classif err: %g %%',classif_err9);
% exemple indepartate
m1=min(Xtrs); M1=max(Xtrs);
m2=min(Xtes); M2=max(Xtes);
% % exemple îndepărtate: 2*M, m-2*M, etc.
XTrainNew = [Xtrs; m1-2*M1; 2*M1];
YTrainNew = [Ytrs; 2; 2];
XTestNew = [Xtes; m2-2*M2; 2*M2];
YTestNew = [Ytes; 2; 2];
Mdl10 = fitcsvm(XTrainNew, YTrainNew, 'Standardize', true, 'KernelFunction', 'RBF',
'KernelScale', 'auto');
[YM10, classif_err10, idxCorrect10, idxIncorrect10, samples_errors10, samples_correct10] =
predictModel(Mdl10, XTestNew, YTestNew);
[Acc10, Rec10, Spec10, Prec10, F1sc10] = computePerfIndicators(Mdl10, XTestNew,
YTestNew);
fprintf('\nFar away samples - Full columns, 2 classes, RBF kernel, standardize true, auto on
fitcsvm classif err: %g %%',classif_err10);
[mean(Acc3), mean(Prec3), mean(Rec3), mean(Spec3), mean(F1sc3);
mean(Acc4), mean(Prec4), mean(Rec4), mean(Spec4), mean(F1sc4);
mean(Acc5), mean(Prec5), mean(Rec5), mean(Spec5), mean(F1sc5);
mean(Acc6), mean(Prec6), mean(Rec6), mean(Spec6), mean(F1sc6);
mean(Acc7), mean(Prec7), mean(Rec7), mean(Spec7), mean(F1sc7);
mean(Acc8), mean(Prec8), mean(Rec8), mean(Spec8), mean(F1sc8);
mean(Acc9), mean(Prec9), mean(Rec9), mean(Spec9), mean(F1sc9);
mean(Acc10), mean(Prec10), mean(Rec10), mean(Spec10), mean(F1sc10)
]
diary off;
