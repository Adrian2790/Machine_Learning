//Considering the full set of input attributes, solve the classification application with fitcnb.
//Recommended investigations:
//- apply fitcnb for the following configurations and compare the results (accuracy, recall, etc.):
//o using discretization with different configurations of bins
//o using normal or kernel distribution, without discretization
////////////////////////////////////////////////////////////////////////////////////////////////////////////
close all;
clear all;
clc;
diary logFileName_P2_NB.txt;
load 'myNBData.mat';
% % ids = find(Y~=1);
% % idns = find(Y==1);
% % XX = X(ids,:);
% % YY = Y(ids,:);
XX=X;
YY=Y;
X1=zeros(size(XX)); X2=zeros(size(XX)); X3=zeros(size(XX));
for i = 1:size(XX,2)
X1(:,i) = discretize(XX(:,i), ceil((max(XX(:,i))-min(XX(:,i)))*2));
X2(:,i) = discretize(XX(:,i), ceil(max(XX(:,i))-min(XX(:,i)))*4);
X3(:,i) = discretize(XX(:,i), ceil(max(XX(:,i))-min(XX(:,i)))*8);
end
[Xd1, ia, ib]=unique(X1,'rows','stable');
Yd1 = YY(ia,:);
[Xd2, ia, ib]=unique(X2,'rows','stable');
Yd2 = YY(ia,:);
[Xd3, ia, ib]=unique(X3,'rows','stable');
Yd3 = YY(ia,:);
[XTrain1, YTrain1, XTest1, YTest1] = buildDatasets(Xd1, Yd1, 0.2, 0);
[XTrain2, YTrain2, XTest2, YTest2] = buildDatasets(Xd2, Yd2, 0.2, 0);
[XTrain3, YTrain3, XTest3, YTest3] = buildDatasets(Xd3, Yd3, 0.2, 0);
Mdl1 = fitcnb(XTrain1, YTrain1);
Mdl2 = fitcnb(XTrain2, YTrain2);
Mdl3 = fitcnb(XTrain3, YTrain3);
Mdl4 = fitcnb(XTrain, YTrain);
Mdl5 = fitcnb(XTrain, YTrain, 'DistributionNames', 'kernel', 'Kernel', 'normal');
YMTest1 = predict(Mdl1, XTest1);
YMTest2 = predict(Mdl2, XTest2);
YMTest3 = predict(Mdl3, XTest3);
YMTest4 = predict(Mdl4, XTest);
YMTest5 = predict(Mdl5, XTest);
[Acc1, Rec1, Spec1, Prec1, F1sc1, classif_err1] = computePerfIndicators(YTest1, YMTest1);
[Acc2, Rec2, Spec2, Prec2, F1sc2, classif_err2] = computePerfIndicators(YTest2, YMTest2);
[Acc3, Rec3, Spec3, Prec3, F1sc3, classif_err3] = computePerfIndicators(YTest3, YMTest3);
[Acc4, Rec4, Spec4, Prec4, F1sc4, classif_err4] = computePerfIndicators(YTest, YMTest4);
[Acc5, Rec5, Spec5, Prec5, F1sc5, classif_err5] = computePerfIndicators(YTest, YMTest5);
[mean(Acc1), mean(Rec1), mean(Spec1), mean(Prec1), mean(F1sc1);
mean(Acc2), mean(Rec2), mean(Spec2), mean(Prec2), mean(F1sc2);
mean(Acc3), mean(Rec3), mean(Spec3), mean(Prec3), mean(F1sc3);
mean(Acc4), mean(Rec4), mean(Spec4), mean(Prec4), mean(F1sc4);
mean(Acc5), mean(Rec5), mean(Spec5), mean(Prec5), mean(F1sc5);
]
fprintf('\nAll data columns, disc (max-min)*2, fitcnb classif err: %g %%', classif_err1);
fprintf('\nAll data columns, disc (max-min)*4, fitcnb classif err: %g %%', classif_err2);
fprintf('\nAll data columns, disc (max-min)*8, fitcnb classif err: %g %%', classif_err3);
fprintf('\nAll data columns, no disc, fitcnb normal distrib classif err: %g %%',classif_err4);
fprintf('\nAll data columns, no disc, fitcnb gaussian kernel classif err: %g %%',classif_err5);
////////////////////////////////////////////////////////////////////////////////////////////////////////////
ans =
0.9573 0.6157 0.9519 0.5533 0.5679
0.9451 0.7161 0.9318 0.6060 0.6072
0.9642 0.6427 0.9679 0.4948 0.5214
0.9642 0.6334 0.9512 0.6086 0.6198
0.9577 0.5829 0.9277 0.5507 0.5525
////////////////////////////////////////////////////////////////////////////////////////////////////////////
All data columns, disc (max-min)*2, fitcnb classif err: 8.94309 %
All data columns, disc (max-min)*4, fitcnb classif err: 11.3821 %
All data columns, disc (max-min)*8, fitcnb classif err: 8.94309 %
All data columns, no disc, fitcnb normal distrib classif err: 8.94309 %
All data columns, no disc, fitcnb gaussian kernel classif err: 10.5691 %
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//- using the best configuration from the previous experiments, compare the results for the
//following training datasets:
//o classification with XTrain
//o classification with a training dataset including redundant attributes - e. g. the last
//attribute copied several times:
//XtrainNew=[Xtrain, Xtrain(:,end) Xtrain(:,end)]
//o classification with noisy data:
//XtrainNew(:,j)=XTrain(:,j) + alpha*randn(size(XTrain(:,j)),
//with alpha = 0.05, 0.2, 0.5.
//;● this modification should be applied only to numerical attributes!

% duplicate last column 2 times
XTrain6=[XTrain, XTrain(:,end) XTrain(:,end)];
XTest6=[XTest, XTest(:,end) XTest(:,end)];
YTrain6 = YTrain;
YTest6 = YTest;
percent = 0.30;
j = sort(randperm(size(XTrain,2), floor(percent*size(XTrain,2))));
alfa = 0.05; %, cu alfa = 0.05, 0.2, 0.5.
XTrain7(:,j)=XTrain(:,j) + alfa*randn(size(XTrain(:,j)));
XTest7(:,j)=XTest(:,j) + alfa*randn(size(XTest(:,j)));
YTrain7 = YTrain;
YTest7 = YTest;
alfa = 0.2;
XTrain8(:,j)=XTrain(:,j) + alfa*randn(size(XTrain(:,j)));
XTest8(:,j)=XTest(:,j) + alfa*randn(size(XTest(:,j)));
YTrain8 = YTrain;
YTest8 = YTest;
alfa = 0.5;
XTrain9(:,j)=XTrain(:,j) + alfa*randn(size(XTrain(:,j)));
XTest9(:,j)=XTest(:,j) + alfa*randn(size(XTest(:,j)));
YTrain9 = YTrain;
YTest9 = YTest;
% % Mdl6 = fitcnb(XTrain6, YTrain6);
% % Mdl7 = fitcnb(XTrain7, YTrain7);
% % Mdl8 = fitcnb(XTrain8, YTrain8);
% % Mdl9 = fitcnb(XTrain9, YTrain9);
Mdl6 = fitcnb(XTrain6, YTrain6, 'DistributionNames', 'kernel', 'Kernel', 'normal');
Mdl7 = fitcnb(XTrain7, YTrain7, 'DistributionNames', 'kernel', 'Kernel', 'normal');
Mdl8 = fitcnb(XTrain8, YTrain8, 'DistributionNames', 'kernel', 'Kernel', 'normal');
Mdl9 = fitcnb(XTrain9, YTrain9, 'DistributionNames', 'kernel', 'Kernel', 'normal');
YMTest6 = predict(Mdl6, XTest6);
YMTest7 = predict(Mdl7, XTest7);
YMTest8 = predict(Mdl8, XTest8);
YMTest9 = predict(Mdl9, XTest9);
[Acc6, Rec6, Spec6, Prec6, F1sc6, classif_err6] = computePerfIndicators(YTest6, YMTest6);
[Acc7, Rec7, Spec7, Prec7, F1sc7, classif_err7] = computePerfIndicators(YTest7, YMTest7);
[Acc8, Rec8, Spec8, Prec8, F1sc8, classif_err8] = computePerfIndicators(YTest8, YMTest8);
[Acc9, Rec9, Spec9, Prec9, F1sc9, classif_err9] = computePerfIndicators(YTest9, YMTest9);
[mean(Acc5), mean(Rec5), mean(Spec5), mean(Prec5), mean(F1sc5);
mean(Acc6), mean(Rec6), mean(Spec6), mean(Prec6), mean(F1sc6);
mean(Acc7), mean(Rec7), mean(Spec7), mean(Prec7), mean(F1sc7);
mean(Acc8), mean(Rec8), mean(Spec8), mean(Prec8), mean(F1sc8);
mean(Acc9), mean(Rec9), mean(Spec9), mean(Prec9), mean(F1sc9);
]
fprintf('\nAll data columns, no disc, fitcnb gaussian kernel, classif err: %g %%',classif_err5);
fprintf('\nAll data columns, no disc, fitcnb gaussian kernel, duplicate columns classif err: %g
%%',classif_err6);
fprintf('\nAll data columns, no disc, fitcnb gaussian kernel, noise a=0.05 classif err: %g
%%',classif_err7);
fprintf('\nAll data columns, no disc, fitcnb gaussian kernel, noise a=0.2 classif err: %g %%',classif_err8);
fprintf('\nAll data columns, no disc, fitcnb gaussian kernel, noise a=0.5 classif err: %g %%',classif_err9);
////////////////////////////////////////////////////////////////////////////////////////////////////////////
ans =
0.9577 0.5829 0.9277 0.5507 0.5525
0.9577 0.5829 0.9379 0.5592 0.5445
0.9512 0.3829 0.8840 0.3341 0.3562
0.9480 0.3810 0.8823 0.3339 0.3552
0.9447 0.3791 0.8805 0.3505 0.3638
////////////////////////////////////////////////////////////////////////////////////////////////////////////
All data columns, no disc, fitcnb gaussian kernel, classif err: 10.5691 %
All data columns, no disc, fitcnb gaussian kernel, duplicate columns classif err: 10.5691 %
All data columns, no disc, fitcnb gaussian kernel, noise a=0.05 classif err: 12.1951 %
All data columns, no disc, fitcnb gaussian kernel, noise a=0.2 classif err: 13.0081 %
All data columns, no disc, fitcnb gaussian kernel, noise a=0.5 classif err: 13.8211 
