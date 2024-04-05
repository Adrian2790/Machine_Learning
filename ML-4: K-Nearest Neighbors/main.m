close all;
clear all;
clc;
diary logFileName_P1.txt;
% % rez = createMyKNNData('Music_origin.csv', 0.2, 0, 'myKNNData.mat'); % se decomenteaza la
prima rulare
clear all;
load 'myKNNData.mat';
% simplified data
markers={'ro', 'b*', 'g+', 'yx'};% the no. of markers = the no. of classes
[Xsn0, Ysn0] = normalizeData(Xs, Ys); % normalized to 0
a = 2;
b = 3;
[Xsnab, Ysnab] = normalizeData(Xs, Ys, a, b);
% exemple indepartate
% % exemple îndepărtate: 2*M, m-2*M, etc.
m1=min(Xs); M1=max(Xs);
Xso = [Xs; m1-2*M1; m1-3*M1; m1-4*M1; 2*M1; 3*M1; 4*M1];
Yso = [Ys; 2; 2; 2; 2; 2; 2];
fprintf('\n Original data Ys:\n');
tabulate(Ys)
fprintf('\n Original data 0 normalized Ysn0:\n');
tabulate(Ysn0)
fprintf('\n Original data normalized to [%g, %g] Ysnab:\n', [a,b]);
tabulate(Ysnab)
figure;
subplot(2,2,1);
listLabels = unique(Ys); %list of labels used for classes
for i=1:numel(listLabels), %for each class
idxLabel = find(Ys==listLabels(i)); %the indices of the samples belonging to the class listLabels(i)
plot(Xs(idxLabel,1),Xs(idxLabel,2),markers{i})
hold on;
end
title('Original Xs');
subplot(2,2,2);
listLabels = unique(Yso); %list of labels used for classes
for i=1:numel(listLabels), %for each class
idxLabel = find(Yso==listLabels(i)); %the indices of the samples belonging to the class listLabels(i)
plot(Xso(idxLabel,1),Xso(idxLabel,2),markers{i})
hold on;
end
title('Original Xs with outliers');
subplot(2,2,3);
listLabels = unique(Ysn0); %list of labels used for classes
for i=1:numel(listLabels), %for each class
idxLabel = find(Ysn0==listLabels(i)); %the indices of the samples belonging to the class listLabels(i)
plot(Xsn0(idxLabel,1),Xsn0(idxLabel,2),markers{i})
hold on;
end
title('After normalization to 0');
subplot(2,2,4);
for i=1:numel(listLabels), %for each class
listLabels = unique(Ysnab); %list of labels used for classes
idxLabel = find(Ysnab==listLabels(i)); %the indices of the samples belonging to the class listLabels(i)
plot(Xsnab(idxLabel,1),Xsnab(idxLabel,2),markers{i})
hold on;
end
title(strcat('After normalization to [',num2str(a),', ', num2str(b), ']'));
m1=min(Xtrs); M1=max(Xtrs);
Xtrso = [Xtrs; m1-2*M1; m1-3*M1; m1-4*M1; 2*M1; 3*M1; 4*M1];
Ytrso = [Ytrs; 2; 2; 2; 2; 2; 2];
m2=min(Xtes); M2=max(Xtes);
Xteso = [Xtes; m2-2*M2; m2-3*M2; m2-4*M2; 2*M2; 3*M2; 4*M2];
Yteso = [Ytes; 2; 2; 2; 2; 2; 2];
[Xtrsn, Ytrsn] = normalizeData(Xtrs, Ytrs); % normalized to 0
[Xtesn, Ytesn] = normalizeData(Xtes, Ytes); % normalized to 0
[Xtrsnab, Ytrsnab] = normalizeData(Xtrs, Ytrs, a, b); % normalized to a b
[Xtesnab, Ytesnab] = normalizeData(Xtes, Ytes, a, b); % normalized to a b
XTrains1 = Xtrs;
YTrains1 = Ytrs;
XTests1 = Xtes;
YTests1 = Ytes;
XTrains2 = Xtrso;
YTrains2 = Ytrso;
XTests2 = Xteso;
YTests2 = Yteso;
XTrains3 = Xtrsn;
YTrains3 = Ytrsn;
XTests3 = Xtesn;
YTests3 = Ytesn;
XTrains4 = Xtrsnab;
YTrains4 = Ytrsnab;
XTests4 = Xtesnab;
YTests4 = Ytesnab;
Mdl1 = fitcknn(XTrains1, YTrains1, 'NumNeighbors', 5);
Mdl2 = fitcknn(XTrains2, YTrains2, 'NumNeighbors', 5);
Mdl3 = fitcknn(XTrains3, YTrains3, 'NumNeighbors', 5);
Mdl4 = fitcknn(XTrains4, YTrains4, 'NumNeighbors', 5);
[YM1tr, classif_err1tr, idxCorrect, idxIncorrect, samples_errors, samples_correct] =
predictModel(Mdl1, XTrains1, YTrains1);
[YM1te, classif_err1te, idxCorrect, idxIncorrect1, samples_errors1, samples_correct] =
predictModel(Mdl1, XTests1, YTests1);
[YM2tr, classif_err2tr, idxCorrect, idxIncorrect, samples_errors, samples_correct] =
predictModel(Mdl2, XTrains2, YTrains2);
[YM2te, classif_err2te, idxCorrect, idxIncorrect2, samples_errors2, samples_correct] =
predictModel(Mdl2, XTests2, YTests2);
[YM3tr, classif_err3tr, idxCorrect, idxIncorrect, samples_errors, samples_correct] =
predictModel(Mdl3, XTrains3, YTrains3);
[YM3te, classif_err3te, idxCorrect, idxIncorrect3, samples_errors3, samples_correct] =
predictModel(Mdl3, XTests3, YTests3);
[YM4tr, classif_err4tr, idxCorrect, idxIncorrect, samples_errors, samples_correct] =
predictModel(Mdl4, XTrains4, YTrains4);
[YM4te, classif_err4te, idxCorrect, idxIncorrect4, samples_errors4, samples_correct] =
predictModel(Mdl4, XTests4, YTests4);
fprintf('\n Original Data:');
fprintf('\n Original Data - kNN classif err train: %g', classif_err1tr);
fprintf('\n Original Data - kNN classif err test: %g', classif_err1te);
fprintf('\n Original Data with outliers:');
fprintf('\n Original Data - kNN classif err train: %g', classif_err2tr);
fprintf('\n Original Data - kNN classif err test: %g', classif_err2te);
fprintf('\n Normalized to 0:');
fprintf('\n 0-Norm Data - kNN classif err train: %g', classif_err3tr);
fprintf('\n 0-Norm Data - kNN classif err test: %g', classif_err3te);
fprintf('\n Normalized to [%g, %g]:',[a,b]);
fprintf('\n [%g, %g]-Norm Data - kNN classif err train: %g', [a, b], classif_err4tr);
fprintf('\n [%g ,%g]-Norm Data - kNN classif err test: %g', [a, b], classif_err4te);
figure;
subplot(2,2,1),
listLabels1 = unique(YTests1(idxIncorrect1,:)); %list of labels used for classes
for i=1:numel(listLabels1) %for each class
idxLabel = find(YTests1(idxIncorrect1,:)==listLabels1(i)); %the indices of the samples belonging to the
class listLabels(i)
plot(samples_errors1(idxLabel,1), samples_errors1(idxLabel,2), markers{i}), hold on;
end
title('Incorrectly classified samples on test in original data');
legend('1', '2', '3', '4');
subplot(2,2,2),
listLabels2 = unique(YTests2(idxIncorrect2,:)); %list of labels used for classes
for i=1:numel(listLabels2) %for each class
idxLabel = find(YTests2(idxIncorrect2,:)==listLabels2(i)); %the indices of the samples belonging to the
class listLabels(i)
plot(samples_errors2(idxLabel,1), samples_errors2(idxLabel,2), markers{i}), hold on;
end
title('Incorrectly classified samples on test in original data with outliers');
legend('1', '2', '3', '4');
subplot(2,2,3),
listLabels3 = unique(YTests3(idxIncorrect3,:)); %list of labels used for classes
for i=1:numel(listLabels3) %for each class
idxLabel = find(YTests3(idxIncorrect3,:)==listLabels3(i)); %the indices of the samples belonging to the
class listLabels(i)
plot(samples_errors3(idxLabel,1), samples_errors3(idxLabel,2), markers{i}), hold on;
end
title('Incorrectly classified samples on test in data normalized to 0');
legend('1', '2', '3', '4');
subplot(2,2,4),
listLabels4 = unique(YTests4(idxIncorrect4,:)); %list of labels used for classes
for i=1:numel(listLabels4) %for each class
idxLabel = find(YTests4(idxIncorrect4,:)==listLabels4(i)); %the indices of the samples belonging to the
class listLabels(i)
plot(samples_errors4(idxLabel,1), samples_errors4(idxLabel,2), markers{i}), hold on;
end
title('Incorrectly classified samples on test in data normalized to [%g, %g]',[a,b]);
legend('1', '2', '3', '4');
save knnP1finals.mat;
diary off
/////////////////////////////////////////////////////////////////////////////////////////////////////////
Original data Ys:
Value Count Percent
0 36 3.40%
1 91 8.59%
2 223 21.06%
3 709 66.95%
Original data 0 normalized Ysn0:
Value Count Percent
0 36 3.40%
1 91 8.59%
2 223 21.06%
3 709 66.95%
Original data normalized to [2, 3] Ysnab:
Value Count Percent
0 36 3.40%
1 91 8.59%
2 223 21.06%
3 709 66.95%
Original Data:
Original Data - kNN classif err train: 28.066
Original Data - kNN classif err test: 36.4929
Original Data with outliers:
Original Data - kNN classif err train: 27.8689
Original Data - kNN classif err test: 35.9447
Normalized to 0:
0-Norm Data - kNN classif err train: 28.184
0-Norm Data - kNN classif err test: 39.3365
Normalized to [2, 3]:
[2, 3]-Norm Data - kNN classif err train: 28.066
[2 ,3]-Norm Data - kNN classif err test: 34.5972

