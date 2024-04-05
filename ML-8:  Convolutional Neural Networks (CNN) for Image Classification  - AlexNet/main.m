function [Acc, Rec, Spec, Prec, Sens, F1sc, TP, TN, FP, FN, classif_err] = computePerfIndicatorsCNN(Mdl, YTrue, YPredicted)
   labelClasses=unique(YTrue);
   for j=1:numel(labelClasses)
   
       idxTP=((YPredicted == labelClasses(j)) & (YTrue==labelClasses(j)));
       idxTN=((YPredicted ~= labelClasses(j)) & (YTrue~=labelClasses(j)));
       idxFP=((YPredicted == labelClasses(j)) & (YTrue~=labelClasses(j)));
       idxFN=((YPredicted ~= labelClasses(j)) & (YTrue==labelClasses(j)));
  
       TP(j)=sum(idxTP);
       TN(j)=sum(idxTN);
       FP(j)=sum(idxFP);
       FN(j)=sum(idxFN);
     
       Acc(j)=(TP(j)+TN(j))/(TP(j)+TN(j)+FN(j)+FP(j));
       Rec(j) = TP(j)/(TP(j)+FN(j));
       Spec(j) = TN(j)/(FP(j)+TN(j));
       Prec(j) = TP(j)/(TP(j)+FP(j));
       F1sc(j) = (2*TP(j))/(2*TP(j) + FN(j) +FP(j));
       Sens(j) = TP(j)/(TP(j)+FN(j));
       %+ other indicators, like recall, precision, F1-score
   end
   idxCorrect = find(YPredicted == YTrue);%correct responses
   idxIncorrect = find(YPredicted~= YTrue);%incorrect responses
  
   classif_err = numel(find(YPredicted~= YTrue))/numel(YTrue)*100;
end
% createInitialData.m
clear all;
close all;
clc;
%% TRAINING AND VALIDATION DATASETS
% the size of the input images
inputSize = [227 227 3];
% indicate the path to the trainingaand validation images
pathImagesTrain='/MATLAB Drive/ML_AlexNet/RMN_BrainCancer/';
%         full path of the folder with training images
%         the folder should include a separate subfolder for each class
%         the images should have the size indicated by inputSize
% create the datastore with the training and validation images
imds = imageDatastore(pathImagesTrain, ... 
   'IncludeSubfolders',true, ...
   'LabelSource','foldernames');
% split the dataset into training and validation datasets
[imdsTrain, imdsValidation] = splitEachLabel(imds,0.7,'randomized');
% obtain information about the training dataset
numTrainImages = numel(imdsTrain); % the number of trainig images
numClasses = numel(categories(imdsTrain.Labels)); %the number of classes
% augment the training and validation dataset
pixelRange = [-30 30];  
imageAugmenter = imageDataAugmenter( ...
   'RandXReflection',true, ...
   'RandXTranslation',pixelRange, ...
   'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
   'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
%% TESTING DATASETS
% indicate the path to the testing dataset
pathImagesTest='/MATLAB Drive/ML_AlexNet/RMN_BrainCancer/';
%         full path of the folder with training images
%         the folder should include a separate subfolder for each class
%         the images should have the size indicated by inputSize
% create the datastore for the testing dataset     
imdsTest = imageDatastore(pathImagesTest, ... 
   'IncludeSubfolders',true, ...
   'LabelSource','foldernames');    
save initialAlexNetData.mat;
% mainAlexNet.m
%% CNN for image classification
%+++++++++++++++++++++++++++++++++++
clear all;
close all;
clc;
%% PARAMETERS
% the name of the file where the trained CNN is saved
nameFile_Results='rezCNN.mat';
% training parameters
MBS = 5;% mini batch size
NEP = 1; % number of epochs
%% TRAINING AND VALIDATION DATASET
load initialAlexNetData.mat;
%% DESIGN THE ARCHITECTURE
% load the pretrained model
net = alexnet;
% take the layers for transfer of learning
layersTransfer = net.Layers(1:end-3);
% create the new architecture: the last fully connected layer is configured for the necessary number of classes
layersNew = [
   layersTransfer   
   fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
   softmaxLayer
   classificationLayer];
%% TRAIN THE CNN
% indicate the training parameters
options = trainingOptions('sgdm', ...
   'MiniBatchSize',MBS,...           
   'MaxEpochs',NEP, ...     
   'InitialLearnRate',1e-4, ... 
   'ValidationData',augimdsValidation, ...
   'ValidationFrequency',3, ...
   'ValidationPatience',Inf, ...
   'Verbose',false, ...
   'Plots','training-progress');
                
% train the model
netTransfer = trainNetwork(augimdsTrain,layersNew,options);
% save the trained model
feval(@save, nameFile_Results, 'netTransfer');
%% VERIFY THE RESULTS
% validation - responses and accuracy
[YPredValidation,scoresValidation] = classify(netTransfer, imdsValidation);
accuracyValidation = mean(YPredValidation == imdsValidation.Labels) 
% training - responses and accuracy
[YPredTrain,scoresTrain] = classify(netTransfer, imdsTrain); 
accuracyTrain = mean(YPredTrain == imdsTrain.Labels) 
% testing- responses and accuracy
[YPredTest,scoresTest] = classify(netTransfer, imdsTest); 
accuracyTest = mean(YPredTest == imdsTest.Labels) 
[Acc, Rec, Spec, Prec, Sens, F1sc, TP, TN, FP, FN, classif_err] = computePerfIndicatorsCNN(netTransfer, imdsTest.Labels, YPredTest);
layer=16;
name = netTransfer.Layers(layer).Name
channels = 1:36;
I = deepDreamImage(netTransfer,name,channels, ...
  'PyramidLevels',1);
figure(1)
I = imtile(I,'ThumbnailSize',[64 64]);
imshow(I)
title(['Layer ',name,' Features'],'Interpreter','none')
save nameFile_Results;
% save all_10_10_CNN.mat;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

% mainHalfImages.m
clear all;
close all;
clc;
%% TRAINING AND VALIDATION DATASETS
% the size of the input images
inputSize = [227 227 3];
% indicate the path to the trainingaand validation images
pathImagesTrain='/MATLAB Drive/ML_AlexNet/RMN_BrainCancerHalf/';
%         full path of the folder with training images
%         the folder should include a separate subfolder for each class
%         the images should have the size indicated by inputSize
% create the datastore with the training and validation images
imds = imageDatastore(pathImagesTrain, ... 
   'IncludeSubfolders',true, ...
   'LabelSource','foldernames');
% split the dataset into training and validation datasets
[imdsTrain, imdsValidation] = splitEachLabel(imds,0.7,'randomized');
% obtain information about the training dataset
numTrainImages = numel(imdsTrain); % the number of trainig images
numClasses = numel(categories(imdsTrain.Labels)); %the number of classes
% augment the training and validation dataset
pixelRange = [-30 30];  
imageAugmenter = imageDataAugmenter( ...
   'RandXReflection',true, ...
   'RandXTranslation',pixelRange, ...
   'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
   'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
%% TESTING DATASETS
% indicate the path to the testing dataset
pathImagesTest='/MATLAB Drive/ML_AlexNet/RMN_BrainCancerHalf/';
%         full path of the folder with training images
%         the folder should include a separate subfolder for each class
%         the images should have the size indicated by inputSize
% create the datastore for the testing dataset     
imdsTest = imageDatastore(pathImagesTest, ... 
   'IncludeSubfolders',true, ...
   'LabelSource','foldernames');    
%% CNN for image classification
%+++++++++++++++++++++++++++++++++++

%% PARAMETERS
% the name of the file where the trained CNN is saved
nameFile_Results='rezCNN.mat';
% training parameters
MBS = 5;% mini batch size
NEP = 1; % number of epochs
% % %% TRAINING AND VALIDATION DATASET
% % load initialAlexNetData.mat;
%% DESIGN THE ARCHITECTURE
% load the pretrained model
net = alexnet;
% take the kayers for transfer of learning
layersTransfer = net.Layers(1:end-3);
% create the new architecture: the last fully connected layer is configured for the necessary number of classes
layersNew = [
   layersTransfer   
   fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
   softmaxLayer
   classificationLayer];
%% TRAIN THE CNN
% indicate the training parameters
options = trainingOptions('sgdm', ...
   'MiniBatchSize',MBS,...           
   'MaxEpochs',NEP, ...     
   'InitialLearnRate',1e-4, ... 
   'ValidationData',augimdsValidation, ...
   'ValidationFrequency',3, ...
   'ValidationPatience',Inf, ...
   'Verbose',false, ...
   'Plots','training-progress');
                
% train the model
netTransfer = trainNetwork(augimdsTrain,layersNew,options);
% save the trained model
feval(@save, nameFile_Results, 'netTransfer');
%% VERIFY THE RESULTS
% validation - responses and accuracy
[YPredValidation,scoresValidation] = classify(netTransfer, imdsValidation);
accuracyValidation = mean(YPredValidation == imdsValidation.Labels) 
% training - responses and accuracy
[YPredTrain,scoresTrain] = classify(netTransfer, imdsTrain); 
accuracyTrain = mean(YPredTrain == imdsTrain.Labels) 
% testing- responses and accuracy
[YPredTest,scoresTest] = classify(netTransfer, imdsTest); 
accuracyTest = mean(YPredTest == imdsTest.Labels) 
[Acc, Rec, Spec, Prec, Sens, F1sc, TP, TN, FP, FN, classif_err] = computePerfIndicatorsCNN(netTransfer, imdsTest.Labels, YPredTest);
layer=16;
name = netTransfer.Layers(layer).Name
channels = 1:36;
I = deepDreamImage(netTransfer,name,channels, ...
  'PyramidLevels',1);
figure(1)
I = imtile(I,'ThumbnailSize',[64 64]);
imshow(I)
title(['Layer ',name,' Features'],'Interpreter','none')
save nameFile_Results;
% save all_10_10_CNN.mat;
