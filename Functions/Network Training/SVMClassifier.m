function SVMClassifier(hObject, eventdata, handles)
cd(handles.data.squeakfolder);
answer = questdlg('Train Which Classifiers', ...
	'Classifiers', ...
	'SVM','CNN','Cancel','Cancel');
switch answer
    case 'SVM'
        [trainingdata, trainingpath] = uigetfile([handles.data.settings.detectionfolder '/*.mat'],'Select Detection File(s) for Training ','MultiSelect', 'on');
        if isnumeric(trainingdata)  % If user cancels
            return
        end
        trainingdata = cellstr(trainingdata);

        % Spectrogram Settings
        wind = .0032;
        noverlap = .0028;
        nfft = .0032;

        settings = inputdlg({'Frequency to pad boxes aboxe and below each box (kHz):'},'Frequency to pad boxes by',[1 60],{'10'});
        padFreq = str2num(settings{1});
        imageSize = [200 200];

        h = waitbar(0,'Initializing');
        X = [];
        Class = [];
        cellSize = [4 4];
        numImages = 0;
        for j = 1:length(trainingdata)  % For Each File
            Calls = loadCallfile(fullfile(trainingpath, trainingdata{j}));
            numImages=numImages+height(Calls);
        end
        hogFeatureSize = 86436;
        trainingFeatures = zeros(numImages,hogFeatureSize,'single');
        for j = 1:length(trainingdata)  % For Each File
            Calls = loadCallfile(fullfile(trainingpath, trainingdata{j}));

            Xtemp = [];
            Classtemp = [];
            Calls=Calls(Calls.Accept==1 & Calls.Type ~= 'Noise', :);

            for i = 1:height(Calls)     % For Each Call
                waitbar(i/height(Calls),h,['Loading File ' num2str(j) ' of '  num2str(length(trainingdata))]);
                audio = Calls.Audio{i};
                if ~isfloat(audio)
                    audio = double(audio) / (double(intmax(class(audio)))+1);
                elseif ~isa(audio,'double')
                    audio = double(audio);
                end

                [s, fr, ti] = spectrogram((audio),round(Calls.Rate(i) * wind),round(Calls.Rate(i) * noverlap),round(Calls.Rate(i) * nfft),Calls.Rate(i),'yaxis');

                x1 = axes2pix(length(ti),ti,Calls.RelBox(i, 1));
                x2 = axes2pix(length(ti),ti,Calls.RelBox(i, 3)) + x1;
                %y1 = axes2pix(length(fr),fr./1000,lowFreq);
                %y2 = axes2pix(length(fr),fr./1000,highFreq);
                y1 = axes2pix(length(fr),fr./1000,Calls.RelBox(i, 2)-padFreq);
                y2 = axes2pix(length(fr),fr./1000,Calls.RelBox(i, 4)+padFreq*2) + y1;

                y1 = max(y1,1); % Make sure that the box isn't too big
                y2 = min(y2,size(s,1));
                I=abs(s(round(y1:y2),round(x1:x2))); % Get the pixels in the box

                % Use median scaling
                med = median(abs(s(:)));
                im = mat2gray(flipud(I),[med*0.65, med*20]);
                Xtemp(:,:,:,i) = single(imresize(im,imageSize));
                img=imresize(im,imageSize);
                img = imbinarize(img);
                trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);  
                
%                 [hog_2x2, vis2x2] = extractHOGFeatures(img,'CellSize',[2 2]);
%                 [hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4]);
%                 [hog_8x8, vis8x8] = extractHOGFeatures(img,'CellSize',[8 8]);
%                 figure; 
%                 subplot(2,3,1:3); imshow(img);
% 
%                 % Visualize the HOG features
%                 subplot(2,3,4);  
%                 plot(vis2x2); 
%                 title({'CellSize = [2 2]'; ['Length = ' num2str(length(hog_2x2))]});
% 
%                 subplot(2,3,5);
%                 plot(vis4x4); 
%                 title({'CellSize = [4 4]'; ['Length = ' num2str(length(hog_4x4))]});
% 
%                 subplot(2,3,6);
%                 plot(vis8x8); 
%                 title({'CellSize = [8 8]'; ['Length = ' num2str(length(hog_8x8))]});
                Classtemp = [Classtemp; categorical(Calls.Type(i))];
            end
            X = cat(4,X,Xtemp);
            Class = [Class; Classtemp];
        end
        close(h)

        %% Make all categories 'Title Case'
        cats = categories(Class);
        for i = 1:length(cats)
            newstr = lower(cats{i}); % Make everything lowercase
            idx = regexp([' ' newstr],'[\ \-\_]'); % Find the start of each word
            newstr(idx) = upper(newstr(idx)); % Make the start of each word uppercase
            Class = mergecats(Class, cats{i}, newstr);
        end
        Class = removecats(Class);

        %% Select the categories to train the neural network with
        call_categories = categories(Class);
        idx = listdlg('ListString',call_categories,'Name','Select categories for training','ListSize',[300,300]);
        calls_to_train_with = ismember(Class,call_categories(idx));
        X = X(:,:,:,calls_to_train_with);
        Class = Class(calls_to_train_with);
        Class = removecats(Class);
        
        %% Train
        % Divide the data into training and validation data.
        % 90% goes to training, 10% to validation.
        [trainInd,valInd] = dividerand(size(X,4),.80,.20);
        TrainX = trainingFeatures(trainInd,:);
        TrainY = Class(trainInd);
        ValX = trainingFeatures(valInd,:);
        ValY = Class(valInd);

        % Augment the data by scaling and translating
%       aug = imageDataAugmenter('RandXScale',[.90 1.10],'RandYScale',[.90 1.10],'RandXTranslation',[-20 20],'RandYTranslation',[-20 20],'RandXShear',[-9 9]);
%       auimds = augmentedImageDatastore(imageSize,TrainX,TrainY,'DataAugmentation',aug);
        classifier = fitcecoc(TrainX, TrainY);
        disp('DONE TRAINING')
        predictedLabels = predict(classifier, ValX);
        figure('color','w')
        [C,order] = confusionmat(predictedLabels,ValY);
        h = heatmap(order,order,C);
        h.Title = 'Confusion Matrix';
        h.XLabel = 'Predicted class';
        h.YLabel = 'True Class';
        h.ColorbarVisible = 'off';
        colormap(inferno);
        disp('DONE TESTING')
        disp("TESTING ACCURACY");
        count=0;
        for i=1:length(predictedLabels)
            if predictedLabels(i)== ValY(1)
                count=count+1;
            end
        end
        accuracy=count/length(predictedLabels);
        accuracy=accuracy*100;
        disp(accuracy)
        
    case 'Cake'
        disp('HELLO')
end


