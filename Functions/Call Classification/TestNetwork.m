function TestNetwork(hObject, eventdata, handles)
[FileName,PathName] = uigetfile(fullfile(handles.data.squeakfolder,'Clustering Models','*.mat'),'Select Network');
load([PathName FileName],'ClassifyNet','wind','noverlap','nfft','imageSize','padFreq');

if exist('ClassifyNet', 'var') ~= 1 
    errordlg('Network not be found. Is this file a trained CNN?')
    return
end
cd(handles.data.squeakfolder);
        [trainingdata, trainingpath] = uigetfile([handles.data.settings.detectionfolder '/*.mat'],'Select Detection File(s) for Training ','MultiSelect', 'on');
        if isnumeric(trainingdata)  % If user cancels
            return
        end
        trainingdata = cellstr(trainingdata);

        % Spectrogram Settings
        wind = .0032;
        noverlap = .0028;
        nfft = .0032;


        h = waitbar(0,'Initializing');
        X = [];
        Class = [];
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

        predicted=classify(ClassifyNet,X);
        disp(predicted)
        count=0;
        for i=1:length(predicted)
            if (predicted(i)==Class(i))
                count= count+1;
            end
        end
        accuracy=count/length(predicted);
        accuracy=accuracy*100;
        disp(accuracy);
        

end
