%------------I.Import 150 songs as training samples-----------------------%
size_files=150;%amount of audio files
fs = zeros(1,size_files);%music track sampling frequency
length = 30;%determine the length uint=seconds
wav = zeros(length*22050,size_files);  %extracted in each song
start = 1;                          %start point (uint = seconds)

for j=1:6                           %keep 6 genres in order 
    if j == 1
        file_info=dir('contest/data/classical');
    elseif j == 2
        file_info=dir('contest/data/electronic');
    elseif j == 3
        file_info=dir('contest/data/jazz');
    elseif j == 4
        file_info=dir('contest/data/punk');
    elseif j == 5
        file_info=dir('contest/data/rock');
    elseif j == 6
        file_info=dir('contest/data/world');
    end
    for i=1:25
        [song_temp,fs(1,i+(j-1)*25)]=audioread(file_info(i+2).name);
        [size_temp, dummy]=size(song_temp);
        if size_temp < length*22050
            wav(1:size_temp,i+(j-1)*25) = song_temp(1:size_temp,1);
        else
            wav(:,i+(j-1)*25) = song_temp(start*22050+1:(length+start)*22050,1);
        end   
    end
end
%-------------------------------------------------------------------------%

confusion = zeros(6,6,10);
for iter=1:10
%-----------------II.Define test set and training set---------------------%
[size_wav,~] = size(wav);
num_test = 5;
test_set=zeros(size_wav,num_test*6);
train_set=zeros(size_wav,150-num_test*6);
set=zeros(25,1);
for i=1:6
    set=randperm(25);               %genrate random order of 25 numbers
    for j=1:5
        test_set(:,j+(i-1)*5)=...
        wav(:,set(j)+(i-1)*25);% take the first 5 numbers as the test set index     
    end
    for j=6:25
        train_set(:,j-5+(i-1)*20)=...
        wav(:,set(j)+(i-1)*25);% take the rest 20 numbers as the test set index 
    end
end
[~,size_train] = size(train_set);
[~,size_test] = size(test_set);

%-------------III. Compute mfcc&merge into 12 channels--------------------%
window=hann(512);%hann window
fft_size=512;    %fftsize
t = zeros(1,36); %merge 40 mfcc into 12 
t(1) =1;t(7:8)=5;t(15:18)= 9;
t(2) = 2; t( 9:10) = 6; t(19:23) = 10;
t(3:4) = 3; t(11:12) = 7; t(24:29) = 11;
t(5:6) = 4; t(13:14) = 8; t(30:36) = 12;
train_mfcc = mfcc1(train_set(:,1),fs(1,1),fft_size,window);
train_mel2 = zeros(12,size(train_mfcc,2)-1,size_train);
test_mfcc = mfcc1(test_set(:,1),fs(1,1),fft_size,window);
test_mel2 = zeros(12,size(test_mfcc,2)-1,size_test);

size_mfcc=0;
for i=1:size_train
     train_mfcc =...                             %call mfcc 
     mfcc1(train_set(:,i),fs(1,i),fft_size,window);
     [~, size_mfcc] = size(train_mfcc(1,:));     %find the number of frames
     train_mfcc(:,size_mfcc)=[];                 %discard the last unused frame
     size_mfcc=size_mfcc-1;
     for j=1:12
        train_mel2(j,:,i) =...                   %merge mfcc coefficients 
        sum(train_mfcc(t==j,:),1);
     end
end
for i=1:size_train
    for j =1:12
        for k = 1:size_mfcc
            if train_mel2(j,k,i)==-Inf      %discard -Inf data  
                train_mel2(j,k,i)=0;
            end
        end
    end
end
train_mfcc = train_mel2;

size_mfcc=0;
for i=1:size_test
     test_mfcc = mfcc1(test_set(:,i),fs(1,i),fft_size,window);
     [~, size_mfcc] = size(test_mfcc(1,:));     %find the number of frames
     test_mfcc(:,size_mfcc)=[];                     %discard the last unused frame
     size_mfcc=size_mfcc-1;
     for j=1:12
        test_mel2(j,:,i) = sum(test_mfcc(t==j,:),1);
     end
end
for i=1:size_test
    for j =1:12
        for k = 1:size_mfcc
            if test_mel2(j,k,i)==-Inf
                test_mel2(j,k,i)=0;
            end
        end
    end
end
test_mfcc = test_mel2;
%-------------------------------------------------------------------------%

%--------------------------III.2.Compute chroma---------------------------%
fftsize_chroma = 1024;
window_chroma = kaiser(1024);
train_chm = chroma(wav(:,1),fs(1,1),fftsize_chroma,window_chroma);
train_chm = zeros(12,size(train_chm,2),size_train);

for i=1:size_train
     train_chm(:,:,i) = chroma(train_set(:,i),fs(1,i),fftsize_chroma,window_chroma);
end

test_chm = chroma(wav(:,1),fs(1,1),fftsize_chroma,window_chroma);
test_chm = zeros(12,size(test_chm,2),size_test);

for i=1:size_test
     test_chm(:,:,i) = chroma(test_set(:,i),fs(1,i),fftsize_chroma,window_chroma);
end
%-------------------------------------------------------------------------%

%-----------------------IV.compute mu and covariance----------------------%
%for mfcc
train_mu = mean(train_mfcc,2);
train_Cov = zeros(12,12,size_train);
for i=1:size_train
    train_Cov(:,:,i) = cov(train_mfcc(:,:,i)');
end

test_mu = mean(test_mfcc,2);
test_Cov = zeros(12,12,size_test);
for i=1:size_test
    test_Cov(:,:,i) = cov(test_mfcc(:,:,i)');
end
%for mfcc end

% %%for chroma
% train_mu = mean(train_chm,2);
% train_Cov = zeros(12,12,size_train);
% for i=1:size_train
%     train_Cov(:,:,i) = cov(train_chm(:,:,i)');
% end
% 
% test_mu = mean(test_chm,2);
% test_Cov = zeros(12,12,size_test);
% for i=1:size_test
%     test_Cov(:,:,i) = cov(test_chm(:,:,i)');
% end
% %%for chroma end

train_iCov = zeros(12,12,size_train);
for i=1:size_train
    train_iCov(:,:,i) = pinv(train_Cov(:,:,i));
end

test_iCov = zeros(12,12,size_test);
for i=1:size_test
    test_iCov(:,:,i) = pinv(test_Cov(:,:,i));
end
%-------------------------------------------------------------------------%

%------------------------IV. compute distance-----------------------------%
gam=0.9;
KL = zeros(size_test,size_train);
d = zeros(size_test,size_train);

for i=1:size_test
    for j=1:size_train
        KL(i,j) = 0.5*(trace(test_Cov(:,:,i)*train_iCov(:,:,j)) + ...
        trace(train_Cov(:,:,j)*test_iCov(:,:,i)) + ...
        trace((test_iCov(:,:,i)+train_iCov(:,:,j))*(test_mu(:,:,i)-train_mu(:,:,j))*...
        (test_mu(:,:,i)-train_mu(:,:,j))'));
        
%         KL(i,j) = 0.5*(trace(test_Cov(:,:,i)*train_iCov(:,:,j))+...
%         (train_mu(:,:,j)-test_mu(:,:,i))'*train_iCov(:,:,j)*...
%         (train_mu(:,:,j)-test_mu(:,:,i))+log(det(train_Cov(:,:,j)/...
%         det(test_Cov(:,:,i)))));
    
        %d(i,j) = 1 - exp(-gam/(KL(i,j)+eps));
        d(i,j) = exp(-gam*(KL(i,j)+eps));
    end
end

%----------------------------V. compute KNN-------------------------------%
vote = zeros(30,5);
for j=1:30
[vote_val,vote_idx]=sort(d(j,:),'descend');
    for i=1:5                       %assign genre to vote indexes
        if vote_idx(i)>=1&&vote_idx(i)<=20
            vote(j,i) = 1;
        elseif vote_idx(i)>=21&&vote_idx(i)<=40
            vote(j,i) = 2;
        elseif vote_idx(i)>=41&&vote_idx(i)<=60
            vote(j,i) = 3;
        elseif vote_idx(i)>=61&&vote_idx(i)<=80
            vote(j,i) = 4;
        elseif vote_idx(i)>=81&&vote_idx(i)<=100
            vote(j,i) = 5;
        elseif vote_idx(i)>=101&&vote_idx(i)<=120
            vote(j,i) = 6;
        end
    end
end

vote_result=zeros(30,1);
for i=1:30                          %find the max 5 vote result
    [~,vote_result_idx]=max(histc(vote(i,:),[1:6]));
    vote_result(i)=vote_result_idx;
end


for i=1:6                           % find the amount of major vote and 
                                    %put the result in confusion matrix
    confusion(i,:,iter)=histc(vote_result(1+(i-1)*5:5+(i-1)*5),[1:6])';
end
end

confusion_avg=zeros(6,6);
confusion_std=zeros(6,6);
for i=1:6
    for j=1:6                       %compute average and std of confusion matrix
        confusion_avg(i,j)=mean(confusion(i,j,:));
        confusion_std(i,j)=std(confusion(i,j,:));
    end
end

                    
Classical=confusion_avg(:,1);       %put the result in the table    
Electronic=confusion_avg(:,2);
Jazz=confusion_avg(:,3);
Punk=confusion_avg(:,4);
Rock=confusion_avg(:,5);
World=confusion_avg(:,6);
True_genre = {'Classical','Electronic','Jazz','Punk','Rock','World'};
T_avg = table(Classical,Electronic,Jazz,Punk,Rock,World,'RowNames',True_genre)

Classical=confusion_std(:,1);
Electronic=confusion_std(:,2);
Jazz=confusion_std(:,3);
Punk=confusion_std(:,4);
Rock=confusion_std(:,5);
World=confusion_std(:,6);
True_genre = {'Classical','Electronic','Jazz','Punk','Rock','World'};
T_std = table(Classical,Electronic,Jazz,Punk,Rock,World,'RowNames',True_genre)