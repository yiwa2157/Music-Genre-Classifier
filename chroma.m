function [ NPCP ] = chroma( wav, Fs, fftSize,window )
%This function compute the NPCP(normalized pitch class profile)of the music
%waveform. It also generates a chromagram of it.

%------------------cut wav sequence into frames--------------------------%
N = fftSize;                            %fftsize=frame size
[length_wav ,~]=size(wav);           %get the track length
num_frame = ceil(length_wav/(N/2));     %find the number of frames
xn=zeros(N,num_frame);                  %create space for the frame structure
wav_cat=vertcat(wav,zeros(N*...         %add zeros to the tail of the frame
(num_frame/2)-length_wav,1));
for n=1:num_frame-1                     %loop number of frames
    for i=1:N                           %loop in one frame
        xn(i,n) = wav_cat(i+(n-1)*(N/2)); %put the track into the frame structure   
    end
end
xn(:,num_frame)=[];                     %discard the last unused frame
num_frame=num_frame-1;
%--------------transfer the frames into frequency domain-----------------%
Y=zeros(N,num_frame);                   %creaate space for fft result
for n=1:num_frame                       
    Y(:,n)=fft(xn(:,n).*window);        %compute fft
end

%-------------------discard the negative frequencies---------------------%
K = N/2+1;                              %size we need for positive frequency
Xn=abs(Y(1:K,:));                       %get the positive frequency part

%---------------------------I. peak detection---------------------------%

%1. find the maxium number of peaks to decide 
%   the size of row of fk(peak frequency) matrix and sm(semitone) matrix
npeaks=zeros(1,num_frame);              %create space for number of peaks
for n=1:num_frame                       %loop frames
    [peak]= findpeaks(Xn(:,n));         %find peaks
    [npeaks(1,n) dummy]=size(peak);     %set the size of the matrix by max npeaks
end

%---------II.Assignment of the peak frequencies to semitones--------------%

%initialization
fk=zeros(max(npeaks),num_frame);    %local maxima frequencies
fk_mag=zeros(max(npeaks),num_frame);%peak frequency magnitude
%sm=zeros(max(npeaks),num_frame);    %semi-tones corresponding to fk 
%c=zeros(max(npeaks),num_frame);    %map semitones to notes
nyquist_freq=Fs/2;                  %nyquist frequency
f0=27.5;                            %lowest frequency
hopSize = fftSize/2;                %frame size

%a.find peak frequency, compute semitone and notes
for n=1:num_frame                   %loop frames
    [fk_mag(1:npeaks(n),n) fk(...   %find peak frequency
    1:npeaks(n),n)]=findpeaks(Xn(:,n));
end

%------------------------noise canceling----------------------------------%
%In each frame, all the peaks whose magnitude is smaller than-60dB of the
%maximum peak magnitude are considered as noise. They will be wiped out.
max_fk_mag = max(fk_mag);           %find max peak in each frame
for n=1:num_frame                   %loop frames
    if fk_mag(1:npeaks(n),n) <= max_fk_mag(n)/1000
        fk_mag(1:npeaks(n),n) = 0;  %delete the small ripples
    end
end
%-------------------------------------------------------------------------%

sm = round(12*log2((fk./hopSize.*nyquist_freq)./f0));%compute semitone 
c = mod(sm,12);                     %compute notes

%-----------------------III.Pitch Class Profile---------------------------%

%a. raised cosine weighting function
r=12*log2((fk-1)/hopSize*nyquist_freq/f0)-sm;
w=cos(pi*abs(r)/2).^2;
if (r<=-1 | r>=1),
    w = 0;
end

%b. compute PCP(Pitch Class Profile)
% c matrix contains the information of notes.If certain frequency 
% corresponds to a note i, it means the power of that frequency should sum
% into row i of PCP matrix.
PCP=zeros(12,num_frame);            %creeate space for PCP    
for i=1:num_frame 
    for j=1:npeaks(i)
        for k=0:11
            if c(j,i)==k            %compute PCP  
                PCP(k+1,i)=PCP(k+1,i)+w(j,i)*(fk_mag(j,i)^2);
            end
        end
    end
end

%-----------------------IV.Normalize PCP ---------------------------------%
%divide all PCPs by the maximum PCP in the note space
NPCP = PCP./max(max(PCP));          %compute NPCP
%-------------------------------------------------------------------------%


% %-----------------------V.draw some plots--------------------------------%
% figure;
% imagesc(10*log10(NPCP));
% colormap(jet)
% caxis([-40 0])
% colorbar
% xlabel('seconds');
% ylabel('notes');
% yticks([1 2 3 4 5 6 7 8 9 10 11 12]);
% yticklabels({'A','A#','B','B#','C','C#','D','D#','E','E#','F','F#'});
% XLABEL=[0:50:(num_frame+50-mod(num_frame,50))];
% 
% %-------------------------------------------------------------------------%
% %Rev2. new: convert unit from frames to seconds on the time axis 
% xticks(XLABEL);
% xticklabels(start+XLABEL.*(hopSize/Fs));
% %-------------------------------------------------------------------------%
% 
% title(['The chromagram of ',name]);
end

