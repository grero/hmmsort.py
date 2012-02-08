%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2008 Joshua Herbst
% 
% hmm_learn is free software: you can redistribute it and/or modify it under 
% the terms of the GNU General Public License as published by the Free Software 
% Foundation, either version 3 of the License, or (at your option) any later version.
% 
% hmm_learn is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
% without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
% PURPOSE.  See the GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License along with 
% this program. If not, see <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [data, spkform, cinv, p] = hmm_learn_tetrode(filename,saveas,varargin)

Args = struct('Channels',[],'Group','','WindowLength',1e6);
[Args,varargin] = getoptargs(varargin,Args);
%get the descriptor
idx = strfind(filename,'highpass');
%parts = strsplit(filename,'_');
%descriptor = ReadDescriptor([parts{1} '_descriptor.txt']);
descriptor = ReadDescriptor([filename(1:idx-1) 'descriptor.txt']);

if ~isempty(Args.Group)
	if ischar(Args.Group)
		g = str2num(Args.Group);
	else
		g = Args.Group;
	end

	channels = find(descriptor.group==g);
elseif ~isempty(Args.Channels)
		if ischar(channels)
			channels = str2num(channels);
		else
			channels = Args.Channels
		end
		Args.Group = descriptor.group(descriptor.channel==channels(1));
		g = Args.Group;
else
	disp('Please specify either a list of channels or a group to sort');
	return
end

disp(['Finding templates for group ' num2str(g) ' spanning channels ' num2str(channels) '...']);
%sometimes we don't have vartest
%if ~exist('vartest')
%	addpath(genpath('/Applications/MATLAB_R2010a.app/toolbox/stats'))
%end

%% specify parameters for preprocessing the data

%----------------------------------------------------
levels=4; % D of the final data to sort   %Yasamin 
%----------------------------------------------------

states=60; % how many states per ring
winstart=1; % where should we start learning
winlength=72000; % how long should the window be, that we learn on
% % scanrate=24000; % what is the scanrate of the data
%----------------------------------------------------
scanrate=30000; % what is the scanrate of the data  %Yasamin
%----------------------------------------------------

splitp=.5/scanrate; % firingrate limit, where we discard a spikeform
neurons=8; % how many neurons do we initialize
iterations=6; % how many iterations of
scanrate = 40000;
upsample_factor = 1;
tolerance = 20;
small_thresh = 1;


% load the file to sort
%just read the streamer file
%load(filename);
%check the header; if header size is 90, we are using UEI, if not, use streamer
header = ReadUEIFile('Filename',filename,'Header');
if header.headerSize == 73
	fid = fopen(filename,'r');
	[header.numChannels,header.samplingRate,scan_order,header.headerSize] = nptParseStreamerHeader(fid);
	fclose(fid);
end
scanrate = header.samplingRate;
M = memmapfile(filename,'format','int16','offset',header.headerSize);
if isempty(strfind(filename,'bandpass')) && isempty(strfind(filename,'highpass'))

	data = zeros(length(channels),length(M.Data)/header.numChannels);
	for ch =1:length(channels)
		data(ch,:) = M.Data(ch:header.numChannels:end);
	end
	% subtract the mean on all channels, each channel should be mean zero
	data=data-repmat(mean(data,2),1,size(data,2));


	%% preprocess the data... filter, upsample, add cwt channel...

	% bandpass filter between 100 Hz and 6 kHz
	data = ftfil(data,scanrate,100,6000);
else
	data = 	double(reshape(M.data,[header.numChannels,numel(M.data)/header.numChannels]));
end
if size(data,1) ~= length(channels)
	data = data(channels,:);
end
% upsample
if upsample_factor ~= 1
	data = spline(1:length(data),data,1:1/upsample_factor:length(data));
end
if ~isempty(Args.WindowLength)
	if ischar(Args.WindowLength)
		winlength = str2num(Args.WindowLength);
	else
		winlength = Args.WindowLength;
	end
else
	%use 30% of the data for learning, but limit to 1 million elements
	winlength = min(0.1*size(data,2),3.0e6);
end
%overwrite default parameters
splitp=.5/scanrate; % firingrate limit, where we discard a spikeform
% add a channel with continuous wavelet transform
% % data=[data;cwt(data,1,'db2')];
% the range on all channels is equalized - this change is cosmetical.
% % scale=max(data,[],2);
% % data=data./repmat(scale,1,size(data,2));


%% define spike templates, tune number of neurons and states
% each spike template consists of zeros, 1.5 periods of a sine, and zeros
% again... the amplitude is random. if the spike template amplitude is too
% large, this leads to poor convergence. (the more channels, the smaller
% amplitude)
spkform = cell(1,neurons);
for i=1:neurons
    spkform{i}=[zeros(levels,12) repmat(sin(linspace(0,3*pi,states-42)),levels,1) ...
        zeros(levels,30)].*repmat(abs(rand(levels,1).*[1;.5;0.9;0.4])... %Yasamin
        *5.*median(abs(data),2)/0.6745,1,states);
end
% calculate the inverse of the covariance matrix of the data
cinv = pinv(cov(data'));

% all the initial parameters are now fixed. data, spkform, cinv...

%% learn spike templates (with clustersplitting)
% extract the data that we learn on
x=data(:,winstart+1:winstart+winlength);

% we learn the spikewaveform with a 'n choose 1' algorithm. if there are
% numerical problems, due to very bad initial parameter estimates, we
% increase the covariance matrix and learn again. usually this solves the
% issue.
try
    [x,cinv,p,spkform] = learn1dbw(x,spkform,1,cinv);
catch
    [x,cinv,p,spkform] = learn1dbw(x,spkform,1,cinv/10);
end
for i=1:iterations-1
    try
        [x,cinv,p,spkform] = learn1dbw(x,spkform,1,cinv,p,splitp,1);
    catch
        try
            [x,cinv,p,spkform] = learn1dbw(x,spkform,1,cinv/3,p,splitp,1);
        catch
            disp(['stopped at iteration' num2str(i)])
            break
        end
    end
end
%%
% tmp = round(100*rand);
% save(['templates_' num2str(tmp)],'spkform','p','splitp','cinv','winlength')

%% remove spike forms which are too sparse

[spkform,p,ind] = remove_sparse(spkform,p,splitp);
disp('included because of p: ');ind

while true
    %% combine spike waveforms
    spks_old = length(spkform);
    [spkform,p] = combine_spikes(spkform,p,cinv,winlength,tolerance);
    spks = length(spkform);
    %% end loop if no templates where combined
    if spks == spks_old
        break
    end
    
    %% learn once more
    oldp = p;

    for count=1:5
        try
            [x,cinv,p,spkform] = learn1dbw(x,spkform,1,cinv,p);
            if count > 1 && sum(abs(oldp./p-1)<ones(size(p))*0.0001) == ...
                    length(spkform)
                break
            else
                oldp=p;
            end
        catch
            disp('learning did not work...')
        end
    end

    %% remove spike forms which are too sparse

    [spkform,p,ind] = remove_sparse(spkform,p,splitp);
    disp('included because of p:');ind

end

%% remove spike forms which are too small

[spkform,p,ind] = remove_stn(spkform,p,cinv,data,small_thresh);
disp('included because of sigma:');ind


%% save the final spike templates and inverse covariance matrix

W=[];
for i=1:length(spkform)
    W=[W spkform{i}(:,:)];
end
figure(100);clf;ndplot(W);

disp(['found ' num2str(length(spkform)) ' different neurons:']);
%convert from cell arary to ND array
spikeForms = zeros(length(spkform),size(spkform{1},1),size(spkform{1},2));
for i=1:length(spkform)
	spikeForms(i,:,:) = spkform{i};
end
if nargin >1
    save(saveas,'data','cinv','spkform','spikeForms','winlength');
end

end



function [data,cinv,p,spkform]=learn1dbw(data,spkform,iterations,cinv,p,...
    splitp,dosplit)

% implementation of a 'n choose 1' baum-welch algorithm for learning the
% spike templates.

% the input variables:
% data - data-array that the templates are learned on
% spkform - cell with the spike template estimates
% iterations - how many iterations should we run?
% cinv - inverse of the covariance matrix of the data
% p - spiking probability
% splitp - firingrate, where spike templates are discarded.
% dosplit - do we discard spikeforms and replace with new guess?

% data and spkform are mandatory

% default number of iterations = 10
if nargin < 3
    iterations=10;
end

% if cinv is not provided, it is re-calculated
if nargin < 4
    c=pinv(cov(data'));
else
    c=cinv;
end

% N is the number of spike templates (rings)
N=length(spkform);

% p, the spiking probability is specified for each ring
if nargin < 5
    p=.1e-8*ones(1,N);
else
    if length(p) < length(spkform)
        p=repmat(p,length(spkform),1);
    end
end

% below which firingrate 'splitp' do we discard spike templates?
if nargin<6
    splitp=3/40000;
end

% per default, we discard spike templates and reinitialize the templates
if nargin<7
    dosplit=0;
end

% dimension of p should be '1 x N'
if size(p,1)>size(p,2)
    p=p';
end

% p_reset is the spiking probability for each iteration of the algorithm
p_reset=p;

% read out the dimension of the data
dim=length(data(:,1));
% read out the number of states per ring
spklength=size(spkform{1},2);
% read out the length of the data
winlength=size(data,2);
% write all the spike-templates into the same array 'W'
W=zeros(dim,N*(spklength-1)+1);
for i=1:N
    W(:,2+(spklength-1)*(i-1):1+i*(spklength-1))=spkform{i}(:,2:end);
end

q=[N*(spklength-1)+1 1:N*(spklength-1)];
tiny=exp(-700);

% run the Baum-Welch-algorithm
for bw=1:iterations
    % reset the spiking probability
    p=p_reset;
    % initialize the array for the forward and backward variable
    g=zeros(N*(spklength-1)+1,winlength);
    % initialize the array for the fitting probabilities
    fit=zeros(N*(spklength-1)+1,winlength);
    % helper array for the backward pass
    b=zeros(N*(spklength-1)+1,1);
    g(1,1)=1;b(1)=1;

    % calculate the output probabilities
    for t=1:winlength
        fit(:,t)=exp(-.5*sum((W-data(:,t)*ones(1,N*(spklength-1)+1))...
            .*(c*(W-data(:,t)*ones(1,N*(spklength-1)+1))),1))';
    end
    % forward pass
    for t=2:winlength
        % forward transitions
        g(:,t)=g(q,t-1);
        g(1,t)=sum(g(2:(spklength-1):2+(N-1)*(spklength-1),t))...
            +g(1,t)-g(1,t-1)*sum(p);
        g(2:(spklength-1):2+(N-1)*(spklength-1),t)=g(1,t-1)*(p');
        %output-probability
        g(:,t)=g(:,t).*fit(:,t);
        %normalize
        g(:,t)=g(:,t)/(sum(g(:,t))+tiny);
    end

    % backward pass
    for t=winlength-1:-1:1
        %output-probability
        b=b.*fit(:,t+1);
        % backward transition
        b(q)=b;
        b(1)=(1-sum(p))*b(end)+p*b(1:(spklength-1):1+(N-1)*(spklength-1));
        b(1+(spklength-1):(spklength-1):1+(N-1)*(spklength-1))=b(end);
        % normalize
        b=b/(sum(b)+tiny);
        % calculate product (forward * backward)
        g(:,t)=g(:,t).*b;

        % g(:,t)=g(:,t)/(sum(g(:,t))+tiny);
    end

    % normalize to get the posterior probabilities
    g=g./(ones(N*(spklength-1)+1,1)*sum(g,1)+tiny);
    % calculate the new spike waveforms
    W=data*g'./(ones(dim,1)*sum(g,2)');
    W(:,1)=0;
    % calculate the new spiking probabilities
    p=sum(g(2:(spklength-1):2+(N-1)*(spklength-1),:),2)'/winlength;
    % calculate the new covariance matrix
    cinv = pinv(cov(data'-g'*W'));

    % extract spkform from 'W'
    maxamp = zeros(1,length(spkform));
    for j=1:length(spkform)
        spkform{j}=[W(:,1) W(:,(j-1)*(spklength-1)+2:j*(spklength-1)+1)];
        maxamp(j)=max(sum(spkform{j}.*(cinv*spkform{j}),1));
    end

    disp('spikes found per template: ')
    p*winlength
    figure(100);clf;ndplot(W,48000);

    % remove templates and reinitialize
    if dosplit
        % for each template, check if firingrate too small
        for i=1:length(spkform)
            if p(i) < splitp
                % discard template 'i', reinitialize with template 'j'
                try
                    j=find(p>=median(p) & p>splitp*4 & maxamp>10);
                    j=j(randint(1,1,length(j))+1);
                    W(:,(i-1)*(spklength-1)+2:i*(spklength-1)+1)=...
                        W(:,(j-1)*(spklength-1)+2:j*(spklength-1)+1)*.98;
                    p(i)=p(j)/2;
                    p(j)=p(j)/2;
                    disp(['waveformupdate: ' num2str(i) ' <- ' num2str(j)])
                    break
                catch
                    disp('clustersplitting failed')
                end
            end

        end
    end
end

% create spkform for output
for j=1:length(spkform)
    spkform{j}=[W(:,1) W(:,(j-1)*(spklength-1)+2:j*(spklength-1)+1)];
end
end

function [forms,pp,ind] = remove_sparse(spkform,p,splitp)

j=0;
ind=[];
% pp is p for the (possibly reduced) templates
pp=[];
% check for each template, if it fires to sparsely
for i=1:length(spkform)
    j=j+1;
    if p(i)<splitp
        j=j-1;
    else
        forms{j}=spkform{i};
        pp(j)=p(i);
        ind=[ind i];
    end
end
end

function [spkform,pp,ind]=remove_stn(forms,p,cinv,data,small_thresh)

% remove spikes that do not exceed twice the energy of an average noise
% patch of identical length are removed from the set of templates.



%% calculate the average energy of a noise patch
% if data is given, average the energy of 500 patches and take the median 
% energy as limit.
if nargin<4
    limit=size(forms{1},2)*3;
else
    tmp=size(forms{1},2);
    test=zeros(1,500);
    for i=1:500
        test(i)=sum(sum(data(:,(i-1)*tmp+1:i*tmp)...
            .*(cinv*data(:,(i-1)*tmp+1:i*tmp))));
    end
    limit=median(test);
end

%% check if the i-th template exceeds twice this limit

j=0;
ind=[];
woe=zeros(1,length(forms));

for i=1:length(forms)
    woe(i)=sum(sum(forms{i}.*(cinv*forms{i})));
    j=j+1;
    if woe(i)<limit*small_thresh
        j=j-1;
    else
        spkform{j}=forms{i};
        pp(j)=p(i);
        ind=[ind i];
    end
end
end

function [spkform,p]=combine_spikes(spkform_old,pp,cinv,winlen,tolerance)

% condense spike templates if they represent the same neuron
if nargin < 5
    tolerance = 4
end
winlen = winlen/tolerance;
alpha = 0.001;
maxp= 12;

spks = length(spkform_old);
dim = size(spkform_old{1},1);
spklen = size(spkform_old{1},2);

%% rotate spikeform order, so correlation can be calculated

% some templates are condensed, small ones not... these are excluded from
% the condensation step and added to the set of spike templates at the end
j = 0;
k = spks;
p = zeros(1,spks);
forms = cell(1,spks);
ind = zeros(1,spks);
for i=1:spks
    % put small templates at the end off the cell, big ones at the
    % beginning
    if trace((cinv*spkform_old{i})*spkform_old{i}') < 3*spklen
        forms{k}=spkform_old{i};
        p(k)=pp(i);
        ind(k)=i;
        k=k-1;
    else
        j=j+1;
        forms{j}=spkform_old{i};
        p(j)=pp(i);
        ind(j)=i;
    end
end

% spkform_old is now divided into 2 groups
spkform_old = forms; clear forms;
spkform = spkform_old;
pp = p;
% index of excluded templates
excl = [j+1 spks];
% number of included templates
spks = j;
numspikes = j;

%% combine large spikes

for rotate=1:numspikes

    % upsample spikeforms by factor 10 and add zeros at beginning and end
    spklennew = spklen*10+11;
    splineform = zeros(dim,spklennew,spks);
    splineform_test = zeros(dim,spklennew,spks);
    for i=1:spks
        for j=1:dim
            splineform(j,:,i) = [zeros(1,10) ...
                spline(1:spklen,spkform{i}(j,:),1:.1:spklen) zeros(1,10)];
            splineform_test(j,:,i) = [zeros(1,10) ...
                ones(1,spklen*10-9) zeros(1,10)];
        end
    end

    % calculate max similarity with respect to the first template and
    % shift by calculated index
    shift=ones(1,spks);
    % loop over templates
    for i=2:spks
        difference_old = Inf;
        % loop over shift index
        for j=1:spklennew
            shifted_template = [splineform(:,j:end,i) splineform(:,1:j-1,i)];
            % calculate similarity for each shift index
            difference = trace((splineform(:,:,1)-shifted_template)*...
                (cinv*(splineform(:,:,1)-shifted_template))');
            % if the similarity is smaller than before, keep new value and
            % save index
            if difference < difference_old
                difference_old = difference;
                shift(i) = j;
            end
        end
        % shift template 'i' by 'shift' samples
        splineform(:,:,i)=[splineform(:,shift(i):end,i) ...
            splineform(:,1:shift(i)-1,i)];
        splineform_test(:,:,i)=[splineform_test(:,shift(i):end,i) ...
            splineform_test(:,1:shift(i)-1,i)];
    end

    % test which templates to combine
    [index, docombine, value] = combine_test(spks,splineform,splineform_test,cinv,winlen,p,maxp,alpha);
    % spikes to be combined: index(1) and index(2)

    % combine spikes, and check again
    while docombine
        % when we combine, we have one template less
        rotate=rotate+1; %#ok<FXSET>

        % combine templates c(1) and b(c(1))
        splineform(:,:,index(1)) = (p(index(1))*splineform(:,:,index(1))+...
            p(index(2))*splineform(:,:,index(2)))/sum(p(index));
        p(index(1)) = sum(p(index));
        splineform_new = zeros(size(splineform)-[0 0 1]);
        splineform_test_new = zeros(size(splineform_test)-[0 0 1]);
        p_new = zeros(1,spks-1);
        shift_new = zeros(1,spks-1);
        k = 0;
        for count = 1:spks
            if count ~= index(2)
                k = k+1;
                splineform_new(:,:,k) = splineform(:,:,count);
                splineform_test_new(:,:,k) = splineform_test(:,:,count);
                p_new(k) = p(count);
                shift_new(k) = shift(count);
            end
        end

        shift = shift_new;
        splineform = splineform_new;
        p = p_new;
        splineform_test = splineform_test_new;
        spks = spks-1;
        
        [index, docombine, value] = combine_test(spks,splineform,splineform_test,cinv,winlen,p,maxp,alpha);
    end
    
    disp(['could not combine anymore: value ' num2str(value)])

    % back-shift the templates
    for i=2:spks
        if shift(i)~=1
            splineform(:,:,i)=[splineform(:,end-shift(i)+2:end,i)...
                splineform(:,1:end-shift(i)+1,i)];
        end
    end

    % down-sample the templates
    clear spkform
    spkform = cell(1,spks);
    for i=1:spks
        spkform{i} = [zeros(size(splineform,1),1) ...
            spline(1:size(splineform,2),splineform(:,:,i),21:10:size(splineform,2)-10)];
    end

    % rotate the order of the templates
    ptemp = zeros(1,spks);
    forms = cell(1,spks);
    for i=1:spks
        forms{i} = spkform{max(mod(i+1,spks+1),1)};
        ptemp(i) = p(max(mod(i+1,spks+1),1));
    end
    spkform = forms;
    p = ptemp;
end

% add excluded templates
i = spks;
for j = excl(1):excl(2)
    i = i+1;
    spkform{i} = spkform_old{j};
    p(i) = pp(j);
end
end

function [index, docombine,value] = combine_test(spks,splineform,...
    splineform_test,cinv,winlen,p,maxp,alpha)

if spks == 1
    index = 1;
    docombine = 0;
    value = 0;
    return
end

% determine common area to test on and restrict splineform to overlapping
% area
splineform = splineform(:,find(sum(splineform_test(1,:,:),3)==spks),:);

h = zeros(spks);
pvalt = zeros(spks);
teststat = zeros(spks);

% calculate the similarity for each pair of templates
for i = 1:spks-1
    for j = i+1:spks
        % downsample the spike templates and calculate the difference
        diffform = spline(1:size(splineform,2),...
            splineform(:,:,i)-splineform(:,:,j),...
            1:10:size(splineform,2));
        
        teststat(i,j) = trace(diffform'*(cinv*diffform))...
            /(1/(min(winlen*p(i),maxp))+1/min(winlen*p(j),maxp));

        if teststat(i,j) < numel(diffform)
            % in this case, the test is not necessary
            pvalt(i,j) = 1;
        else
            [h(i,j), pvalt(i,j)] = vartest([sqrt(teststat(i,j)/2)...
                -sqrt(teststat(i,j)/2) zeros(1,numel(diffform)-2)],...
                1);
        end

        % copy symmetric value
        pvalt(j,i)=pvalt(i,j);
    end
end

teststat = teststat+teststat'+eye(size(teststat,1))*max(max(teststat,[],1));

[a,b] = min(teststat,[],2);
[a,c] = min(a);

% determine the indices of maximum similarity
index = [c(1) b(c(1))];
maximum = pvalt(index(1),index(2));
value = teststat(index(1),index(2))/numel(diffform);
% if confidence level is reached...
if maximum > alpha
    disp(['combine spikes, p-value: ' num2str(maximum)])
    disp(['combine spikes, test-value: ' num2str(teststat(index(1),index(2)))])
    disp(['overlapping area: ' num2str(floor(size(splineform,2)/10))])
    docombine = 1;
else
    docombine = 0;    
end
end

function [] = ndplot (data, scanrate)

if nargin < 2
    scanrate = 24000;
end


data=data-repmat(mean(data,2),1,size(data,2));

offset = cumsum([0 abs(max(data'))]+[abs(min(data')) 0]);


hold on;
for i = 1 : size(data,1)
    plot([1:length(data(i,:))] / scanrate, data(i,:) + offset(i));
end

hold off;
end
