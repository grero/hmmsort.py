%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2008 Joshua Herbst
% 
% hmm_decode is free software: you can redistribute it and/or modify it under 
% the terms of the GNU General Public License as published by the Free Software 
% Foundation, either version 3 of the License, or (at your option) any later version.
% 
% hmm_decode is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
% without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
% PURPOSE.  See the GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License along with 
% this program. If not, see <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [mlseq,ll] = hmm_decode(varargin)

Args = struct('SourceFile',[],'Channels',[],'save',0,'Group','','hdf5',0,'DescriptorFile',[],'hdf5Path',[],'spikeForms',[],'data',[],'reorder',[],'maxSize',[],'maxCells',30,'DataFile','','samplingRate',[],'fileName',[],'patchLength',[],'prob',[],'cinv',[],'outlierThreshold',4,'parseOutput',0, 'SaveFile', '');
Args.flags = {'save','hdf5','parseOutput'};
[Args,varargin] = getOptArgs(varargin,Args);
% specify file to sort, should consist of:

% data      DxT array, data to sort
% spikeForms   N-dimensional cell of DxK spike templates
%samplingRate  the sampling rate of the data
% cinv      inverse of the covariance of the model. Will be computed if not
%           given

try
    %if no arguments are given, print a help message
    if nargin == 0
        disp('Usage: hmm_decode SourceFile[<source file> fileName <template file> [save] [hdf5] [SaveFile <file to save to>]');
        mlseq = [];
        ll = -Inf;
        return
    end
	%check if we are not given a filename, but Args.Group is given
	nargin
	if isempty(Args.fileName) && ~isempty(Args.Group) && ~isempty(Args.SourceFile)
		%try to locate the waveforms.bin file as well as the cut-file and get the spikeforms from those	
		%get the session name by parsing the source file
		if ischar(Args.Group)
			Args.Group = str2num(Args.Group);
		end
		[pth,f,e] = fileparts(Args.SourceFile);
	    idx = strfind(f,'_highpass');
		sessionName = f(1:idx-1);
		%construct waveforms file and cut file
		waveformsFile = sprintf('%s/../%sg%.4dwaveforms.bin',pth,sessionName,Args.Group);
		cutFile = sprintf('%s/../%sg%.4dwaveforms.cut',pth,sessionName,Args.Group);
		if exist(waveformsFile) && exist(cutFile)
			%get the waveforms and the cluster ids
			[t,wv] = nptLoadingEngine(waveformsFile);
			cids = load(cutFile);
			uc = unique(cids);
			uc = uc(uc>0);
			spkforms = {}
			spikeForms = zeros(length(uc),size(wv,2),size(wv,3)+1);
			for c=1:length(uc)
				spkform{uc(c)} = [0;squeeze(mean(wv(cids==uc(c),:,:),1))]';
				spikeForms(c,:,2:end) = mean(wv(cids==uc(c),:,:),1);
			end
		end
	%check if we are given spikeForms
	elseif isempty(Args.spikeForms) 
        if ~isempty(Args.reorder)
            if ischar(Args.reorder)
                reorder = load(Args.reorder);
            else
                reorder = Args.reorder;
            end
        end
		if ~Args.hdf5
			if isempty(Args.cinv)
				load(Args.fileName,'spkform','cinv')
			else
				load(Args.fileName,'spkform')
				cinv = Args.cinv;
			end;
		else
			if isempty(Args.hdf5Path)
				try
					spikeForms = hdf5read(Args.fileName,'/spikeForms');
				catch e
					spikeForms = hdf5read(Args.fileName,'/spkform');
					spikeForms = cell2array(spikeForms);
				end
            	try    
					cinv = hdf5read(Args.fileName,'/cinv');
				catch
				end
			else
				try
					spikeForms = hdf5read(Args.fileName, [Args.hdf5Path '/spikeForms']);
				catch e
					spikeForms = hdf5read(Args.fileName, [Args.hdf5Path '/spforms']);
					spikeForms = cell2array(spikeForms);
				end
				try
					cinv = hdf5read(Args.fileName,[Args.hdf5Path '/cinv']);
					Args.Channels = double(hdf5read(Args.fileName,[Args.hdf5Path '/channels']))+1
				catch
					%do nothing
				end
			end
			%need to permute; if hdf5, spikeforms are stored in row order,
			%i.e. numSpikeForms X nchannels X nstates;
			
			spikeForms = permute(spikeForms,[3,2,1]);
            %%debub
			if ischar(Args.maxCells)
				Args.maxCells = str2num(Args.maxCells);
			end
			if Args.maxCells > 0
				if size(spikeForms,1)>Args.maxCells
					disp(['A maximum of 30 cells can be sorted together.' num2str(size(spikeForms,1)-Args.maxCells) ' cells were discarded'] );
					spikeForms = spikeForms(1:Args.maxCells,:,:);
				end
			end
            %%%
			size(spikeForms)
            Z = zeros(size(spikeForms,1),size(spikeForms,2),size(spikeForms,3)+1);
            Z(:,:,2:end) = spikeForms;
            
			for i=1:size(spikeForms,1)
				%spkform{i} = squeeze(spikeForms(i,:,:));
				%TODO: this might not work for more than one channel. Apparently, squeeze remove all singleton dimensions, which means that the resulting dimension when there is only one channel reverts back to matlab's default one dimensional vector, which is a column vector
                spkform{i} = squeeze(Z(i,:,:));
				if size(spkform{i},1) > size(spkform{i},2)
					spkform{i} = spkform{i}';
				end
			end
			%p = hdf5read([filename '.hdf5'],'/p');
			%cinv = hdf5read([filename '.hdf5'],'/cinv');
		end
	end
	if ~exist('spkform') && ~isempty(Args.spikeForms)
		spkform = Args.spikeForms;
	end
	if ~isempty(Args.SourceFile) && isempty(Args.data)
        fparts = split(Args.SourceFile,'.');
        if fparts(end) == 'mat'
            fdata = load(Args.SourceFile);
            if isfield(fdata, 'rh')
                data = fdata.rh.data.analogData;
                samplingRate = fdata.rh.data.analogInfo.SampleRate;
            elseif isfield(fdata, 'highpassdata')
                data = fdata.highpassdata.data.data;
                samplingRate = fdata.highpassdata.data.sampling_rate;
            else
                error('Unknown input file structure');
            end
        else

            header = ReadUEIFile('Filename',Args.SourceFile,'Header');

            if header.headerSize == 73
                fid = fopen(Args.SourceFile,'r');
                [header.numChannels,header.samplingRate,scan_order] = nptParseStreamerHeader(fid);
                fclose(fid);
            elseif header.headerSize == 75
                fid = fopen(Args.SourceFile,'r');
                fseek(fid,4,0);
                header.numChannels = fread(fid, 1, 'uint16');
                header.transpose = fread(fid, 1, 'uint8');
                header.samplingRate = fread(fid, 1, 'uint32');
                fclose(fid);
            end
            samplingRate = header.samplingRate;
            M = memmapfile(Args.SourceFile,'format','int16','offset',header.headerSize);
        end
		%data = 	double(reshape(M.data,[header.numChannels,numel(M.data)/header.numChannels]));
        if ischar(Args.Channels)
            Args.Channels = str2num(Args.Channels);
        end
		channels = Args.Channels;
        if ~isempty(Args.reorder)
				data = 	double(reshape(M.data,[header.numChannels,numel(M.data)/header.numChannels]));
                if max(reorder)>size(data,1)
                    tdata = zeros(max(reorder),size(data,2));
                    tdata(reorder,:) = data;
                    data = tdata;
                    clear tdata;
                else
                    data = data(reorder,:);
                end
        end
        if ~isempty(Args.Channels)
			channels = Args.Channels;
        elseif ~isempty(Args.Group)
            %need to load the descriptor
            if ischar(Args.Group)
                Args.Group = str2num(Args.Group);
            end
            g = Args.Group;
            idx = strfind(Args.SourceFile,'highpass');
            idx = idx(end);
            if isempty(Args.DescriptorFile)
				try
					descriptor = ReadDescriptor([Args.SourceFile(1:idx-1) 'descriptor.txt']);
				catch
				end
            else
                descriptor = ReadDescriptor(Args.DescriptorFile);
            end

            %channels = find(descriptor.group==g);
			if exist('descriptor')
				channels = [];
				j = 0;
				for i=1:descriptor.number_of_channels
					if strcmpi(descriptor.state{i},'Active') 
						j = j+1;
						if descriptor.group(i)==g
							channels = [channels j]; 
						end
					end
				end
			else
				%if no descriptor could be found, simply use the group number as the channel number
				channels = Args.Group;
			end
			%get the data
			if ~exist('data')
				data = zeros(size(M.Data,1)/header.numChannels,length(channels));
				for c=1:length(channels)
					data(:,c) = M.Data(channels(c):header.numChannels:end);
				end
			elseif size(data,1) < size(data,2)
				data = data';
				if size(data,2) > 1
					data = data(:,channels);
				end
			end
            disp(['Sorting waveforms for group ' num2str(g) ' spanning channels ' num2str(channels) ' ...']);
            if length(channels)==0
                disp('Channel mismatch. Could not proceed');
                return
            end
            %data = 	double(reshape(M.data,[header.numChannels,numel(M.data)/header.numChannels]));
            %data = data(channels,:);
            %Args.Channels = channels;
        %else
        %    disp('Channel mismatch. Could not proceed')
        %    return
        end
    else
        %data = 	double(reshape(M.data,[header.numChannels,numel(M.data)/header.numChannels]));
		data = Args.data;
        %check that data has the right dimension
        if size(data,1) < size(data,2)
            data = data';
        end
    end
    if ~exist('channels')
        channels = 1:size(data,2);
    end
	if ~exist('cinv')
		if ~isempty(Args.cinv)
			cinv = Args.cinv;
		else
			C = cov(data);
			S = sqrt(diag(C));
			if size(data,2)==1
				%remove outliers
				thresh = Args.outlierThreshold;
				cinv = pinv(cov(data(abs(data)<thresh*S)));
			else
				warning('Outlier detection not implemented for more than 1 channel');
				cinv = pinv(cov(data));
			end
		end
	end
	if size(data,1) ~= length(channels)
		data = data';
	end
    if ~isempty(Args.maxSize)
        if ischar(Args.maxSize)
            Args.maxSize = str2num(Args.maxSize)
        end
        data = data(:,1:Args.maxSize);
    end
	% sort the data
	% length of the pieces to be sorted at once
	if isempty(Args.patchLength) 
		patchlength = 20000;
	elseif ischar(Args.patchLength)
			patchlength = str2num(Args.patchLength);
	else
		patchlength = Args.patchLength;
	end
	patchlength = min(patchlength,size(data,2));
	% spiking probability
	if isempty(Args.prob)
		%use the heuristic from the paper
		%p = 2^3KD/2
		p = 2.^(-3*size(spkform{1},2)*size(spkform{1},1)/2);
		%p = 1e-10;
	elseif ischar(Args.prob)
		p = str2num(Args.prob);
	else
		p = Args.prob;
	end
	p


	% mlseq is an NxT array with the states of all N templates at each time
	[mlseq,ll] = cutsort(data, spkform, cinv, patchlength, p);
	%save sequence to sorting file; if a source file was used, save to a file consistent with that name
	if Args.save
		if ~isempty(Args.SourceFile) || ~isempty(Args.SaveFile)
			if Args.parseOutput
				parseHMMOutput(mlseq,spikeForms(:,:,2:end),sessionName,Args.Group,data,samplingRate);
			end
			%parts = strsplit(Args.SourceFile,'_');
			%note strfind finds all the matches
            if isempty(Args.SaveFile)
                idx = strfind(Args.SourceFile,'highpass');
                if length(idx)>1
                    offset = idx(1)+length('highpass')+1;
                else
                    offset = 1;
                end
                idx = idx(end);
                if isempty(Args.Group) && ~isempty(Args.Channels)
                    if isempty(Args.DescriptorFile)
                        descriptor = ReadDescriptor([Args.SourceFile(1:idx-1) 'descriptor.txt']);
                    else
                        descriptor = ReadDescriptor(Args.DescriptorFile);
                    end
                    Args.Group = descriptor.group(descriptor.channel==Args.Channels(1));

                end
                nparts = strsplit(Args.SourceFile,'.');
                if ~isempty(Args.Group)
                    fname = sprintf('%sg%.4d.%.4d.mat',Args.SourceFile(offset:idx-1),Args.Group,str2num(nparts{end}));
                elseif ~isempty(Args.hdf5Path)
                    fname = sprintf('%s%s.%.4d.mat',Args.SourceFile(offset:idx-1),strrep(Args.hdf5Path,'/',''),str2num(nparts{end}));
                else
                    fname = sprintf('%s.%.4d.mat',Args.SourceFile(offset:idx-1),str2num(nparts{end}));
                end
            else
                fname = Args.SaveFile;
            end
			disp(['Saving data to file ' fname '...']);
			%if exist(fname)
			if ~exist('spikeForms')
				spikeForms = cell2array(spkform);
			end
			if ndims(spikeForms) == 2
				spikeForms = shiftdim(spikeForms,-1);
				spikeForms = permute(spikeForms,[2,1,3]);
			end
			save(fname,'mlseq','spikeForms','ll');
		else
			save([Args.fileName '.mat'],'mlseq','ll','-append');
			fname = [Args.fileName '.mat'];
		end
		if length(Args.Channels)>0
			Channels = Args.Channels;
			save(fname,'Channels','-append');
		end
	end
	catch err
		disp('An error occurred');
		err
        err.stack.file
        err.stack.name
        err.stack.line
        if isdeployed
            exit(100);
        end
	end
end

function [mlseq,LL] = cutsort(data, spkform, cinv, patchlength, p)

% data          DxT array, data to sort
% spkform       N-dimensional cell of DxK spike templates
% cinv          inverse of the covariance of the model
% patchlength   length of the pieces to be sorted at once
% p             spiking probability

overlaps = min(2,length(spkform));

if nargin<4
    patchlength=40000;
    p=1e-10;
end

% seq is the array with all concatenated pieces of state combinations
seq=zeros(1,size(data,2));
% specify window to sort
window=[1 patchlength];

disp(['sorting window ' num2str(window(1)) ':' num2str(window(2))])
% sort first window
[ll,seq(window(1):window(2))] = ...
    joshviterbi(data(:,window(1):window(2)),spkform,overlaps,cinv,p);

test = 1;
LL = ll;
while test
    % specify the window to sort next
    % find last entry, where all rings where non active
	ly = max(find(seq(window(1):window(2))==0));
	%if we happend to find the same window, increment by 1
	if ly ==1
		ly  = ly+1;
	end
    window(1) = ly+window(1)-1;
    window(2)=window(1)+patchlength;

    if window(2)>size(data,2)
        window(2)=size(data,2);
        test = 0;
    end

    disp(['sorting window ' num2str(window(1)) ':' num2str(window(2))])

    [ll,seq(window(1):window(2))] = ...
        joshviterbi(data(:,window(1):window(2)),spkform,overlaps,cinv,p);
	%add the log-likelihoods
	LL = LL + ll;
end

% separate the complex state combination into the individual states
mlseq=zeros(length(spkform),size(data,2));

% count the number of states per template
numstates = zeros(1,length(spkform));
for i=1:length(numstates)
    numstates(i)=size(spkform{i},2);
end

% loop over all time points
for j=1:size(data,2)
    % retrieve state combination from seq vector
    state=seq(j);
    % loop over all rings
    for i=1:length(numstates)
        % loop with state modulo numstates to calculate the separate states
        mlseq(i,j) = mod(state,numstates(i));
        % save remaining value
        state = round((state-mlseq(i,j))/numstates(i));
    end
end
end

function [ll,mlseq] = joshviterbi(data,spkform,overlaps,cinv,p)

% data          DxT array, data to sort
% spkform       N-dimensional cell of DxK spike templates
% overlaps      number of simultaneously active rings allowed
% cinv          inverse of the covariance of the model
% p             spiking probability

if nargin<5
    p = 1e-10;
end

numspikes = length(spkform);
mu_cell = cell(1,numspikes);

% loop over all templates and represent as vector
for i = 1:numspikes
    s = size(spkform{i});
    mu_cell{i} = reshape(spkform{i},1,s(1)*s(2));
end

% represent the data as vector
s = size(data);
d = reshape(data,1,s(1)*s(2));

% how many states per template
numstates = zeros(1,length(spkform));
% cell with the transition matrices
trmatrix_cell = cell(1,length(spkform));
% cell with the active states per ring
active_states = cell(1,length(spkform));

for i=1:length(spkform)
    % how many states does template i have?
    %numstates(i) = length(spkform{i});
    numstates(i) = size(spkform{i},2);
    % create the transition matrix
    trmatrix_cell{i} = neurontrmatrix(0.5,numstates(i));
    % pass vector with active states
    active_states{i} = [0 ones(1,numstates(i)-1)];
end

% calculate the viterbi
% d         data
% p         vector with spiking probabilities
% cinv      inverse of the data covariance
% numstates templates per ring
% mu_cell   templates
% []        ???
% overlaps  how many active rings at once
% size(data,1)
[mlseq,ll] = viterbi_nd(d,ones(1,length(spkform))*p,...
    cinv,numstates,mu_cell,[],overlaps,size(data,1));

end

function m = neurontrmatrix(p,n)
% create NxN transitionmatrix according to ring structure

m = zeros(n);
m(1:(n-1),2:n)=eye(n-1);
m(1,1)=p;
m(1,2)=1-p;
m(n,1)=1;

end
