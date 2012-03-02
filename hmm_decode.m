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


function mlseq = hmm_decode(filename, patchlength, p,varargin)

Args = struct('SourceFile',[],'Channels',[],'save',0,'Group','','hdf5',0);
Args.flags = {'save','hdf5'};
[Args,varargin] = getoptargs(varargin,Args);
addpath helper_functions
% specify file to sort, should consist of:

% data      DxT array, data to sort
% spkform   N-dimensional cell of DxK spike templates
% cinv      inverse of the covariance of the model
try
	if ~Args.hdf5
		load([filename '.mat'])
	else
		spikeForms = hdf5read([filename '.hdf5'],'/spikeForms');
		%need to permute
		spikeForms = permute(spikeForms,[3,2,1]);
		for i=1:size(spikeForms,1)
			spkform{i} = squeeze(spikeForms(i,:,:));
		end
		%p = hdf5read([filename '.hdf5'],'/p');
		cinv = hdf5read([filename '.hdf5'],'/cinv');
	end
	if ~isempty(Args.SourceFile)
		header = ReadUEIFile('Filename',Args.SourceFile,'Header');
		if header.headerSize == 73
			fid = fopen(Args.SourceFile,'r');
			[header.numChannels,header.samplingRate,scan_order,header.headerSize] = nptParseStreamerHeader(fid);
			fclose(fid);
		end
		M = memmapfile(Args.SourceFile,'format','int16','offset',header.headerSize);
		if header.numChannels > size(spkform{1},1) 
			if ischar(Args.Channels)
				Args.Channels = str2num(Args.Channels);
			end
			if length(Args.Channels)~=0
				data = 	double(reshape(M.data,[header.numChannels,numel(M.data)/header.numChannels]));
				data = data(Args.Channels,:);
			elseif ~isempty(Args.Group)
				%need to load the descriptor
				if ischar(Args.Group)
					Args.Group = str2num(Args.Group);
				end
				g = Args.Group;
				idx = strfind(Args.SourceFile,'highpass');
				idx = idx(end);
				descriptor = ReadDescriptor([Args.SourceFile(1:idx-1) 'descriptor.txt']);
				channels = find(descriptor.group==g);
				disp(['Sorting waveforms for group ' num2str(g) ' spanning channels ' num2str(channels) ' ...']);
				if length(channels)==0
					disp('Channel mismatch. Could not proceed');
					return
				end
				data = 	double(reshape(M.data,[header.numChannels,numel(M.data)/header.numChannels]));
				data = data(channels,:);
				Args.Channels = channels;
			else
				disp('Channel mismatch. Could not proceed')
				return
			end
		else
			data = 	double(reshape(M.data,[header.numChannels,numel(M.data)/header.numChannels]));
		end
	catch
		disp('An error occurred');
		lasterror.message
		exit(100);
	end
end

% sort the data
% length of the pieces to be sorted at once
if nargin < 2
    patchlength = 20000;
else
	if ischar(patchlength)
		patchlength = str2num(patchlength);
	end
end
patchlength = min(patchlength,size(data,2));
% spiking probability
if nargin <3
    p = 1e-10;
else
	if ischar(p)
		p = str2num(p);
	end
end


% mlseq is an NxT array with the states of all N templates at each time
mlseq = cutsort(data, spkform, cinv, patchlength, p);
%save sequence to sorting file; if a source file was used, save to a file consistent with that name
if Args.save
	if ~isempty(Args.SourceFile)
		parts = strsplit(Args.SourceFile,'_');
		idx = strfind(Args.SourceFile,'highpass');
		if isempty(Args.Group) && ~isempty(Args.Channels)
			descriptor = ReadDescriptor([Args.SourceFile(1:idx-1) 'descriptor.txt']);
			Args.Group = descriptor.group(descriptor.channel==Args.Channels(1));
		end
		nparts = strsplit(Args.SourceFile,'.');
		fname = sprintf('hmmsort/%sg%.4d.%.4d.mat',Args.SourceFile(1:idx-2),Args.Group,str2num(nparts{end}));
		disp(['Saving data to file ' fname '...']);
		%if exist(fname)
		save(fname,'mlseq','spikeForms');
		%else
		%	save(fname,'mlseq','spikeForms');
		%end
	else
		save([filename '.mat'],'mlseq','-append');
		fname = [filename '.mat'];
	end
	if length(Args.Channels)>0
		Channels = Args.Channels;
		save(fname,'Channels','-append');
	end
end
end

function mlseq = cutsort(data, spkform, cinv, patchlength, p)

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
while test
    % specify the window to sort next
    % find last entry, where all rings where non active
    window(1) = max(find(seq(window(1):window(2))==0))+window(1)-1;
    window(2)=window(1)+patchlength;

    if window(2)>size(data,2)
        window(2)=size(data,2);
        test = 0;
    end

    disp(['sorting window ' num2str(window(1)) ':' num2str(window(2))])

    [ll,seq(window(1):window(2))] = ...
        joshviterbi(data(:,window(1):window(2)),spkform,overlaps,cinv,p);
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
    numstates(i) = length(spkform{i});
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
