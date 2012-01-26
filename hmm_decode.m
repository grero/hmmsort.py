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


function mlseq = hmm_decode(filename, patchlength, p)

addpath helper_functions
% specify file to sort, should consist of:

% data      DxT array, data to sort
% spkform   N-dimensional cell of DxK spike templates
% cinv      inverse of the covariance of the model
load([filename '.mat'])

% sort the data
% length of the pieces to be sorted at once
if nargin < 2
    patchlength = 20000;
end

% spiking probability
if nargin <3
    p = 1e-10;
end

% mlseq is an NxT array with the states of all N templates at each time
mlseq = cutsort(data, spkform, cinv, patchlength, p);

end

function mlseq = cutsort(data, spkform, cinv, patchlength, p)

% data          DxT array, data to sort
% spkform       N-dimensional cell of DxK spike templates
% cinv          inverse of the covariance of the model
% patchlength   length of the pieces to be sorted at once
% p             spiking probability

overlaps = 2;

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