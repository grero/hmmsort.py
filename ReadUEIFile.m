function data = ReadUEIFile(varargin)

% data = ReadUEIFile(varargin) returns a structure,
%      data, which contains the following fields.
%               data.headerSize
%               data.samplingRate
%               data.numChannels
%               data.inputRange
%               data.numBits
%               data.scanOrder
%               data.rawdata
%               data.timevector
%
%     data = ReadUEIFile('P1', V1, 'P2', V2,...) specifies the
%      amount of data to be read from the file, and the format of the
%      DATA matrix.
%
%        Valid Property Names (P1, P2,...) and Property Values (V1, V2,...)
%        include:
%           FileName   -  'filename.bin'
%           Samples    -  [sample range] example, [1 30000]
%           Channels   -  [channel indices] 1 2 3 etc. One based indexing
%           Units      -  {'Volts'} 'MicroVolts', 'MilliVolts' or  'Daqunits'
%           Header     -  flag to only read header(default 0)
%
%        The Samples and Time properties are mutually exclusive, i.e.
%        either Samples or Time can be defined at once.
%
%        The default values for the Units properties
%        are indicated by braces {}.
%

%Get Input Args
Args = struct('FileName',[],'Samples',[], ...
    'Channels',[],'Units','Volts','Header',0,'num_channels',-1);
Args = getOptArgs(varargin,Args,'flags',{'Header'});

%Open Raw data file
fid = fopen(Args.FileName,'r','ieee-le');

%Read Header
data.headerSize = fread(fid,1,'int32');
data.samplingRate = fread(fid,1,'uint32');
if data.samplingRate == 30000;
    % This is the true samplingrate determined with a known signal (Master8).
    data.samplingRate = 29990;
end
data.numChannels = fread(fid,1,'uchar');
if(Args.num_channels~=-1)
	data.numChannels = Args.num_channels;
end
data.inputRange = fread(fid,2,'float64');
data.numBits = fread(fid,1,'uchar');
data.scanOrder = fread(fid,data.numChannels,'uchar');

if Args.Header
    data.rawdata = 'only retrieved header';
    data.time = 'only retrieved header';
else
    % Read a chunk of the data or the entire file
    if ~isempty(Args.Samples)
        if strcmpi(Args.Samples,'end')
            dirlist = nptDir(Args.FileName);
            endPos = (dirlist.bytes-data.headerSize);
            numSamples = data.samplingRate*1;
            startPos = endPos-(numSamples*data.numChannels*2);
            Args.Samples = [(endPos/2/data.numChannels)-(numSamples-1) (endPos/2/data.numChannels)];
        else
            numSamples = diff(Args.Samples)+1;
            startPos = (Args.Samples(1)-1)*data.numChannels*2;
        end
    else % read the entire file
        s=fseek(fid,0,'eof');
        numSamples = (ftell(fid)-data.headerSize)/2/data.numChannels; %2 b/c it's a 16bit
        startPos = 0;
        Args.Samples = [1 numSamples];
    end
    
    % Move to the beginning of the data
    s=fseek(fid,data.headerSize,'bof');
    
    % Move to the reading position within the data
    s=fseek(fid,startPos,'cof');

    % Load the raw data chunk across all channels
    rawdata = fread(fid , [data.numChannels  numSamples] , 'int16');
    
    % Include only specified Channels
    % Channel numbers are indexed by 1 not the scanOrder numbers
    if ~isempty(Args.Channels)
        rawdata = rawdata(Args.Channels,:);
        data.numChannels = length(Args.Channels);
    end
    
    % Close the Data File
    s=fclose(fid);
    if ~strcmpi(Args.Units,'daqunits')
        rawdata = rawdata/(2^(data.numBits+1)/(diff(data.inputRange)/2))+ mean(data.inputRange);
        if strcmpi(Args.Units,'MilliVolts')
            rawdata = rawdata.*1e3;
        elseif strcmpi(Args.Units,'MicroVolts')
            rawdata = rawdata.*1e6;
        end
    end
    % Place rawdata into 
    data.rawdata=rawdata;
    data.timevector = Args.Samples(1):Args.Samples(2);
end




















%     %how much memory do we have?
%     m = feature('DumpMem');
%     if (.9*m)<(numSamples*numChannels*8) %8 b/c it's a double(64bit).
%         fprintf('\nNot enough memory to read data, specify a smaller range or less channels!!\nLargest available memory block is %d bytes\n',m)
%         data.rawdata = [NaN];
%         return
%     end


%so we will try to read all of the samples across all channels but if
%we don't have enough memory then we will have to loop.

% while(we don't have all the data)
%1.  how much memory is available
%2.  read a large chunk of the data across all channels that is smaller
%    than 90% of the available memory.
%3.  remove unwanted channels
%4.  concatonate new data

%     while size(rawdata,2)<numSamples
%         m = feature('DumpMem'); %get new available memory
%         if ((numSamples-size(rawdata,2))*data.numChannels*8)<(.9*m)        %can we read the rest?
%             newdata = fread(fid , [data.numChannels  (numSamples-size(rawdata,2))] , 'int16');
%         else
%             newdata = fread(fid , [data.numChannels  floor(.9*m/data.numChannels/8)] , 'int16');
%         end
%         if ~isempty(Args.Channels)
%             newdata = newdata(Args.Channels,:);
%         end
%         rawdata = [rawdata newdata];
%     end
