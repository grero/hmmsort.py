function [] = transferHippocampusData()
%TRANSFERHIPPOCAMPUSDATA To be called in the hmmsort_pbs.py to be sumitted
%as a job onto PBS queue.
% 
%   Detailed explanation goes here

cwd = pwd;

dataName = strrep(cwd,filesep,'_');
dataName = [dataName,'.tar.gz']; % tar.gz name 

indexDay = strfind(cwd,'2018');

picassoDir = fullfile(filesep,'volume1','Hippocampus','Data','picasso');
dayStr = cwd(indexDay:indexDay+7);
targetDir = fullfile(picassoDir, dayStr, 'transferTemp'); % directory to save the tar.gz file temporarily in hippocampus

cd ..

system(['tar -czf ',dataName,' cwd']);

system(['scp -P 8398 ',dataName,' hippocampus@cortex.nus.edu.sg:~/']);

system('ssh -p 8398 hippocampus@cortex.nus.edu.sg');

system(['mkdir -p ', targetDir]);

system(['mv ',dataName,' ',targetDir]);

cd(targetDir)

system(['tar -xzf ',dataName]);

system("find . -name '2018????' | while IFS= read file; do mv $file; done");

system(['rm ',dataName]);

end

