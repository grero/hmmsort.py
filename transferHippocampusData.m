function [] = transferHippocampusData()
%TRANSFERHIPPOCAMPUSDATA To be called in the hmmsort_pbs.py to be sumitted
%as a job onto PBS queue.
% 
%   Detailed explanation goes here

cwd = pwd;

indexStarting = strfind(cwd,'2018');

picassoDir = fullfile(filesep,'volume1','Hippocampus','Data','picasso');
dayStr = [cwd(indexStarting:indexStarting+7),'temp'];
targetDir = fullfile(picassoDir, dayStr);

cd ..

system('tar -czf data.tar.gz cwd');

system(['scp -P 8398 testTarTar.tar.gz hippocampus@cortex.nus.edu.sg:',targetDir]);

system('ssh -p 8398 hippocampus@cortex.nus.edu.sg');

system(['mkdir -p ', targetDir]);

cd(targetDir)

system('tar -xzf data.tar.gz');


end

