function [] = transferHippocampusData()
%TRANSFERHIPPOCAMPUSDATA To be called in the hmmsort_pbs.py to be sumitted
%as a job onto PBS queue.
% 
%   Detailed explanation goes here

cwd = pwd;

dataName = strrep(cwd,filesep,'_');
fileName = [dataName,'.tar.gz']; % tar.gz name 

indexDay = strfind(cwd,'2018');

picassoDir = fullfile(filesep,'volume1','Hippocampus','Data','picasso');
dayStr = cwd(indexDay:indexDay+7);
dayDir = fullfile(picassoDir, dayStr); % directory of the day
targetDir = fullfile(dayDir, 'transferTemp', dataName); % directory to save the tar.gz file temporarily in hippocampus

sshHippocampus = 'ssh -p 8398 hippocampus@cortex.nus.edu.sg';

cd ..

system(['tar -cvzf ',fileName,' ',cwd]);
disp(' ');

system(['scp -P 8398 ',fileName,' hippocampus@cortex.nus.edu.sg:~/']);
disp(['Secured copied ',fileName,' to home directory of hippocampus ...']);
disp(' ');

system([sshHippocampus,' mkdir -p ', targetDir]);
disp(['Made a directory ',targetDir,' ...']);
disp(' ');

system([sshHippocampus,' mv -v ',fileName,' ',targetDir]);
disp(' ');

system([sshHippocampus,' tar -xvzf ',fullfile(targetDir,fileName),' -C ',targetDir]);
disp(' ')

system([sshHippocampus, ' find ',targetDir,' -name ',dayStr,' | while IFS= read file; do ',sshHippocampus,' cp -vrRup $file ',picassoDir,'; done']);
disp(' ');

%system([sshHippocampus,' rm -v ',fullfile(targetDir,dataName)]);

end

