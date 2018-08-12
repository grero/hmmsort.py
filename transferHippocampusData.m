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
tempDir = fullfile(dayDir, 'transferTemp'); % direcotry of the temporary folder
targetDir = fullfile(tempDir, dataName); % directory to save the tar.gz file temporarily in hippocampus

sshHippocampus = 'ssh -p 8398 hippocampus@cortex.nus.edu.sg';

cd ..

system(['tar -cvzf ',fileName,' ',cwd]);
disp(' ');

[flag, count] = resetFlags;
while ~flag && count < 100
    try
        system(['scp -P 8398 ',fileName,' hippocampus@cortex.nus.edu.sg:~/']);
        disp(['Secured copied ',fileName,' to home directory of hippocampus ...']);
        flag = 1;
    catch
        disp('Retrying scp tar file to home directory of hippocampus...')
        pause(10)
    end
end
disp(' ');

[flag, count] = resetFlags;
while ~flag && count < 100
    try
        system([sshHippocampus,' mkdir -p ', targetDir]);
        disp(['Made a directory ',targetDir,' ...']);
        flag = 1;
    catch
        disp('Retrying making temporary directory...')
        pause(10)
    end
end
disp(' ');

flag = 0;
count = 1;
while ~flag && count < 100
    try
        system([sshHippocampus,' mv -v ',fileName,' ',targetDir]);
        flag = 1;
    catch
        disp('Retrying moving tar file...')
    end
end
disp(' ');

[flag, count] = resetFlags;
while ~flag && count < 100
    try
        system([sshHippocampus,' tar -xvzf ',fullfile(targetDir,fileName),' -C ',targetDir]);
        flag = 1;
    catch
        disp('Retrying extracting tar file...')
    end
end
disp(' ')

[flag, count] = resetFlags;
while ~flag && count < 100
    try
        system([sshHippocampus, ' find ',targetDir,' -name ',dayStr,' | while IFS= read file; do ',sshHippocampus,' cp -vrRup $file ',picassoDir,'; done']);
        flag = 1;
	delete(fileName); % delete tar file
	system([sshHippocampus,' rm -rv ',fullfile(tempDir,[dataName,'*'])]) % remove directory in hippocapus temporary folder
        fid = fopen(fullfile(cwd,'transferred.txt'),'w'); % to mark the channel has been successfully transferred
        fclose(fid);
   catch
        disp('Retrying moving files to proper places...')
    end
end
disp(' ');

%system([sshHippocampus,' rm -v ',fullfile(targetDir,dataName)]);

end

function [flag, count] = resetFlags()
flag = 0;
count = 1;
end

