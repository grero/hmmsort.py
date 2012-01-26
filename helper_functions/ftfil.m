function fildata = ftfil(data,frequency,hi_pass,lo_pass)

s=size(data);
fildata=data;

for i=1:s(1)
    y=fft(data(i,:));
    % cut of everysting below hi_pass Hz
    lim=round((hi_pass/frequency)*length(y));
    y(1:lim)=0;
    y(end-lim+2:end)=0;
    %         % cut of everysting above lo_pass Hz
    lim=round((lo_pass/frequency)*length(y));
    y(lim:end-lim+1)=0;
    fildata(i,:)=real(ifft(y));
end
