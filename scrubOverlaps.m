function mlseq = scrubOverlaps(mlseq, cinv, spikeForms, data, theta)
  max_state = size(spikeForms,3);
  noise_threshold = theta*sqrt(1/cinv);
  %find portions with overlaps
  oidx = find(sum(mlseq > 0,1)>1);
  %start of each overlap
  sidx = find(any(mlseq(:,oidx)==1,1));
  %end of each overlap
  eidx = find(any(mlseq(:,oidx)==max_state,1));
  %loop through each overlap
  nd = length(data);
  for i = 1:length(eidx)
    j = oidx(sidx(i));
    %find the beginning of this section, i.e. the first point where
    while j > 0 && any(mlseq(:,j))
      j = j -1;
    end
    segment_start = j+1;
    %find the end
    j = oidx(eidx(i));
    while j < nd && any(mlseq(:,j))
      j = j + 1;
    end
    segment_end = j-1;
    dd = data(segment_start:segment_end);
    peak = max(abs(dd));
    if peak < noise_threshold
      %cancel the overlap
      mlseq(:,segment_start:segment_end) = 0;
    end
  end
end
