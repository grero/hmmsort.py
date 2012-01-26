%% load the data and learn model parameters

[data, spkform, cinv] = ...
    hmm_learn('channel1','data_with_parameters');

%% use learned model parameters to sort the data

mlseq = hmm_decode('data_with_parameters', 20000, 1e-20);
%% plot all
figure;plot(data(1,:));hold on;
plot(spkform{1}(1,mlseq(1,:)+1),'r');
plot(spkform{2}(1,mlseq(2,:)+1),'g');
plot(spkform{3}(1,mlseq(3,:)+1),'m');