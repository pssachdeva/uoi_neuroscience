slCharacterEncoding('US-ASCII')

ptrain = 0.9;

stim_path = 'DynRip.cchspct.6oct.100hz.mat';
load(stim_path)
stim{1} = spct(3:98,1:round(ptrain*size(spct,2)));
test_stim = spct(3:98, round(ptrain*size(spct,2))+1:end);

base_response_file = '/storage/data/ecog/Wvlt_4to1200_54band_CAR1/Wave';
mark_file = '/storage/data/ecog/Wvlt_4to1200_54band_CAR1/mrk11.htk';
for idx=2:2
    disp(idx)
    response_file = join([base_response_file, int2str(idx), '.htk']);
    [response, n_fs] = readhtk(response_file);
    response = response';
    [mark, m_fs] = readhtk(mark_file);

    % sampling rate is in kHz
    n_fs = round(1000 * n_fs);
    m_fs = round(1000 * m_fs);

    ids = find(mark > 0.8 * max(mark));

    fs_mult = round(m_fs / (50e6/16384));
    d_ids = round((n_fs/(fs_mult * (50e6/16384))) * [ids(1) ids(end)]);

    %%%STRF parameters
    params{1} =0;
    params{2}{1} = 20;
    params{2}{2} = [0:19];
    params{3} = 'DirectFit';
    %need to look at what these are....but these work for high-gamma!
    stparams{2} = [.1 .05 .01 .005];
    stparams{1} = [0 3 6];

    HGfrange = [28:35];

    % calculate STRF
    %extract the baseline activity,data before and after stim, plus some
    bstmp=[];
    bstmp = [abs(response(:,1:d_ids(1,1)-5)) abs(response(:,d_ids(1,2)+5:end))];

    %baseline statistics
    mu = mean(bstmp,2); sd = std(bstmp');

    %calculate the evoked response and z-score relative to silence
    zscr = abs(response(:,d_ids(1,1):d_ids(1,2)))-repmat(mu,1,d_ids(1,2)-d_ids(1,1)+1);
    zscr = zscr./repmat(sd',1,d_ids(1,2)-d_ids(1,1)+1);
    resp = mean(zscr(HGfrange,:));

    %resample neural at 100 Hz
    resp  = resample(resp,100,n_fs);
    resp_train = resp(1:round(ptrain*length(resp)));
    if length(resp_train) ~= size(stim{1},2)
        resp_train = resp_train(1:size(stim{1},2));
    end
    
    resp_test = resp(round(ptrain*length(resp))+1:end);
    if length(resp_test) ~= size(test_stim, 2)
        resp_test = resp_test(1:size(test_stim,2));
    end

    rsp{1} = resp_train;
    %extract and save strf
    [strf, outmodelParam] = STRFestimate(stim,rsp,100,params,stparams);
    
    %base_group = join(['/Wave', int2str(idx)]);
    %h5create('matlab_strf.h5', join([base_group, '/strf']), size(strf))
    %h5create('matlab_strf.h5', join([base_group, '/train_stim']), size(stim{1}))
    %h5create('matlab_strf.h5', join([base_group, '/test_stim']), size(test_stim))
    %h5create('matlab_strf.h5', join([base_group, '/train_resp']),  size(resp_train))
    %h5create('matlab_strf.h5', join([base_group, '/test_resp']), size(resp_test))
    %h5write('matlab_strf.h5', join([base_group, '/strf']), strf)
    %h5write('matlab_strf.h5', join([base_group, '/train_stim']), stim{1})
    %h5write('matlab_strf.h5', join([base_group, '/test_stim']), test_stim)
    %h5write('matlab_strf.h5', join([base_group, '/train_resp']),  resp_train)
    %h5write('matlab_strf.h5', join([base_group, '/test_resp']), resp_test)
end