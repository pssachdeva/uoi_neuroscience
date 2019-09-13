import subprocess


print('20080516_R1, Recording 0')
cells = [0, 2, 3, 4, 5]
for cell in cells:
    subprocess.call([
        'python3', 'ret1_strf_script.py',
        '--data_path=/Users/psachdeva/data/ret1/data/20080516_R1.mat',
        '--random_path=/Users/psachdeva/data/ret1/data/ran1.bin',
        '--results_path=/Users/psachdeva/fits/uoineuro/tuning/ret1/20080516_R1.h5',
        '--results_group=Lasso',
        '--method=Lasso',
        '--verbose',
        '--standardize',
        '--cv=5',
        '--recording_idx=0',
        '--cell=%s' % cell,
        '--window_length=0.40'
    ])

    subprocess.call([
        'python3', 'ret1_strf_script.py',
        '--data_path=/Users/psachdeva/data/ret1/data/20080516_R1.mat',
        '--random_path=/Users/psachdeva/data/ret1/data/ran1.bin',
        '--results_path=/Users/psachdeva/fits/uoineuro/tuning/ret1/20080516_R1.h5',
        '--results_group=UoI_AIC',
        '--method=UoI_Lasso',
        '--verbose',
        '--standardize',
        '--estimation_score=AIC',
        '--recording_idx=0',
        '--cell=%s' % cell,
        '--window_length=0.40'
    ])

print('20080516_R1, Recording 2')
cells = [0, 2]
for cell in cells:
    subprocess.call([
        'python3', 'ret1_strf_script.py',
        '--data_path=/Users/psachdeva/data/ret1/data/20080516_R1.mat',
        '--random_path=/Users/psachdeva/data/ret1/data/ran1.bin',
        '--results_path=/Users/psachdeva/fits/uoineuro/tuning/ret1/20080516_R1.h5',
        '--results_group=Lasso',
        '--method=Lasso',
        '--verbose',
        '--standardize',
        '--cv=5',
        '--recording_idx=2',
        '--cell=%s' % cell,
        '--window_length=0.40'
    ])

    subprocess.call([
        'python3', 'ret1_strf_script.py',
        '--data_path=/Users/psachdeva/data/ret1/data/20080516_R1.mat',
        '--random_path=/Users/psachdeva/data/ret1/data/ran1.bin',
        '--results_path=/Users/psachdeva/fits/uoineuro/tuning/ret1/20080516_R1.h5',
        '--results_group=UoI_AIC',
        '--method=UoI_Lasso',
        '--verbose',
        '--standardize',
        '--estimation_score=AIC',
        '--recording_idx=2',
        '--cell=%s' % cell,
        '--window_length=0.40'
    ])

print('20080628_R4, Recording 2')
# stopped at cell 2
cells = [0, 2, 3, 13, 14]
for cell in cells:
    subprocess.call([
        'python3', 'ret1_strf_script.py',
        '--data_path=/Users/psachdeva/data/ret1/data/20080628_R4.mat',
        '--random_path=/Users/psachdeva/data/ret1/data/ran1.bin',
        '--results_path=/Users/psachdeva/fits/uoineuro/tuning/ret1/20080628_R4.h5',
        '--results_group=Lasso',
        '--method=Lasso',
        '--verbose',
        '--standardize',
        '--cv=5',
        '--recording_idx=0',
        '--cell=%s' % cell,
        '--window_length=0.40'
    ])

    subprocess.call([
        'python3', 'ret1_strf_script.py',
        '--data_path=/Users/psachdeva/data/ret1/data/20080628_R4.mat',
        '--random_path=/Users/psachdeva/data/ret1/data/ran1.bin',
        '--results_path=/Users/psachdeva/fits/uoineuro/tuning/ret1/20080628_R4.h5',
        '--results_group=UoI_AIC',
        '--method=UoI_Lasso',
        '--verbose',
        '--standardize',
        '--estimation_score=AIC',
        '--recording_idx=0',
        '--cell=%s' % cell,
        '--window_length=0.40'
    ])
