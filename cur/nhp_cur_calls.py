import subprocess

# indy_20160407_02
subprocess.call([
    'python3', 'column_select_m1_nhp.py',
    '--data_path=/storage/data/nhp/indy_20160407_02.mat',
    '--results_path=/home/psachdeva/fits/uoineuro/cur/indy_20160407_02.h5',
    '--bin_width=0.25',
    '--left_idx=0',
    '--right_idx=0',
    '--reps=20',
    '--region=M1',
    '--min_max_k=30',
    '--max_max_k=136',
    '--max_k_spacing=2',
    '--n_boots=20',
    '--boots_frac=0.9',
    '--stability_selection=0.9',
    '--verbose'
])

subprocess.call([
    'python3', 'kalman_filter_m1_nhp.py',
    '--data_path=/storage/data/nhp/indy_20160407_02.mat',
    '--results_path=/home/psachdeva/fits/uoineuro/cur/indy_20160407_02.h5',
    '--bin_width=0.25',
    '--left_idx=0',
    '--right_idx=0',
    '--reps=20',
    '--min_max_k=30',
    '--max_max_k=136',
    '--max_k_spacing=2',
    '--train_frac=0.8',
    '--verbose'
])

# indy_20160411_01
subprocess.call([
    'python3', 'column_select_m1_nhp.py',
    '--data_path=/storage/data/nhp/indy_20160411_01.mat',
    '--results_path=/home/psachdeva/fits/uoineuro/cur/indy_20160411_01.h5',
    '--bin_width=0.25',
    '--left_idx=0',
    '--right_idx=0',
    '--reps=20',
    '--region=M1',
    '--min_max_k=30',
    '--max_max_k=146',
    '--max_k_spacing=2',
    '--n_boots=20',
    '--boots_frac=0.9',
    '--stability_selection=0.9',
    '--verbose'
])

subprocess.call([
    'python3', 'kalman_filter_m1_nhp.py',
    '--data_path=/storage/data/nhp/indy_20160411_01.mat',
    '--results_path=/home/psachdeva/fits/uoineuro/cur/indy_20160411_01.h5',
    '--bin_width=0.25',
    '--left_idx=0',
    '--right_idx=0',
    '--reps=20',
    '--min_max_k=30',
    '--max_max_k=146',
    '--max_k_spacing=2',
    '--train_frac=0.8',
    '--verbose'
])
