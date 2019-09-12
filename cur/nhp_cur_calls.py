import subprocess

monkeys = ['indy_20160407_02',
           'indy_20160411_01',
           'indy_20160411_02',
           'indy_20160418_01',
           'indy_20160420_01',
           'indy_20160426_01',
           'indy_20160622_01',
           'indy_20160627_01']
lefts = ['0', '0', '0', '0', '10', '0', '4000', '0']
rights = ['0', '0', '2500', '0', '6000', '5700', '8000', '13200']
k_maxes = ['136', '146', '147', '164', '169', '215', '230', '224']

for monkey, left, right, k_max in zip(monkeys, lefts, rights, k_maxes):
    print(monkey)

    subprocess.call([
        'python3', 'column_select_m1_nhp.py',
        '--data_path=/storage/data/nhp/' + monkey + '.mat',
        '--results_path=/home/psachdeva/fits/uoineuro/cur/' + monkey + '.h5',
        '--bin_width=0.25',
        '--left_idx=' + left,
        '--right_idx=' + right,
        '--reps=20',
        '--region=M1',
        '--min_max_k=30',
        '--max_max_k=' + k_max,
        '--max_k_spacing=2',
        '--n_boots=20',
        '--boots_frac=0.9',
        '--stability_selection=0.9',
        '--verbose'
    ])

    subprocess.call([
        'python3', 'kalman_filter_m1_nhp.py',
        '--data_path=/storage/data/nhp/' + monkey + '.mat',
        '--results_path=/home/psachdeva/fits/uoineuro/cur/' + monkey + '.h5',
        '--bin_width=0.25',
        '--left_idx=' + left,
        '--right_idx=' + right,
        '--reps=20',
        '--min_max_k=30',
        '--max_max_k=' + k_max,
        '--max_k_spacing=2',
        '--train_frac=0.8',
        '--verbose'
    ])
