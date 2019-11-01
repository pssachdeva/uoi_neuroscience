"""Calls poisson.py script on nhp data, for reproducing UoI Poisson / glmnet
results."""
import os
import subprocess

base = os.path.join(os.environ['HOME'])

data_base = os.path.join(base, 'data/nhp')
results_base = os.path.join(base, 'fits/uoineuro/coupling/nhp')
monkeys = ['indy_20160407_02',
           'indy_20160418_01']

# iterate over monkeys
for idx, monkey in enumerate(monkeys):
    print('Monkey ' + str(idx))
    subprocess.call([
        'python3', 'poisson.py',
        '--dataset=NHP',
        '--data_path=' + os.path.join(data_base, monkey) + '.mat',
        '--results_path=' + os.path.join(results_base, monkey + '.h5'),
        '--results_group=glmnet_poisson',
        '--fitter=glmnet',
        '--n_folds=10',
        '--standardize',
        '--random_state=2332',
        '--verbose'
    ])

    subprocess.call([
        'python3', 'poisson.py',
        '--dataset=PVC11',
        '--data_path=' + os.path.join(data_base, monkey) + '.mat',
        '--results_path=' + os.path.join(results_base, monkey + '.h5'),
        '--results_group=uoi_poisson_log',
        '--estimation_score=log',
        '--fitter=UoI_Poisson',
        '--n_folds=10',
        '--standardize',
        '--random_state=2332',
        '--verbose'
    ])

    subprocess.call([
        'python3', 'poisson.py',
        '--dataset=PVC11',
        '--data_path=' + os.path.join(data_base, monkey) + '.mat',
        '--results_path=' + os.path.join(results_base, monkey + '.h5'),
        '--results_group=uoi_poisson_AIC',
        '--estimation_score=AIC',
        '--fitter=UoI_Poisson',
        '--n_folds=10',
        '--standardize',
        '--random_state=2332',
        '--verbose'
    ])

    subprocess.call([
        'python3', 'poisson.py',
        '--dataset=PVC11',
        '--data_path=' + os.path.join(data_base, monkey) + '.mat',
        '--results_path=' + os.path.join(results_base, monkey + '.h5'),
        '--results_group=uoi_poisson_BIC',
        '--estimation_score=BIC',
        '--fitter=UoI_Poisson',
        '--n_folds=10',
        '--standardize',
        '--random_state=2332',
        '--verbose'
    ])
