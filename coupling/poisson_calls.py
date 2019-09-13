import os
import subprocess

base = os.path.join(os.environ['HOME'])

data_base = os.path.join(base, 'data/pvc11/data/spikes_gratings')
results_base = os.path.join(base, 'fits/uoineuro/coupling/pvc11')

monkeys = ['data_monkey1_gratings.mat',
           'data_monkey2_gratings.mat',
           'data_monkey3_gratings.mat']

# PVC11, Monkey 1
for idx, monkey in enumerate(monkeys):
    print('Monkey ' + str(idx))
    subprocess.call([
        'python3', 'poisson.py',
        '--dataset=PVC11',
        '--data_path=' + os.path.join(data_base, monkey),
        '--results_path=' + os.path.join(results_base, 'monkey' + str(idx + 1) + '.h5'),
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
        '--data_path=' + os.path.join(data_base, monkey),
        '--results_path=' + os.path.join(results_base, 'monkey' + str(idx + 1) + '.h5'),
        '--results_group=uoi_poisson_log',
        '--fitter=UoI_Poisson',
        '--n_folds=10',
        '--standardize',
        '--random_state=2332',
        '--verbose'
    ])
