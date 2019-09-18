"""Run logistic regression fits on venral sensorimotor cortex data, classifing
vowels and consonants from high-gamma recordings."""
import os
import subprocess

est_scores = ['acc', 'log', 'BIC']
est_targets = ['test', 'test', 'train']
tasks = ['c', 'v']

# data and results paths
data_path = '/storage/data/vsmc/' + \
            'EC2_blocks_1_8_9_15_76_89_105_CV_AA_ff_align_window_-0.5_to_0.79_none_AA_avg.h5'
results_path = os.path.join(os.environ['HOME'],
                            'fits/uoineuro/classification',
                            'consonant_vowel_vsmc.h5')

# iterate over estimation options
for est_score, est_target in zip(est_scores, est_targets):
    print('Estimation Score: ', est_score)
    # iterate over consonant vs. vowel
    for task in tasks:
        print('Task: ', task)
        group = task + '_unshared_' + est_score + '_' + est_target
        subprocess.call([
            'python3', 'consonant_vowel_vsmc.py',
            '--data_path=' + data_path,
            '--results_path=' + results_path,
            '--results_group=' + group,
            '--task=' + task,
            '--n_folds=10',
            '--n_Cs=50',
            '--cv=5',
            '--estimation_score=' + est_score,
            '--estimation_target=' + est_target
        ])
