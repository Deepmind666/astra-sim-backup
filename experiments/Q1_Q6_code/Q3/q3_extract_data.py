#!/usr/bin/env python3
import json
import os

# 读取所有seed的数据
seeds = ['seed11', 'seed22', 'seed33']
all_data = {}

for seed in seeds:
    with open(f'{seed}/task_drift_summary.json', 'r', encoding='utf-8') as f:
        all_data[seed] = json.load(f)

# 输出关键指标
tasks = ['wiki', 'gsm8k', 'humaneval', 'cmrc2018', 'piqa', 'winogrande']

print('=' * 100)
print('Q3 TASK INTERNAL DRIFT ANALYSIS - COMPLETE DATA EXTRACTION')
print('=' * 100)

# 1. Switch Rate Analysis
print('\n' + '=' * 50)
print('1. SWITCH RATE ANALYSIS')
print('=' * 50)
header = f"{'Task':<12} | {'Switch Rate':<12} | {'CI 95%':<25} | {'Shuffle Base':<12} | {'Delta':<12}"
print(header)
print('-' * 90)

for task in tasks:
    switch_rates = []
    shuffle_rates = []
    ci_lows = []
    ci_highs = []
    for seed in seeds:
        task_data = all_data[seed]['tasks'].get(task, {})
        if 'switch_rate' in task_data:
            switch_rates.append(task_data['switch_rate'])
            shuffle_rates.append(task_data['switch_rate_shuffle'])
            ci_lows.append(task_data['switch_rate_ci_95'][0])
            ci_highs.append(task_data['switch_rate_ci_95'][1])

    if switch_rates:
        avg_switch = sum(switch_rates) / len(switch_rates)
        avg_shuffle = sum(shuffle_rates) / len(shuffle_rates)
        avg_ci_low = sum(ci_lows) / len(ci_lows)
        avg_ci_high = sum(ci_highs) / len(ci_highs)
        delta = avg_switch - avg_shuffle
        print(f'{task:<12} | {avg_switch:.4f}       | [{avg_ci_low:.4f}, {avg_ci_high:.4f}]     | {avg_shuffle:.4f}       | {delta:+.4f}')

# 2. Run Length Analysis
print('\n' + '=' * 50)
print('2. RUN LENGTH ANALYSIS')
print('=' * 50)
header = f"{'Task':<12} | {'Run Mean':<10} | {'Run Median':<10} | {'Run P90':<10} | {'Geom Mean':<10} | {'Delta':<12}"
print(header)
print('-' * 90)

for task in tasks:
    run_means = []
    run_medians = []
    run_p90s = []
    geom_means = []
    for seed in seeds:
        task_data = all_data[seed]['tasks'].get(task, {})
        if 'run_mean' in task_data:
            run_means.append(task_data['run_mean'])
            run_medians.append(task_data['run_median'])
            run_p90s.append(task_data['run_p90'])
            geom_means.append(task_data['geom_run_mean'])

    if run_means:
        avg_run = sum(run_means) / len(run_means)
        avg_median = sum(run_medians) / len(run_medians)
        avg_p90 = sum(run_p90s) / len(run_p90s)
        avg_geom = sum(geom_means) / len(geom_means)
        delta = avg_run - avg_geom
        print(f'{task:<12} | {avg_run:.4f}     | {avg_median:.1f}        | {avg_p90:.1f}        | {avg_geom:.4f}     | {delta:+.4f}')

# 3. Window Drift Analysis
print('\n' + '=' * 50)
print('3. WINDOW DRIFT ANALYSIS (JSD)')
print('=' * 50)
header = f"{'Task':<12} | {'JSD Mean':<12} | {'CI 95%':<25} | {'JSD Shuffle':<12} | {'Delta':<12}"
print(header)
print('-' * 90)

for task in tasks:
    jsd_means = []
    jsd_shuffles = []
    ci_lows = []
    ci_highs = []
    for seed in seeds:
        task_data = all_data[seed]['tasks'].get(task, {})
        if 'window_jsd_mean' in task_data:
            jsd_means.append(task_data['window_jsd_mean'])
            jsd_shuffles.append(task_data['window_jsd_mean_shuffle'])
            ci_lows.append(task_data['window_jsd_ci_95'][0])
            ci_highs.append(task_data['window_jsd_ci_95'][1])

    if jsd_means:
        avg_jsd = sum(jsd_means) / len(jsd_means)
        avg_shuffle = sum(jsd_shuffles) / len(jsd_shuffles)
        avg_ci_low = sum(ci_lows) / len(ci_lows)
        avg_ci_high = sum(ci_highs) / len(ci_highs)
        delta = avg_jsd - avg_shuffle
        print(f'{task:<12} | {avg_jsd:.6f}     | [{avg_ci_low:.6f}, {avg_ci_high:.6f}] | {avg_shuffle:.6f}     | {delta:+.6f}')

# 4. Window Size Sensitivity
print('\n' + '=' * 50)
print('4. WINDOW SIZE SENSITIVITY (JSD)')
print('=' * 50)
header = f"{'Task':<12} | {'W=64':<12} | {'W=128':<12} | {'W=256':<12}"
print(header)
print('-' * 60)

for task in tasks:
    w64_vals = []
    w128_vals = []
    w256_vals = []
    for seed in seeds:
        task_data = all_data[seed]['tasks'].get(task, {})
        ws = task_data.get('window_stats', {})
        if '64' in ws:
            w64_vals.append(ws['64']['window_jsd_mean'])
        if '128' in ws:
            w128_vals.append(ws['128']['window_jsd_mean'])
        if '256' in ws:
            w256_vals.append(ws['256']['window_jsd_mean'])

    if w64_vals:
        avg_64 = sum(w64_vals) / len(w64_vals)
        avg_128 = sum(w128_vals) / len(w128_vals) if w128_vals else 0
        avg_256 = sum(w256_vals) / len(w256_vals) if w256_vals else 0
        print(f'{task:<12} | {avg_64:.6f}     | {avg_128:.6f}     | {avg_256:.6f}')

# 5. Boundary Effect
print('\n' + '=' * 50)
print('5. BOUNDARY EFFECT ANALYSIS')
print('=' * 50)
header = f"{'Task':<12} | {'#Boundaries':<12} | {'Bound JSD':<12} | {'Random JSD':<12} | {'Ratio':<10}"
print(header)
print('-' * 70)

for task in tasks:
    n_bounds = []
    bound_jsds = []
    random_jsds = []
    for seed in seeds:
        task_data = all_data[seed]['tasks'].get(task, {})
        if 'n_boundaries' in task_data:
            n_bounds.append(task_data['n_boundaries'])
            if task_data.get('boundary_jsd_mean') is not None:
                bound_jsds.append(task_data['boundary_jsd_mean'])
            if task_data.get('random_jsd_mean') is not None:
                random_jsds.append(task_data['random_jsd_mean'])

    if n_bounds:
        avg_n = sum(n_bounds) / len(n_bounds)
        avg_bound = sum(bound_jsds) / len(bound_jsds) if bound_jsds else 0
        avg_random = sum(random_jsds) / len(random_jsds) if random_jsds else 0
        ratio = avg_bound / avg_random if avg_random > 0 else 0
        print(f'{task:<12} | {avg_n:.1f}          | {avg_bound:.6f}     | {avg_random:.6f}     | {ratio:.2f}x')

# 6. Sample Size Check
print('\n' + '=' * 50)
print('6. SAMPLE SIZE & DATA QUALITY')
print('=' * 50)
header = f"{'Task':<12} | {'#Tokens':<10} | {'#Segments':<10} | {'Avg Seg Len':<12} | {'Warning':<20}"
print(header)
print('-' * 80)

for task in tasks:
    task_data = all_data['seed11']['tasks'].get(task, {})
    if 'n_tokens' in task_data:
        n_tokens = task_data['n_tokens']
        n_segments = task_data['n_segments']
        avg_seg = task_data['avg_segment_len']
        warning = task_data.get('short_task_warning', 'None')
        if warning is None:
            warning = 'None'
        print(f'{task:<12} | {n_tokens:<10} | {n_segments:<10} | {avg_seg:<12.1f} | {warning[:20]}')

# 7. KS Test Results
print('\n' + '=' * 50)
print('7. KS TEST FOR GEOMETRIC DISTRIBUTION')
print('=' * 50)
header = f"{'Task':<12} | {'KS Stat':<12} | {'p-value':<15} | {'Reject Null?':<12}"
print(header)
print('-' * 60)

for task in tasks:
    task_data = all_data['seed11']['tasks'].get(task, {})
    ks = task_data.get('ks_test_geometric', {})
    if ks:
        ks_stat = ks.get('ks_statistic', 0)
        p_val = ks.get('p_value', 1)
        reject = ks.get('reject_null', False)
        p_str = f'{p_val:.2e}' if p_val < 0.01 else f'{p_val:.4f}'
        print(f'{task:<12} | {ks_stat:.6f}     | {p_str:<15} | {str(reject):<12}')

# 8. TV Distance Analysis
print('\n' + '=' * 50)
print('8. WINDOW DRIFT (TV DISTANCE)')
print('=' * 50)
header = f"{'Task':<12} | {'TV Mean':<12} | {'TV Shuffle':<12} | {'Delta':<12}"
print(header)
print('-' * 60)

for task in tasks:
    tv_means = []
    tv_shuffles = []
    for seed in seeds:
        task_data = all_data[seed]['tasks'].get(task, {})
        if 'window_tv_mean' in task_data:
            tv_means.append(task_data['window_tv_mean'])
            tv_shuffles.append(task_data['window_tv_mean_shuffle'])

    if tv_means:
        avg_tv = sum(tv_means) / len(tv_means)
        avg_shuffle = sum(tv_shuffles) / len(tv_shuffles)
        delta = avg_tv - avg_shuffle
        print(f'{task:<12} | {avg_tv:.6f}     | {avg_shuffle:.6f}     | {delta:+.6f}')

print('\n' + '=' * 100)
print('DATA EXTRACTION COMPLETE')
print('=' * 100)
