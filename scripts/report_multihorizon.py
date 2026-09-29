"""Summarize finished preregistered runs without selecting on test scores."""
import argparse
import json
from pathlib import Path
import numpy as np
from traffic_forecasting.multihorizon_data import atomic_json


def summarize(report):
    results={}
    for trial in report['mean_validation_mae']:
        rows=[r for r in report['runs'] if r['trial']==trial]
        scores={}
        for horizon in ['15','30','60','overall']:
            for metric in ['mae_mph','rmse_mph']:
                values=[(r['test']['all']['overall'] if horizon=='overall' else r['test']['all']['horizons'][horizon])[metric] for r in rows]
                scores[f'{horizon}_{metric}']={'mean':float(np.mean(values)),'seed_sample_std':float(np.std(values,ddof=1))}
        results[trial]={'seeds':[r['seed'] for r in rows], 'scores':scores,
                        'parameters':rows[0]['parameters'],
                        'mean_training_seconds':float(np.mean([r['training_seconds'] for r in rows])),
                        'mean_batch1_median_ms':float(np.mean([r['inference']['1']['median_ms'] for r in rows]))}
    pairs={}
    for original,modified in [('stgcn_reference','stgcn_reference_residual'),
                              ('stgcn_hyperparameters','stgcn_hyperparameters_residual'),
                              ('stgcn_reference','stgcn_hyperparameters'),
                              ('stgcn_reference_residual','stgcn_hyperparameters_residual'),
                              ('stgcn_recursive_reference','stgcn_reference')]:
        base={r['seed']:r for r in report['runs'] if r['trial']==original}
        changed={r['seed']:r for r in report['runs'] if r['trial']==modified}
        differences=[base[s]['test']['all']['overall']['mae_mph']-changed[s]['test']['all']['overall']['mae_mph'] for s in base]
        mean_base=np.mean([base[s]['test']['all']['overall']['mae_mph'] for s in base])
        pairs[f'{original} -> {modified}']={'paired_mae_reduction_mph_mean':float(np.mean(differences)),
                    'paired_mae_reduction_mph_seed_std':float(np.std(differences,ddof=1)),
                    'relative_reduction_percent':float(np.mean(differences)/mean_base*100)}
    return {'dataset':report['dataset'],'selected_by_validation':report['selected_by_mean_validation'],
            'trials':results,'paired_ablations':pairs,'baseline_metrics':report['baselines']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not (a.runs/'evaluation_complete.json').exists():raise SystemExit('Final evaluation has not completed')
    a.output.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for path in sorted(a.runs.glob('*/test_report.json')):
        report=json.loads(path.read_text());result=summarize(report);summaries.append(result)
        atomic_json(a.output/f'{result["dataset"]}_summary.json',result)
    lines=['# Measured multi-horizon results','', 'Selected models were determined by validation before final test scoring.','']
    for result in summaries:
        lines += [f'## {result["dataset"]}', '',f'Validation selection: `{result["selected_by_validation"]}`','',
                  '| Model | 15-min MAE | 30-min MAE | 60-min MAE | Parameters |','|---|---:|---:|---:|---:|']
        for trial,row in result['trials'].items():
            fields=[f'{row["scores"][h+"_mae_mph"]["mean"]:.3f} ± {row["scores"][h+"_mae_mph"]["seed_sample_std"]:.3f}' for h in ['15','30','60']]
            lines.append('| '+ ' | '.join([trial,*fields,str(row['parameters'])])+' |')
        lines += ['', 'Variation is sample standard deviation across training seeds, not uncertainty across roads.', '']
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
