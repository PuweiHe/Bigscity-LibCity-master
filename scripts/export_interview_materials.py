"""Build the user's local interview evidence folder without copying source data."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, default=ROOT.parent / 'interview_materials')
    args = parser.parse_args(); dest = args.destination
    dest.mkdir(parents=True, exist_ok=True)
    # Include all model dependencies and new scripts; exclude unrelated raw datasets.
    include = ['traffic_forecasting', 'libcity', 'tests', 'scripts', 'configs',
               'docs/forecasting', 'artifacts', 'portfolio', 'requirements', 'readme.md', 'LICENSE']
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
    records = []
    def copy(source, target):
        if not source.is_file() or source.suffix in {'.pyc', '.tmp'}: return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append({'file': str(target.relative_to(dest)), 'source': str(source),
                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bytes': target.stat().st_size})
    for name in sorted(set(names)):
        if any(name == p or name.startswith(p + '/') or (p == 'requirements' and name.startswith(p)) for p in include):
            copy(ROOT / name, dest / 'repository_snapshot' / name)
    training = ROOT.parent / 'model_training'
    for name in ['README.md', 'environment.json', 'run_metr_la.sh', 'run_sttn.sh', 'run_multihorizon.sh', 'plot_learning_curves.py', 'start_multihorizon.py', 'CONTINUATION.md']:
        copy(training / name, dest / 'training' / name)
    for folder in ['logs', 'plots', 'runs']:
        for path in sorted((training / folder).rglob('*')):
            if path.name.startswith('.') or path.suffix in {'.tmp', '.lock'}: continue
            copy(path, dest / 'training' / path.relative_to(training))
    for dataset in ['PEMS_BAY']:
        copy(training / 'data' / dataset / 'source_manifest.json', dest / 'data_provenance' / f'{dataset}.json')
    source = ROOT.parent.parent / 'PuweiHe resume.tex'
    tex = source.read_text()
    begin = tex.index('\\resumeexperience\n    {Institute of Automation, Chinese Academy of Sciences}')
    end = tex.index('\\end{itemize}', begin) + len('\\end{itemize}')
    (dest / 'resume').mkdir(exist_ok=True)
    (dest / 'resume' / 'CASIA_section.tex').write_text(tex[begin:end] + '\n')
    subprocess.run(['git', 'diff', '--binary', '348e4ea'], cwd=ROOT, stdout=(dest / 'changes_since_published_sttn.patch').open('w'), check=True)
    (dest / 'MANIFEST.json').write_text(json.dumps({
        'repository_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'working_tree_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        'raw_data_included': False, 'files': records}, indent=2) + '\n')
    print(f'Exported {len(records)} evidence/source files to {dest}')


if __name__ == '__main__': main()
