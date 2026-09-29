"""Re-score published one-step checkpoints without changing historical reports."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from traffic_forecasting.metr_la_study import Windows, starts_for_split, evaluate, model_class
from traffic_forecasting.multihorizon_data import atomic_json, sha256


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    root=Path(__file__).resolve().parents[1]; torch.set_num_threads(4)
    series=np.load(args.data_dir/'METR_LA.npz')['data'].astype(np.float32)
    graph=np.load(args.data_dir/'METR_LA_rn_adj.npy').astype(np.float32)
    train=series[:int(.7*len(series))]; valid=train[train>0]
    mean,std=float(valid.mean()),float(valid.std())
    starts=starts_for_split(int(.8*len(series)),len(series),1)
    loader=DataLoader(Windows(series,starts,mean,std),batch_size=16)
    report={'normalized_zero_observed_collisions':int(((series==mean)&(series>0)).sum()),'models':{}}
    for name in ['STGCN','DCRNN','STTN']:
        path=root/'artifacts/metr_la'/f'{name.lower()}.pt'
        ck=torch.load(path,map_location='cpu',weights_only=True)
        config={'input_window':12,'output_window':1,'device':torch.device('cpu'),**ck['config']}
        model=model_class(name)(config,{'num_nodes':207,'feature_dim':1,'output_dim':1,'adj_mx':graph})
        model.load_state_dict(ck['state_dict'])
        actual=evaluate(model,loader,mean,std)
        source=root/'docs/forecasting'/('metr_la_sttn_report.json' if name=='STTN' else 'metr_la_report.json')
        expected=json.loads(source.read_text())['models'][name]['test']
        report['models'][name]={'checkpoint_sha256':sha256(path),'actual':actual,'published':expected,
                              'mae_absolute_difference':abs(actual['mae_mph']-expected['mae_mph'])}
        atomic_json(args.output,report)
        print(name,actual,flush=True)


if __name__=='__main__': main()
