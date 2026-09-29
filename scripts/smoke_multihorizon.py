"""Check actual training data shapes, finite gradients, and tiny-batch learning."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from traffic_forecasting.multihorizon_data import MaskedWindows, load_development, atomic_json
from traffic_forecasting.multihorizon_study import build_model, predict, masked_loss

root=Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
values, graph, _, manifest=load_development(root.parent/'model_training/data','METR_LA')
data=MaskedWindows(values,0,96,manifest['mean'],manifest['std'])
batch=next(iter(DataLoader(data,batch_size=8)))
trials=json.loads((root/'configs/forecasting/multihorizon.json').read_text())['trials']
rows=[]
for trial in trials:
    torch.manual_seed(42)
    model=build_model(trial,graph)
    optimizer=torch.optim.Adam(model.parameters(),lr=trial['learning_rate'])
    model.train(); losses=[]; start=time.monotonic()
    for step in range(30):
        optimizer.zero_grad(set_to_none=True)
        output=predict(model,batch,True,step)
        loss=masked_loss(output,batch['y'],batch['y_mask'],manifest['std'])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
        optimizer.step();losses.append(loss.item())
    row={'trial':trial['id'],'initial_loss':losses[0],'final_loss':losses[-1],
         'steps':30,'seconds':time.monotonic()-start,'shape':list(output.shape)}
    if not np.isfinite(losses).all() or min(losses[-5:])>=losses[0]:
        raise RuntimeError(f'Tiny batch did not learn: {row}')
    rows.append(row); print(json.dumps(row),flush=True)
atomic_json(root.parent/'model_training/runs/audit/smoke_multihorizon.json',rows)
