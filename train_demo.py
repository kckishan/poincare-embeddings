import torch
import numpy as np
from time import time
import json

from datasets import WordNetDataset
from poincare import PoincareDistance, PoincareEmbedding, RiemannianSGD
from vis import generate_report

with open('demo/config.json','r') as f:
    config = json.loads(f.read())

print('Training demo model with config:')

print(json.dumps(config, indent=4, sort_keys=True))

print()

print("Construction dataset...")
data = WordNetDataset(filename=config['data'])
dataloader = torch.utils.data.DataLoader(data,batch_size=config['batch_size'])

print()

print("total entries:",data.N)
print('total unique items',len(data.items))
print(data.neg_samples,'negative samples per one positive')

torch.save(data,'demo/data.pt')

print()

print('Training...')

model = PoincareEmbedding(data.n_items)

model.initialize_embedding()

optimizer = RiemannianSGD(model.parameters())

total_time = 0
for epoch in range(config['n_epochs']):
    epoch_loss = []
    start = time()
    
    if epoch<config['n_burn_in']:
        lr = config['lr']/config['c']
    else:
        lr = config['lr']
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        x,y = batch
        
        preds = model(x,y)
        loss = model.loss(preds)
        
        loss.backward()
        
        optimizer.step(lr=lr)
        
        epoch_loss.append(loss.data.item())
        
        
    time_per_epoch = time()-start
    total_time += time_per_epoch

    model.log.append(np.mean(epoch_loss))

    estimated_time = (total_time/(epoch+1))*(config['n_epochs']-epoch-1)

    minutes_left = int(estimated_time/60.)

    seconds_left = int(estimated_time-60*minutes_left)

    print('Epoch',epoch+1,'/',config['n_epochs'],'|',
         'loss:',"%.4f" % model.log[-1],'|',
         "time per epoch:","%.2f" % time_per_epoch,'sec.','|',
         'estimated training time:',minutes_left,'min.',seconds_left,'sec.',
         end='\r')

print('')
print('Trainig finished!')
torch.save(model,'demo/model.pt')

generate_report(model,data)