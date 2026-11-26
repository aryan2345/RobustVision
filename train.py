"""CIFAR-10 robustness study: matched CNNs with and without corruption augmentation."""
import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        for a,b in [(3,32),(32,64),(64,128)]:
            layers += [nn.Conv2d(a,b,3,padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.Conv2d(b,b,3,padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2)]
        self.net=nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(128,10))
    def forward(self,x): return self.net(x)

def corrupt(x,kind,level,g):
    if kind=='noise': return (x+torch.randn(x.shape,generator=g)*level).clamp(0,1)
    if kind=='occlusion':
        x=x.clone(); n=max(1,round(x.shape[-1]*level)); h,w=x.shape[-2:]; i=int(torch.randint(h-n+1,(1,),generator=g)); j=int(torch.randint(w-n+1,(1,),generator=g)); x[:,i:i+n,j:j+n]=0.5
    return x

class View(Dataset):
    def __init__(self,base,indices,train=False,robust=False,kind='clean',level=0,seed=42):
        self.base=base; self.indices=indices; self.train=train; self.robust=robust; self.kind=kind; self.level=level; self.seed=seed
        self.aug=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip()]); self.tensor=transforms.ToTensor(); self.norm=transforms.Normalize((.4914,.4822,.4465),(.247,.243,.261))
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        x,y=self.base[self.indices[i]]; x=self.tensor(self.aug(x) if self.train else x)
        if self.train and self.robust and random.random()<.5:
            k=random.choice(['noise','occlusion']); x=corrupt(x,k,random.uniform(.05,.2) if k=='noise' else random.uniform(.15,.4),torch.default_generator)
        elif not self.train: x=corrupt(x,self.kind,self.level,torch.Generator().manual_seed(self.seed+i))
        return self.norm(x),y

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def evaluate(model,loader,device):
    model.eval(); ys=[]; ps=[]; loss=0
    with torch.no_grad():
        for x,y in loader:
            z=model(x.to(device)); loss+=nn.functional.cross_entropy(z,y.to(device),reduction='sum').item(); ys.extend(y.tolist()); ps.extend(z.argmax(1).cpu().tolist())
    return {'accuracy':float(np.mean(np.array(ys)==ps)), 'macro_f1':float(f1_score(ys,ps,labels=list(range(10)),average='macro',zero_division=0)), 'loss':loss/len(ys)},ys,ps

def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=15); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--seed',type=int,default=42); p.add_argument('--data',default='data'); p.add_argument('--output',default='results'); p.add_argument('--smoke',action='store_true',help='Synthetic data pipeline check only; not scientific results'); a=p.parse_args(); seed_all(a.seed); torch.set_num_threads(min(4,torch.get_num_threads())); device='cuda' if torch.cuda.is_available() else 'cpu'; out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    if a.smoke:
        train=datasets.FakeData(size=160,image_size=(3,32,32),num_classes=10); test=datasets.FakeData(size=40,image_size=(3,32,32),num_classes=10,random_offset=1000); a.epochs=1
    else:
        train=datasets.CIFAR10(a.data,train=True,download=True); test=datasets.CIFAR10(a.data,train=False,download=True)
    order=np.random.default_rng(a.seed).permutation(len(train)); n=int(.1*len(train)); vi=order[:n].tolist(); ti=order[n:].tolist(); ix=list(range(len(test)))
    def loader(v,shuffle=False): return DataLoader(v,batch_size=a.batch_size,shuffle=shuffle,num_workers=0,generator=torch.Generator().manual_seed(a.seed))
    val=loader(View(train,vi)); conditions=[('clean',0),('noise',.05),('noise',.1),('noise',.2),('noise',.3),('occlusion',.125),('occlusion',.25),('occlusion',.375),('occlusion',.5)]
    rows=[]; curves=[]
    for name,robust in [('baseline',False),('robust',True)]:
        seed_all(a.seed); model=CNN().to(device); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001); best=-1
        dl=loader(View(train,ti,train=True,robust=robust),True)
        for epoch in range(a.epochs):
            model.train(); total=0
            for x,y in dl:
                x,y=x.to(device),y.to(device); opt.zero_grad(); loss=nn.functional.cross_entropy(model(x),y); loss.backward(); opt.step(); total+=loss.item()*len(y)
            metrics,_,_=evaluate(model,val,device); curves.append({'model':name,'epoch':epoch+1,'train_loss':total/len(ti),'val_accuracy':metrics['accuracy']}); print(name,epoch+1,metrics,flush=True)
            if metrics['accuracy']>best: best=metrics['accuracy']; torch.save(model.state_dict(),out/f'{name}.pt')
        model.load_state_dict(torch.load(out/f'{name}.pt',map_location=device,weights_only=True))
        for kind,level in conditions:
            m,y,pred=evaluate(model,loader(View(test,ix,kind=kind,level=level,seed=a.seed)),device); rows.append({'model':name,'condition':kind,'severity':level,**m})
            if (kind,level) in [('clean',0),('occlusion',.5)]:
                cm=confusion_matrix(y,pred,labels=list(range(10))); np.savetxt(out/f'{name}_{kind}_confusion.csv',cm,fmt='%d',delimiter=','); fig,ax=plt.subplots(); im=ax.imshow(cm,cmap='Blues'); fig.colorbar(im); ax.set(xlabel='Predicted class ID',ylabel='True class ID',title=f'{name}: {kind}'); fig.savefig(out/f'{name}_{kind}_confusion.png',dpi=150); plt.close(fig)
    for filename,data in [('metrics.csv',rows),('training.csv',curves)]:
        with open(out/filename,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    for kind in ['noise','occlusion']:
        fig,ax=plt.subplots()
        for name in ['baseline','robust']:
            r=[r for r in rows if r['model']==name and r['condition'] in ['clean',kind]]; ax.plot([r['severity'] for r in r],[100*r['accuracy'] for r in r],marker='o',label=name)
        ax.set(xlabel='Noise standard deviation' if kind=='noise' else 'Occluded side / image side',ylabel='Test accuracy (%)',title=('Synthetic smoke test: ' if a.smoke else 'CIFAR-10: ')+kind); ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out/f'{kind}_robustness.png',dpi=150); plt.close(fig)
    (out/'run.json').write_text(json.dumps({**vars(a),'device':device,'train_size':len(ti),'validation_size':len(vi),'test_size':len(ix),'dataset':'synthetic FakeData' if a.smoke else 'CIFAR-10'},indent=2))
if __name__=='__main__': main()
