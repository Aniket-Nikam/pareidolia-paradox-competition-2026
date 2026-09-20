"""Group-aware uncertainty, calibration metrics and sampled resource measurements."""
import os
import time
import threading
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,brier_score_loss,log_loss


class Monitor:
    def __enter__(self):
        import psutil
        self.psutil=psutil; self.process=psutil.Process(os.getpid())
        self.start=time.perf_counter(); self.rss=0; self.system=0; self.stop=threading.Event()
        def sample():
            while not self.stop.is_set():
                self.rss=max(self.rss,self.process.memory_info().rss)
                self.system=max(self.system,psutil.virtual_memory().used)
                self.stop.wait(.1)
        self.thread=threading.Thread(target=sample,daemon=True); self.thread.start()
        return self
    def __exit__(self,*args):
        self.stop.set(); self.thread.join()
        self.measurements={'seconds':time.perf_counter()-self.start,'peak_process_rss_bytes_sampled':self.rss,
                           'peak_system_used_ram_bytes_sampled':self.system,'sample_interval_seconds':.1,'peak_vram_bytes':0}


def summary(y,pred,probability=None):
    cm=confusion_matrix(y,pred,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
    result={'n':len(y),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'accuracy':float(accuracy_score(y,pred)),'recall_depth':float(tn/(tn+fp)) if tn+fp else None,
            'recall_rise':float(tp/(tp+fn)) if tp+fn else None,'confusion_matrix':cm.tolist(),
            'false_positive_rise':int(fp),'false_negative_rise':int(fn)}
    if probability is not None: result['calibration']=calibration(y,probability)
    return result


def calibration(y,p,bins=15):
    p=np.clip(np.asarray(p),1e-7,1-1e-7); y=np.asarray(y)
    index=np.minimum((p*bins).astype(int),bins-1); rows=[]; ece=0
    for i in range(bins):
        ix=index==i
        if not ix.any(): continue
        mean=float(p[ix].mean()); rate=float(y[ix].mean()); n=int(ix.sum())
        ece+=n/len(y)*abs(mean-rate)
        rows.append({'lower':i/bins,'upper':(i+1)/bins,'n':n,'mean_probability':mean,'observed_rise_fraction':rate})
    return {'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),
            'ece_15_equal_width':float(ece),'bins':rows}


def group_bootstrap(y,groups,predictions,resamples=10000,seed=2026):
    """Paired cluster bootstrap stratified by pure-depth/pure-rise/mixed label groups.

    All rows in a sampled duplicate group receive the same multiplicity. Numbers
    of groups per composition stratum stay fixed; class row totals may fluctuate.
    Intervals condition on already-fitted OOF models, not retraining/selection.
    """
    y=np.asarray(y); predictions=np.asarray(predictions)
    _,g=np.unique(groups,return_inverse=True); n=g.max()+1
    n0=np.bincount(g,weights=y==0,minlength=n); n1=np.bincount(g,weights=y==1,minlength=n)
    stats=np.column_stack([n0,n1]+[np.bincount(g,weights=(y==c)&(pred==c),minlength=n) for pred in predictions for c in [0,1]])
    strata=[np.flatnonzero((n0>0)&(n1==0)),np.flatnonzero((n1>0)&(n0==0)),np.flatnonzero((n0>0)&(n1>0))]
    rng=np.random.default_rng(seed); output=[]
    for start in range(0,resamples,100):
        batch=min(100,resamples-start); totals=np.zeros((batch,stats.shape[1]))
        for ix in strata:
            if len(ix):
                weights=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=batch)
                totals+=weights@stats[ix]
        output.append(np.column_stack([.5*(totals[:,2+2*j]/totals[:,0]+totals[:,3+2*j]/totals[:,1]) for j in range(len(predictions))]))
    return np.vstack(output)
