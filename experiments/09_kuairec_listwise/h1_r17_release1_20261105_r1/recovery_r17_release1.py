"""Continuous R17 transition: preserve effective correction, accept new delta at 1."""
import copy
import types
from pathlib import Path
import torch
from .recovery import RecoveryController, fingerprint
from .recovery_r14_g01 import G01Controller as ParentController
from .run import build_context
from .training import cpu_copy, sha256

GAMMA=1.0
T_BLOCK=40
R_BLOCK=3
NAME='H1-R17-Release1-R20'
PARENT_SHA256='aa565c3bf6daa2809ff1d30230d07aa57468a1286da1fe3eeb177a83380228ea'

class G01Controller(ParentController):
    def __init__(self,trainer,parent_binding,execution_binding,digest):
        RecoveryController.__init__(self,trainer,'decay',parent_binding,execution_binding,digest,start_phase=21,stop_phase=25)
        self.policy='release1'
        self.protocol.update(policy=self.policy,name=NAME,gamma=1.0,t_block=40,r_block=3,
                             t_lr='original_unchanged',mixer_gate='B_t+effective_M_R17+(raw_M_t-raw_M_R17)')
        self.release_anchor=None

    def checkpoint(self):
        state=RecoveryController.checkpoint(self)
        if self.release_anchor is not None:
            state['release1']=cpu_copy(self.release_anchor)
        return state

    def restore(self,state,**kwargs):
        if fingerprint(state.get('release1'))!=fingerprint(self.release_anchor):
            raise ValueError('Release anchor mismatch')
        RecoveryController.restore(self,state,**kwargs)

    def fork_release(self,parent,exact_rng=True):
        inner=copy.deepcopy(parent['inner'])
        assert (inner['r_updates'],inner['t_updates'])==(11781,160)
        assert inner['progress']==dict(phase=20,update=693,boundary_prepared=True)
        assert parent['g01']['gamma']==0.1
        inner['progress']=dict(phase=21,update=0,boundary_prepared=False)
        inner['arm']='H1'
        RecoveryController.fork(self,inner,exact_rng=exact_rng)
        mixer=self.trainer.alignment.mixer
        original=mixer.crossed_codebooks
        with torch.no_grad():
            raw,_=original(False)
            base=mixer.base_codebooks
            m0=parent['g01']['anchor']['M0'].to(raw)
            effective=base+m0+0.1*(raw-base-m0)
            self.release_anchor=dict(raw_M=cpu_copy(raw-base),effective_M=cpu_copy(effective-base),
                                     C=cpu_copy(effective),parent_sha256=PARENT_SHA256)
        anchor=self.release_anchor
        def gated(module,return_attention=False):
            raw,weights=original(return_attention)
            base=module.base_codebooks
            result=base+anchor['effective_M'].to(raw)+(raw-base-anchor['raw_M'].to(raw))
            return result,weights if return_attention else None
        mixer.crossed_codebooks=types.MethodType(gated,mixer)
        mixer._release_original=original
        mixer._release_anchor=anchor
        return self.checkpoint()

def build_g01(parent_path,device,dataset=None,assets=None,inspection=False):
    assert sha256(parent_path)==PARENT_SHA256
    parent=torch.load(parent_path,map_location='cpu',weights_only=False)
    inner=parent['inner']
    config=copy.deepcopy(inner['binding']['config'])
    if dataset: config['dataset']=str(dataset)
    if assets: config['assets']=str(assets)
    trainer,loaded,data,current=build_context(config,device,inner['precision'],True)
    ctrl=G01Controller(trainer,inner['binding'],current,PARENT_SHA256)
    ctrl.fork_release(parent,exact_rng=not inspection)
    return ctrl,loaded,data

def gate_metrics(trainer):
    mixer=trainer.alignment.mixer
    with torch.no_grad():
        raw,_=mixer._release_original(False)
        eff,_=mixer.crossed_codebooks(False)
        base=mixer.base_codebooks
        anchor=mixer._release_anchor
        return dict(gamma=1.0,raw_mixer_norm=float((raw-base).norm()),
                    raw_increment_norm=float((raw-base-anchor['raw_M'].to(raw)).norm()),
                    effective_mixer_increment_norm=float((eff-base-anchor['effective_M'].to(raw)).norm()),
                    effective_codebook_displacement_from_R17_norm=float((eff-anchor['C'].to(raw)).norm()),
                    effective_codebook_norm=float(eff.norm()))
