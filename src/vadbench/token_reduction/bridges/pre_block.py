"""Single-site pre-block reduction for audited HF VideoMAE/TimeSformer.

Signals come from the previous complete block in the same forward. Native
attention/backends are untouched. TimeSformer keeps whole time trajectories;
its suffix only replaces H*W reshape factorization by explicit logical S*1.
This is slot packing, never a new physical image grid or position encoding.
"""
from __future__ import annotations
import ast
import hashlib
import inspect
import textwrap
from types import MethodType
import torch
import torch.nn.functional as F
from ..relation_selection import METHODS,select_relational_tokens


_CLASSIC_VMA_ATTENTION='0a61132635324fd6c3eca69f8cfecd5c739eaa6e88f2714d564b0abe00a6a225'
_BLOCKS={'videomae':{'010fae3fb1e62c0020bc179f4275f0e95237a8e5ab0154f4c0b1ac8f9706f0cb',
                     '073012c994d6a23697593ef01218a25ccafa6345c62c88221dde1af59599dfd6'},
         'timesformer':{'e79f48b3db7efd29f6c116b4ce6f94cc64a3443ad01f9f25c3b84ee1b060acbf'}}
_ATTENTION={'videomae':{'4e7d12354b5cf6951a27b998de6902734cb2127a04a514c5978839ba392c20c4',
                        _CLASSIC_VMA_ATTENTION},
            'timesformer':{'d3964ecc282f3f2181526aaf069937102ccebb8a5f3b92b6b68b81b6030d8dc9'}}


def _tree(method):
    return ast.parse(textwrap.dedent(inspect.getsource(method)))


def _identity(method):
    node=_tree(method).body[0]
    for child in ast.walk(node):
        if isinstance(child,(ast.FunctionDef,ast.ClassDef)) and getattr(child,'type_params',None)==[]:
            child._fields=tuple(f for f in child._fields if f!='type_params')
    return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()


def _logical_spatial_forward(block):
    """Only alter the audited reshape factor; preserve all native operations."""
    method=type(block).forward;tree=_tree(method)
    matches=[n for n in tree.body[0].body if isinstance(n,ast.Assign) and len(n.targets)==1
             and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='num_patch_width']
    if len(matches)!=1:raise ValueError('native spatial packing assignment unavailable')
    matches[0].value=ast.Constant(value=1)
    ast.fix_missing_locations(tree);namespace=dict(method.__globals__)
    exec(compile(tree,'<audited-timesformer-logical-spatial-packing>','exec'),namespace)
    return namespace['forward']


class PreBlockIntervention:
    """Attach once per run, automatically clear all signal state per forward.

    ``target_depth`` is zero-based block input; signal source is depth-1.
    TS budgets refer to spatial trajectories, not arbitrary flat tokens.
    """
    def __init__(self,bridge,target_depth,method,keep_ratio):
        self.bridge=bridge;self.encoder=bridge.receipt().encoder_id
        if self.encoder not in _BLOCKS:raise ValueError('pre-block supports VideoMAE/TimeSformer only')
        if method not in METHODS or not 0<keep_ratio<=1:raise ValueError('invalid method/budget')
        self.model=bridge._root;self.blocks=bridge._blocks
        if type(target_depth) is not int or not 1<=target_depth<len(self.blocks):raise ValueError('target block needs a previous signal block')
        self.target=target_depth;self.source=target_depth-1;self.method=method;self.keep_ratio=keep_ratio
        for block in self.blocks:
            if 'forward' in block.__dict__ or _identity(type(block).forward) not in _BLOCKS[self.encoder]:raise ValueError('unsupported native block forward')
        self.attention=self.blocks[self.source].attention.attention
        attention_identity=_identity(type(self.attention).forward)
        if attention_identity not in _ATTENTION[self.encoder]:raise ValueError('unsupported source attention')
        self.classic_vma=attention_identity==_CLASSIC_VMA_ATTENTION
        self.frames=int(self.model.config.num_frames) if self.encoder=='timesformer' else None
        if self.encoder=='timesformer':
            if any(b.attention_type!='divided_space_time' for b in self.blocks):raise ValueError('TimeSformer requires audited divided_space_time')
            self.logical_forward=_logical_spatial_forward(self.blocks[self.target])
        self.handles=[];self.overridden=[];self.active=False;self.forward_count=0
        self._clear()

    def _clear(self):
        self.keys=self.scores=self.selection=None
        self.inputs={};self.outputs={};self.mlp_inputs={};self.before_shape=self.after_shape=None
        self.completed=False

    def __enter__(self):
        if self.active or hasattr(self.model,'_vadbench_pre_block'):raise ValueError('nested pre-block intervention')
        if any(m.training for m in self.model.modules()):raise ValueError('pre-block is eval-only')
        for block in self.blocks:
            if 'forward' in block.__dict__ or block._forward_hooks or block._forward_pre_hooks:raise ValueError('concurrent block intervention')
        self.active=True;self.model._vadbench_pre_block=self
        try:
            self.handles.append(self.model.register_forward_pre_hook(self._begin))
            self.handles.append(self.model.register_forward_hook(self._end))
            # Reduce after full source block, then record the returned shape.
            self.handles.append(self.blocks[self.source].register_forward_hook(self._reduce))
            for i,b in enumerate(self.blocks):
                self.handles.append(b.register_forward_pre_hook(self._input(i)))
                self.handles.append(b.register_forward_hook(self._output(i)))
                self.handles.append(b.layernorm_after.register_forward_pre_hook(self._mlp(i)))
            if self.encoder=='timesformer':
                for b in self.blocks[self.target:]:
                    b.forward=MethodType(self.logical_forward,b);self.overridden.append(b)
            if self.method!='uniform' and self.keep_ratio!=1:
                if self.encoder=='videomae':
                    if self.classic_vma:
                        self.handles.append(self.attention.dropout.register_forward_pre_hook(self._observe_classic_vma_scores))
                        if self.method=='visionzip_internal':self.handles.append(self.attention.register_forward_pre_hook(self._observe_classic_vma_keys))
                    else:self.handles.append(self.attention.register_forward_pre_hook(self._observe_vma))
                else:
                    if self.method=='visionzip_internal':self.handles.append(self.attention.qkv.register_forward_hook(self._observe_ts_keys))
                    self.handles.append(self.attention.attn_drop.register_forward_pre_hook(self._observe_ts_scores))
        except Exception:
            self.close();raise
        return self

    def _begin(self,module,inputs):
        self._clear();self.forward_count+=1

    def _end(self,module,inputs,output):
        self.completed=True;self.keys=self.scores=None

    def _input(self,i):
        def hook(module,inputs):self.inputs[i]=list(inputs[0].shape)
        return hook

    def _output(self,i):
        def hook(module,inputs,output):self.outputs[i]=list((output[0] if isinstance(output,tuple) else output).shape)
        return hook

    def _mlp(self,i):
        def hook(module,inputs):self.mlp_inputs[i]=list(inputs[0].shape)
        return hook

    def _observe_vma(self,attention,inputs):
        x=inputs[0];heads=attention.num_attention_heads
        kb=torch.zeros_like(attention.v_bias) if attention.q_bias is not None else None
        q=F.linear(x,attention.query.weight,attention.q_bias)
        k=F.linear(x,attention.key.weight,kb)
        q=q.reshape(x.shape[0],x.shape[1],heads,-1).transpose(1,2)
        k=k.reshape(x.shape[0],x.shape[1],heads,-1).transpose(1,2)
        self.scores=((q*attention.scaling)@k.transpose(-2,-1)).softmax(-1).mean(dim=(1,2))
        if self.method=='visionzip_internal':self.keys=k

    def _observe_classic_vma_scores(self,module,inputs):
        # Classic native eager attention exposes its actual softmax directly.
        self.scores=inputs[0].mean(dim=(1,2))

    def _observe_classic_vma_keys(self,attention,inputs):
        x=inputs[0]
        bias=torch.zeros_like(attention.v_bias) if attention.q_bias is not None else None
        self.keys=attention.transpose_for_scores(F.linear(x,attention.key.weight,bias))

    def _observe_ts_keys(self,module,inputs,output):
        bt,n,three_width=output.shape;heads=self.attention.num_heads
        keys=output.reshape(bt,n,3,heads,three_width//(3*heads))[:,:,1].permute(0,2,1,3)[:,:,1:]
        self.keys=keys.reshape(bt//self.frames,self.frames,heads,n-1,-1).mean(dim=1)

    def _observe_ts_scores(self,module,inputs):
        patch=inputs[0][:,:,1:,1:]
        # Exclude CLS keys from selector normalization as well as queries.
        patch=patch/patch.sum(dim=-1,keepdim=True)
        received=patch.mean(dim=(1,2));self.scores=received.reshape(-1,self.frames,received.shape[-1]).mean(dim=1)

    def _reduce(self,module,inputs,output):
        hidden=output[0] if isinstance(output,tuple) else output
        self.before_shape=list(hidden.shape)
        if self.keep_ratio==1:reduced=hidden
        else:
            if self.method!='uniform' and self.scores is None:raise ValueError('same-forward previous attention signal missing')
            if self.method=='visionzip_internal' and self.keys is None:raise ValueError('same-forward native keys missing')
            if self.encoder=='videomae':patches=hidden
            else:
                if (hidden.shape[1]-1)%self.frames:raise ValueError('incomplete spatial time trajectories')
                patches=hidden[:,1:].reshape(hidden.shape[0],-1,self.frames*hidden.shape[2])
            selected=select_relational_tokens(patches,method=self.method,keep_ratio=self.keep_ratio,keys=self.keys,attention_scores=self.scores)
            self.selection=selected
            if self.encoder=='videomae':reduced=selected.tokens
            else:reduced=torch.cat((hidden[:,:1],selected.tokens.reshape(hidden.shape[0],-1,hidden.shape[2])),dim=1)
        self.after_shape=list(reduced.shape);self.keys=self.scores=None
        return (reduced,*output[1:]) if isinstance(output,tuple) else reduced

    def receipt(self):
        indices=set(range(len(self.blocks)))
        if not self.completed or set(self.inputs)!=indices or set(self.outputs)!=indices or set(self.mlp_inputs)!=indices:raise ValueError('incomplete native model execution')
        before,after=self.before_shape[1],self.after_shape[1]
        for i in indices:
            if self.inputs[i][1]!=(before if i<self.target else after) or self.mlp_inputs[i][1]!=(before if i<self.target else after) or self.outputs[i][1]!=(before if i<self.source else after):raise ValueError('actual native sequence lengths violate pre-block contract')
        return {'encoder':self.encoder,'site':'previous_complete_block_output_before_target_LN_attention','target_block':self.target+1,'signal_block':self.source+1,
                'method':self.method,'keep_ratio':self.keep_ratio,'before_tokens':before,'after_tokens':after,'inputs':self.inputs,'outputs':self.outputs,'mlp_inputs':self.mlp_inputs,
                'budget_unit':'spatial_trajectory' if self.frames else 'tubelet_patch','time_frames_preserved':self.frames,
                'spatial_packing':'logical_Sx1_not_physical_image_grid' if self.frames else None,'native_attention_backend_unchanged':True,'forward_count':self.forward_count}

    def close(self):
        for h in self.handles:h.remove()
        self.handles.clear()
        for block in self.overridden:del block.forward
        self.overridden.clear()
        if getattr(self.model,'_vadbench_pre_block',None) is self:delattr(self.model,'_vadbench_pre_block')
        self.active=False;self.keys=self.scores=self.selection=None

    def __exit__(self,*args):self.close()
