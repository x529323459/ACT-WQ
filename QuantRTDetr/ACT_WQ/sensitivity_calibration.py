"""Fisher-inspired elementwise reconstruction-gradient sensitivity calibration.

Production extraction of the verified original-basis S algorithm; not strict Fisher.
No historical artifacts or experiment runners are required.
"""
import copy
import time
from contextlib import contextmanager
from types import MethodType
import torch
from ACT_WQ import act_calibration as cal
import hashlib
import json
import gc
from pathlib import Path
from itertools import islice

class _ExactSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, quantized):
        # Same derivative as x+(Q(x)-x).detach(), without cancellation error.
        return quantized

    @staticmethod
    def backward(ctx, grad):
        return grad, None


@contextmanager
def activation_ste(model):
    states = []
    inplace = cal._patch_inplace_acts(model, enable=False)
    try:
        for layer in cal._get_gptq_layers(model).values():
            aq = layer.a_quantizer
            if not aq.ready():
                continue
            had = 'quantize' in aq.__dict__
            previous = aq.__dict__.get('quantize')
            normal = aq.quantize
            def ste(self, x, _normal=normal):
                with torch.no_grad():
                    q = _normal(x)
                return _ExactSTE.apply(x, q)
            states.append((aq, had, previous))
            aq.quantize = MethodType(ste, aq)
        yield
    finally:
        for aq, had, previous in states:
            if had:
                aq.quantize = previous
            else:
                del aq.quantize
        cal._restore_inplace_acts(inplace)


def check_structure(a, b, path='output'):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.shape == b.shape, path
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for k in a:
            check_structure(a[k], b[k], path+'.'+str(k))
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (aa, bb) in enumerate(zip(a, b)):
            check_structure(aa, bb, path+'.'+str(i))
    else:
        assert type(a) is type(b) and a == b, path



FIELDS=('clip','scale','zero','clip_min')

def semantic_kind(name):
    if name.startswith(('backbone.','encoder.','decoder.input_proj.')):
        return 'fixed_spatial_or_encoder_token'
    if name.endswith('.cross_attn.value_proj'):
        return 'fixed_multiscale_memory_token'
    return 'query_identity_unproven_layer_fallback'


@contextmanager
def record_original_indices(records, context):
    """Observe original RNG RETURN values; never generate replacement indices."""
    raw_fw=cal.TensorHookMgr._fw
    def observed_fw(manager,name):
        raw_hook=raw_fw(manager,name)
        def hook(mod,args,output):
            call=context['calls'].get(name,0);context['calls'][name]=call+1
            x=args[0];draws=[];channels=[]
            raw_perm,raw_int=torch.randperm,torch.randint
            def perm(*a,**kw):
                v=raw_perm(*a,**kw);channels.append(v.detach().cpu());return v
            def randint(*a,**kw):
                v=raw_int(*a,**kw);draws.append(v.detach().cpu());return v
            try:
                torch.randperm,torch.randint=perm,randint
                raw_hook(mod,args,output)
            finally:torch.randperm,torch.randint=raw_perm,raw_int
            if not channels:return
            axis=cal._get_ch_axis(mod,x);C=x.shape[axis]
            ch=channels[0][:len(draws)];assert len(ch)==len(draws)
            selected=sorted(ch.tolist());lookup={c:i for i,c in enumerate(selected)}
            ordered=torch.empty((len(selected),len(draws[0])),dtype=torch.int64)
            for c,idx in zip(ch.tolist(),draws):ordered[lookup[c]]=idx
            flat=x.detach().movedim(axis,0).reshape(C,-1)
            values=flat[selected].gather(1,ordered.to(x.device)).cpu()
            row={'batch':context['batch'],'image_ids':list(context['image_ids']),
                 'call':call,'shape':list(x.shape),'axis':axis,'channel_draw_order':ch,
                 'indices':ordered,'x':values,'channel_ids':selected,'full_channels':C}
            def on_grad(grad):
                dest=records.setdefault(name,[])
                if len(dest)<manager.max_batches:dest.append(row)
            manager.tensor_hook_handles.append(x.register_hook(on_grad))
        return hook
    cal.TensorHookMgr._fw=observed_fw
    try:yield
    finally:cal.TensorHookMgr._fw=raw_fw


@contextmanager
def trace_layout(model, trace):
    """Capture actual top-k output and memory layout without replacing computation."""
    dec=model.decoder
    previous={n:dec.__dict__.get(n) for n in ('_get_encoder_input','_get_decoder_input')}
    present={n:n in dec.__dict__ for n in previous}
    enc_original=dec._get_encoder_input;query_original=dec._get_decoder_input
    def enc(self,*a,**kw):
        result=enc_original(*a,**kw)
        memory,shapes,starts=result
        trace['memory_shape']=list(memory.shape)
        trace['_memory_tensor']=memory
        trace['spatial_shapes']=copy.deepcopy(shapes)
        trace['level_start_index']=list(starts)
        return result
    def query(self,*a,**kw):
        raw=torch.topk
        def observe(*args,**kwargs):
            result=raw(*args,**kwargs)
            trace['topk']=result.indices.detach().cpu()
            return result
        try:
            torch.topk=observe
            return query_original(*a,**kw)
        finally:torch.topk=raw
    dec._get_encoder_input=MethodType(enc,dec);dec._get_decoder_input=MethodType(query,dec)
    try:yield
    finally:
        for n in previous:
            if present[n]:setattr(dec,n,previous[n])
            else:delattr(dec,n)


def collect_gradients(fp,base,loader,records,device="cuda"):
    if not records:raise ValueError("No activation samples collected")
    start=time.perf_counter()
    student=copy.deepcopy(base).to(device).eval().requires_grad_(False)
    teacher=copy.deepcopy(fp).to(device).eval().requires_grad_(False)
    probe=copy.deepcopy(base).to(device).eval()
    for m in cal._get_gptq_layers(probe).values():m.a_quantizer._ready=False
    # Original graph (weight requires_grad) and original inplace patch for exact sample replay.
    patches=cal._patch_inplace_acts(probe,enable=False)
    teacher_trace={};student_trace={};probe_trace={}
    pending=[];handles=[];calls={};batch_id=0
    received={n:[None]*len(rs) for n,rs in records.items()}
    replayed={n:[False]*len(rs) for n,rs in records.items()}
    shapes_checked={n:[False]*len(rs) for n,rs in records.items()}
    semantic_rows=[];losses=[]
    def hook_for(n,mode):
        def hook(mod,args,output):
            tag=(mode,n);call=calls.get(tag,0);calls[tag]=call+1
            positions=[j for j,r in enumerate(records[n]) if r['batch']==batch_id and r['call']==call]
            if not positions:return
            assert len(positions)==1;j=positions[0];r=records[n][j];x=args[0]
            assert list(x.shape)==r['shape'] and cal._get_ch_axis(mod,x)==r['axis'],(mode,n,batch_id,call)
            if semantic_kind(n)=='fixed_multiscale_memory_token':
                tr=probe_trace if mode=='fp' else student_trace
                assert x.data_ptr()==tr['_memory_tensor'].data_ptr(), 'value_proj must consume unsorted encoder memory'
            if mode=='fp':
                flat=x.detach().movedim(r['axis'],0).reshape(x.shape[r['axis']],-1)[r.get('channel_ids',list(range(x.shape[r['axis']])))]
                sample=flat.gather(1,r['indices'].to(x.device)).cpu()
                assert torch.equal(sample,r['x']),(n,batch_id,call,'float replay')
                replayed[n][j]=True
            else:
                shapes_checked[n][j]=True
                if semantic_kind(n).endswith('fallback') or not x.requires_grad:return
                idx=r['indices'].to(x.device)
                def on_grad(g):
                    flat=g.detach().movedim(r['axis'],0).reshape(g.shape[r['axis']],-1)[r.get('channel_ids',list(range(g.shape[r['axis']])))]
                    assert received[n][j] is None
                    received[n][j]=flat.gather(1,idx).cpu().double().square()
                pending.append(x.register_hook(on_grad))
        return hook
    try:
        for mode,model in [('fp',probe),('q',student)]:
            for n,m in cal._get_gptq_layers(model).items():
                if n in records:handles.append(m.register_forward_hook(hook_for(n,mode)))
        with trace_layout(teacher,teacher_trace),trace_layout(student,student_trace),trace_layout(probe,probe_trace),activation_ste(student):
            for batch_id,batch in enumerate(islice(loader,max(r['batch'] for rs in records.values() for r in rs)+1)):
                calls.clear();ids=[int(t['image_id']) for t in batch[1]]
                assert all(r['image_ids']==ids for rs in records.values() for r in rs if r['batch']==batch_id)
                x=batch[0].to(device)
                with torch.no_grad():t=teacher(x)
                p=probe(x)
                check_structure(p,t)
                assert torch.equal(cal._flatten_out(p).detach(),cal._flatten_out(t)), 'Original float graph differs from teacher'
                del p
                x=x.requires_grad_(True);s=student(x);check_structure(s,t)
                for key in ['memory_shape','spatial_shapes','level_start_index']:
                    assert teacher_trace[key]==student_trace[key]==probe_trace[key],key
                assert torch.equal(teacher_trace['topk'],probe_trace['topk'])
                ft,qt=teacher_trace['topk'],student_trace['topk']
                assert ft.shape==qt.shape
                # No query remapping attempted: fallback was predeclared for every query layer.
                overlap=[len(set(a.tolist()) & set(b.tolist())) for a,b in zip(ft,qt)]
                semantic_rows.append({'batch':batch_id,'image_ids':ids,
                    'fp_topk':ft.tolist(),'q_topk':qt.tolist(),'same_rank_count':int((ft==qt).sum()),
                    'common_query_identity_count_by_image':overlap,
                    **{k:teacher_trace[k] for k in ['memory_shape','spatial_shapes','level_start_index']}})
                loss=cal._proxy_distill_loss(s,t);assert torch.isfinite(loss)
                loss.backward();losses.append({'batch':batch_id,'images':len(ids),'mse':float(loss.detach())})
                print('SENSITIVITY',batch_id,float(loss.detach()),flush=True)
                for h in pending:h.remove()
                pending.clear();del x,t,s,loss
    finally:
        for h in handles+pending:h.remove()
        cal._restore_inplace_acts(patches)
    assert all(all(v) for v in replayed.values()) and all(all(v) for v in shapes_checked.values())
    assert all('quantize' not in m.a_quantizer.__dict__ for m in cal._get_gptq_layers(student).values())
    # Original floating samples stay intact, missing single records force entire layer fallback.
    gradient={};coverage=[]
    for n,rs in records.items():
        values=received[n]
        reason='query_identity_unproven' if semantic_kind(n).endswith('fallback') else 'missing_input_gradient' if any(v is None for v in values) else None
        C=rs[0].get('full_channels',rs[0]['x'].shape[0])
        g2=None if reason else [torch.cat([v[r.get('channel_ids',list(range(C))).index(c)] for r,v in zip(rs,values) if c in r.get('channel_ids',list(range(C)))]) if any(c in r.get('channel_ids',list(range(C))) for r in rs) else torch.empty(0,dtype=torch.float64) for c in range(C)]
        gradient[n]={'g2':g2,'fallback':reason}
        coverage.append({'layer':n,'semantic_kind':semantic_kind(n),'fallback':reason,
            'original_calls':len(rs),'matched_gradient_calls':sum(v is not None for v in values),
            'channels':C,'samples':sum(r['x'].numel() for r in rs),
            'nonzero':sum(int(torch.count_nonzero(v)) for v in g2) if g2 is not None else 0})
    return student,teacher,gradient,{'seconds':time.perf_counter()-start,'batches':losses,
        'semantic_batches':semantic_rows,'coverage':coverage,'all_float_samples_exactly_replayed':True,
        'all_record_shapes_and_calls_checked':True,'ste_restored':True,
        'output_loss_semantics':'Original flattened output-slot MSE retained; top-k query sets differ, so this is not object-matched detection loss; query-dependent layer gradients are not applied.',
        'mapping_policy':'Fixed batch/image/channel and fixed spatial/token positions only; all query-dependent layers fallback for every channel, even if some top-k entries happen to coincide.'}


def weights(g2):
    if g2 is None:return None,'layer_fallback'
    if not torch.isfinite(g2).all() or (g2<0).any():return None,'nonfinite_or_invalid'
    m=g2.double().mean()
    if not torch.isfinite(m) or m<=0:return None,'zero_or_invalid_mean'
    w=((g2.double()+.01*m)/(1.01*m)).float()
    if not torch.isfinite(w).all():return None,'invalid_after_float32_cast'
    return w,None


def original_choices(x,bit=4):
    qmax=2**bit-1
    neg,pos=x[x<0],x[x>0];assert len(neg) and len(pos)
    result=[]
    for pn in (.995,.997,.999):
        for pp in (.995,.997,.999,.9995,.9997,.9999):
            cn=max(torch.quantile(neg.abs(),pn).item(),1e-6)
            cp=max(torch.quantile(pos,pp).item(),1e-6)
            lo,hi=-cn,cp;s=(hi-lo)/(qmax+1e-12)
            z=int(max(0,min(qmax,round(-lo/s))))
            lo,hi=-z*s,(qmax-z)*s
            clipped=torch.clamp(x,lo,hi)
            q=torch.clamp(torch.round(clipped/s+z),0,qmax)
            deq=(q-z)*s
            error=(deq-x).pow(2)  # ORIGINAL float32 residual and square
            ss=(hi-lo)/(qmax+1e-12);zz=int(max(0,min(qmax,round(-lo/ss))))
            p=dict(zip(FIELDS,torch.tensor([hi,ss,zz,lo],dtype=torch.float32).tolist()))
            result.append((p,error))
    return result


def score(error,w):
    # Explicit float32 multiplication and ORIGINAL float32 mean, including U=1.
    assert error.dtype==torch.float32 and w.dtype==torch.float32
    return (error*w).mean().item()


def search(records,gradient,bootstrap,act_bits=4):
    u,s=copy.deepcopy(bootstrap),copy.deepcopy(bootstrap);rows=[];started=time.perf_counter()
    for n,rs in records.items():
        C=len(bootstrap[n]['scale'])
        xall=[torch.cat([r['x'][r.get('channel_ids',list(range(C))).index(c)] for r in rs if c in r.get('channel_ids',list(range(C)))]) if any(c in r.get('channel_ids',list(range(C))) for r in rs) else torch.empty(0,dtype=torch.float32) for c in range(C)]
        g=gradient[n]['g2']
        if g is not None:assert all(a.shape==b.shape for a,b in zip(g,xall))
        for c,x in enumerate(xall):
            mixed=bool((x<0).any() and (x>0).any())
            row={'layer':n,'channel':c,'samples':len(x),'mixed_sign':mixed,
                 'layer_fallback':gradient[n]['fallback'],'changed':False}
            if act_bits == 8 and x.numel() and torch.count_nonzero(x):
                from ACT_WQ.clip_search_8bit import select
                w, reason = weights(g[c] if g is not None else None)
                pu, ps, detail = select(x, w)
                for k in FIELDS:
                    u[n][k][c], s[n][k][c] = pu[k], ps[k]
                assert all(u[n][k][c] == bootstrap[n][k][c] for k in FIELDS), (n,c,'U does not recover bootstrap')
                row.update(detail)
                row.update(reason=reason, changed=pu != ps)
                rows.append(row)
                continue
            if not mixed:
                row['reason']='original_single_side_or_zero_fixed';rows.append(row);continue
            options=original_choices(x,act_bits);ones=torch.ones_like(x)
            w,reason=weights(g[c] if g is not None else None)
            if w is None:w=ones
            unweighted=[score(e,ones) for p,e in options]
            assert unweighted==[e.mean().item() for p,e in options]
            weighted=[score(e,w) for p,e in options]
            iu=min(range(18),key=lambda i:unweighted[i]);is_=min(range(18),key=lambda i:weighted[i])
            for k in FIELDS:
                u[n][k][c]=options[iu][0][k];s[n][k][c]=options[is_][0][k]
            assert all(u[n][k][c]==bootstrap[n][k][c] for k in FIELDS),(n,c,'U does not recover A')
            row.update({'reason':reason,'selected_U':iu,'selected_S':is_,
                'changed':any(u[n][k][c]!=s[n][k][c] for k in FIELDS),
                'mse_U':unweighted[iu],'mse_S':unweighted[is_],
                'weighted_U':weighted[iu],'weighted_S':weighted[is_],
                'weight_mean_f32':float(w.mean()),'weight_cv':float(w.std(unbiased=False)/w.mean()),
                'g2_nonzero':int(torch.count_nonzero(g[c])) if g is not None else 0})
            assert row['mse_U']<=row['mse_S'] and row['weighted_S']<=row['weighted_U']
            rows.append(row)
        print('SEARCH_ORIGINAL',n,flush=True)
    assert u==bootstrap
    return u,s,{'seconds':time.perf_counter()-started,'channels':rows,
        'changed_channels':sum(r['changed'] for r in rows),
        'weighted_mixed_channels':sum(r['mixed_sign'] and r.get('reason') is None for r in rows),
        'arithmetic':'CPU float32 x; original Python scalar s,z and aligned bounds; original f32 clamp/divide/add/round/dequant/residual/square. Normalize g2 in f64 then cast weights to f32; (f32 squared error * f32 weight).mean() in f32. U=ones uses the SAME score. Store final qparams in f32 only AFTER selection; first minimum wins.',
        'candidate_policy':'expanded_8bit_v1' if act_bits == 8 else 'legacy_18',
        'single_side':'Expanded search including Min-Max' if act_bits == 8 else 'Original bootstrap parameters retained; .999 percentile, no expanded candidates.'}


def tensor_hash(t):
    t=t.detach().cpu().contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def sample_fingerprint(records):
    """Order-stable digest of the actual layer/channel/position/value samples."""
    h=hashlib.sha256()
    for name in sorted(records):
        h.update(name.encode('utf-8'))
        for r in sorted(records[name],key=lambda z:(z['batch'],z['call'])):
            meta={k:r[k] for k in ('batch','image_ids','call','shape','axis','channel_ids','full_channels')}
            h.update(json.dumps(meta,sort_keys=True,separators=(',',':')).encode('utf-8'))
            for key in ('channel_draw_order','indices','x'):
                h.update(tensor_hash(r[key]).encode('ascii'))
    return h.hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def apply_parameters(model,params,bits):
    for name,m in cal._get_gptq_layers(model).items():
        axis=1 if isinstance(m,cal.GPTQConv2d) else -1
        m.enable_activation_quant(bits=bits,symmetric=False,perchannel=True,ch_axis=axis)
        if name in params:m.set_activation_qparams(**params[name])


def calibrate_unweighted(fp_model,q_model,dataloader,*,act_bits=4,num_batches=256,
                         per_batch_samples=16384,max_channels=-1,device='cuda',output_dir=None,
                         sample_cache=None):
    """Activation calibration control with every Fisher-related step removed.

    This is the paired control for :func:`calibrate_sensitivity`.  It keeps the
    same quantized model, calibration loader, random channel/position sampling,
    candidate grid and per-channel asymmetric quantizer.  It performs no
    teacher/student loss, backward pass, STE, gradient-square weighting,
    normalization, semantic matching or gradient fallback.
    """
    del fp_model  # Deliberately unused: no teacher or distillation in this arm.
    if act_bits<2 or act_bits>16:raise ValueError('act_bits must be 2..16')
    if num_batches<=0 or per_batch_samples<=0 or max_channels==0:raise ValueError('Invalid sampling limit')
    out=Path(output_dir) if output_dir else None
    if out:out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();model=None
    original_hashes={n:tensor_hash(p) for n,p in q_model.named_parameters()}
    samples={};records={};batches=[]

    if sample_cache:
        payload=torch.load(str(sample_cache),map_location='cpu')
        records=payload.get('records',{});batches=payload.get('batches',[])
        if not records or not batches:raise RuntimeError('Invalid or empty activation sample cache: '+str(sample_cache))
        targets=cal._get_gptq_layers(q_model)
        unknown=sorted(set(records)-set(targets))
        if unknown:raise RuntimeError('Activation sample cache contains unknown layers: '+', '.join(unknown[:5]))
        for name,rows in records.items():
            for row in rows:
                C=int(row['full_channels']);ids=[int(c) for c in row['channel_ids']]
                values=row['x'].detach().cpu().float()
                if values.ndim!=2 or values.shape[0]!=len(ids):
                    raise RuntimeError('Malformed cached activation samples for '+name)
                layer=samples.setdefault(name,{'C':C,'xs':[[] for _ in range(C)]})
                if layer['C']!=C:raise RuntimeError('Activation channel count changed for '+name)
                for i,c in enumerate(ids):layer['xs'][c].append(values[i])
        sample_source=str(Path(sample_cache).resolve())
    else:
        model=copy.deepcopy(q_model).to(device).eval()
        targets=cal._get_gptq_layers(model);counts={n:0 for n in targets}
        handles=[];calls={};context={'batch':0,'image_ids':[]}

    def hook_for(name):
        def hook(mod,args,output):
            del output
            call=calls.get(name,0);calls[name]=call+1
            if counts[name]>=num_batches or not args or not isinstance(args[0],torch.Tensor):return
            x=args[0]
            # The production bootstrap only samples tensors participating in
            # autograd.  Preserve that layer boundary without doing backward.
            if not x.requires_grad:return
            axis=cal._get_ch_axis(mod,x);axis=axis if axis>=0 else x.dim()+axis
            C=x.shape[axis];m_ch=C if max_channels is None or max_channels<0 else min(C,max_channels)
            k=max(1,per_batch_samples//m_ch);ch=torch.randperm(C,device=x.device)[:m_ch]
            flat=x.detach().movedim(axis,0).reshape(C,-1);draws=[];values=[]
            for c in ch.tolist():
                kk=min(k,flat.shape[1]);idx=torch.randint(0,flat.shape[1],(kk,),device=x.device)
                draws.append(idx.detach().cpu());values.append(flat[c,idx].cpu())
            ordered_ids=sorted(ch.tolist());lookup={c:i for i,c in enumerate(ch.tolist())}
            ordered_idx=torch.stack([draws[lookup[c]] for c in ordered_ids])
            ordered_x=torch.stack([values[lookup[c]] for c in ordered_ids]).float()
            layer=samples.setdefault(name,{'C':C,'xs':[[] for _ in range(C)]})
            if layer['C']!=C:raise RuntimeError('Activation channel count changed for '+name)
            for row,c in enumerate(ordered_ids):layer['xs'][c].append(ordered_x[row])
            records.setdefault(name,[]).append({'batch':context['batch'],'image_ids':list(context['image_ids']),
                'call':call,'shape':list(x.shape),'axis':axis,'channel_draw_order':ch.detach().cpu(),
                'indices':ordered_idx,'x':ordered_x,'channel_ids':ordered_ids,'full_channels':C})
            counts[name]+=1
        return hook

    if not sample_cache:
        patches=cal._patch_inplace_acts(model,enable=False)
        try:
            for name,module in targets.items():handles.append(module.register_forward_hook(hook_for(name)))
            for i,batch in enumerate(islice(dataloader,num_batches)):
                calls.clear();context['batch']=i
                context['image_ids']=[int(t['image_id']) for t in batch[1]]
                batches.append({'batch':i,'images':len(batch[0]),'image_ids':list(context['image_ids'])})
                # Standalone fallback. Paired runners use Full's exact cache.
                result=model(batch[0].to(device));del result
                print('UNWEIGHTED_BATCH',i,'images',len(context['image_ids']),flush=True)
                if all(v>=num_batches for v in counts.values()):break
        finally:
            for h in handles:h.remove()
            cal._restore_inplace_acts(patches)
            model.cpu();gc.collect();torch.cuda.empty_cache()
        sample_source='independent forward collection'
    if not samples:raise RuntimeError('No activation samples; check model graph and calibration data')

    params={};channel_rows=[]
    for name,layer in samples.items():
        xs=[torch.cat(v) if v else torch.empty(0,dtype=torch.float32) for v in layer['xs']]
        # Ones preserve the production candidate set/arithmetic while removing
        # all gradient-derived weighting.
        ones=torch.ones(len(xs),dtype=torch.float32)
        clip,scale,zero,clip_min=cal._grid_search_clip_channelwise(xs,ones,bit=act_bits)
        params[name]={'clip':clip.tolist(),'scale':scale.tolist(),'zero':zero.tolist(),'clip_min':clip_min.tolist()}
        channel_rows.append({'layer':name,'channels':len(xs),'sampled_channels':sum(v.numel()>0 for v in xs),
            'samples':sum(v.numel() for v in xs)})
    if {n:tensor_hash(p) for n,p in q_model.named_parameters()}!=original_hashes:
        raise AssertionError('Parameters changed during unweighted collection')
    apply_parameters(q_model,params,act_bits)
    summary={'mode':'unweighted activation reconstruction MSE (no Fisher-related mechanism)',
        'removed':['teacher/student reconstruction loss','backward gradients','STE',
            'elementwise gradient-square weighting','weight normalization/smoothing',
            'semantic matching and gradient fallback'],
        'kept':['matched production WbAb model/data preparation','per-channel asymmetric activations',
            'production random channel/position sampling','production clipping candidates',
            'ordinary unweighted float32 reconstruction MSE'],
        'images':sum(b['images'] for b in batches),'batches':len(batches),
        'initialized_layers':len(params),'parameters_unchanged_during_collection':True,
        'sample_source':sample_source,
        'sample_fingerprint':sample_fingerprint(records),
        'seconds':time.perf_counter()-started}
    if out:
        torch.save({'records':records,'batches':batches},out/'unweighted_sample_cache.pth')
        dump(out/'activation_params.json',params);dump(out/'unweighted_sampling.json',channel_rows)
        dump(out/'calibration_summary.json',summary)
    print('ACTIVATION SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)
    return params,summary


@contextmanager
def cleanup_legacy_hooks():
    """The original routine only detaches on success; cover exception exits too."""
    attach=cal.TensorHookMgr.attach;managers=[]
    def tracked(manager):
        managers.append(manager)
        return attach(manager)
    cal.TensorHookMgr.attach=tracked
    try:yield
    finally:
        cal.TensorHookMgr.attach=attach
        for manager in managers:manager.detach()


def calibrate_sensitivity(fp_model,q_model,dataloader,*,act_bits=4,num_batches=256,
                          per_batch_samples=16384,max_channels=-1,device='cuda',output_dir=None):
    """Calibrate isolated copies and commit only final activation parameters.

    Original FP samples/calls, quantized-bootstrap STE gradients, original f32
    candidate arithmetic. Query-dependent layers always fall back to all ones.
    Caller model devices/training/parameter flags are never changed by collection.
    """
    if act_bits<2 or act_bits>16:raise ValueError('act_bits must be 2..16')
    if num_batches<=0 or per_batch_samples<=0 or max_channels==0:raise ValueError('Invalid sampling limit')
    out=Path(output_dir) if output_dir else None
    if out:out.mkdir(parents=True,exist_ok=True)
    original_hashes={n:tensor_hash(p) for n,p in q_model.named_parameters()}
    fp=copy.deepcopy(fp_model);base=copy.deepcopy(q_model)
    context={'calls':{},'batch':0,'image_ids':[]};records={};batches=[]
    class RecordingLoader:
        def __iter__(self):
            for i,b in enumerate(dataloader):
                ids=[int(t['image_id']) for t in b[1]]
                context.update({'calls':{},'batch':i,'image_ids':ids})
                batches.append({'batch':i,'images':len(b[0]),'image_ids':ids})
                print('ORIGINAL_BATCH',i,'images',len(ids),flush=True)
                yield b
    started=time.perf_counter()
    print('ACTIVATION stage 1/3: original float samples, indices and bootstrap',flush=True)
    with cleanup_legacy_hooks(),record_original_indices(records,context):
        bootstrap=cal.calibrate_gptq_activations(fp,base,RecordingLoader(),act_bits=act_bits,
            num_batches=num_batches,per_batch_samples=per_batch_samples,max_channels=max_channels,device=device)
    if not records:raise RuntimeError('No activation samples; check model graph and calibration data')
    assert {n:tensor_hash(p) for n,p in base.named_parameters()}==original_hashes,'Parameters updated during original calibration'
    base.zero_grad(set_to_none=True);base.cpu();fp.cpu();gc.collect();torch.cuda.empty_cache()
    before_base={n:tensor_hash(t) for n,t in base.state_dict().items()}
    before_fp={n:tensor_hash(t) for n,t in fp.state_dict().items()}
    if out:
        torch.save({'records':records,'batches':batches},out/'original_index_cache.pth')
        dump(out/'bootstrap_params.json',bootstrap)
    print('ACTIVATION stage 2/3: fixed bootstrap student, reconstruction MSE, exact STE',flush=True)
    print('REPRODUCIBILITY: CUDA grid_sample backward is nondeterministic in this environment; seed fixes original samples, not bitwise gradient replay. Original operator and f32 scoring are retained.',flush=True)
    student,teacher,gradient,diag=collect_gradients(fp,base,dataloader,records,device)
    assert {n:tensor_hash(t) for n,t in student.state_dict().items()}==before_base
    assert {n:tensor_hash(t) for n,t in teacher.state_dict().items()}==before_fp
    assert all(not m._w_offline_done and not m.w_quantizer.ready() for m in cal._get_gptq_layers(student).values())
    diag.update({'student_state_unchanged':True,'teacher_state_unchanged':True,'weights_float_and_unquantized':True})
    if out:
        dump(out/'collection_diagnostics.json',diag)
        torch.save(gradient,out/'original_position_gradients.pth')
    del student,teacher,fp,base;gc.collect();torch.cuda.empty_cache()
    print('ACTIVATION stage 3/3: original candidates, element weights inside f32 squared-error mean',flush=True)
    uniform,sensitive,search_diag=search(records,gradient,bootstrap,act_bits)
    assert uniform==bootstrap
    assert {n:tensor_hash(p) for n,p in q_model.named_parameters()}==original_hashes
    apply_parameters(q_model,sensitive,act_bits)
    fallback={}
    for r in diag['coverage']:
        if r['fallback']:fallback[r['fallback']]=fallback.get(r['fallback'],0)+1
    summary={'mode':'Fisher-inspired elementwise reconstruction-gradient squared sensitivity (not strict Fisher)',
        'images':sum(b['images'] for b in batches),'batches':len(batches),'initialized_layers':len(bootstrap),
        'eligible_layers':sum(r['fallback'] is None for r in diag['coverage']),
        'eligible_channels':sum(r['channels'] for r in diag['coverage'] if r['fallback'] is None),
        'fallback_layers_by_reason':fallback,'weighted_mixed_channels':search_diag['weighted_mixed_channels'],
        'changed_channels':search_diag['changed_channels'],'uniform_exact_bootstrap':True,
        'sample_fingerprint':sample_fingerprint(records),
        'parameters_unchanged_during_collection':True,'seconds':time.perf_counter()-started}
    if out:
        dump(out/'activation_params.json',sensitive);dump(out/'search_diagnostics.json',search_diag)
        dump(out/'calibration_summary.json',summary)
    print('ACTIVATION SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)
    return sensitive,summary


def tensor_search(records, gradient, bit):
    """Pool identical samples/element weights; change only qparam sharing group.

    All four granularity/weight cells use channel-bootstrap gradient caches.
    This is a frozen-gradient granularity ablation, not tensor-bootstrap PTQ.
    """
    uniform, sensitive, diagnostics = {}, {}, []
    for name, rows in records.items():
        channels = rows[0]['full_channels']
        xs = [torch.cat([r['x'][r['channel_ids'].index(c)] for r in rows
                        if c in r['channel_ids']])
              if any(c in r['channel_ids'] for r in rows)
              else torch.empty(0,dtype=torch.float32) for c in range(channels)]
        g = gradient[name]['g2']
        ws = []
        reasons = []
        for c,x in enumerate(xs):
            w,reason = weights(g[c] if g is not None and x.numel() else None)
            if w is None:
                w = torch.ones_like(x)
            if w.shape != x.shape:
                raise RuntimeError('Tensor pool gradient/sample mismatch: '+name)
            ws.append(w); reasons.append(reason)
        x,w = torch.cat(xs),torch.cat(ws)
        bootstrap = cal._grid_search_clip_channelwise([x],torch.ones(1),bit)
        # Legacy helper returns clip, scale, zero, clip_min, matching FIELDS.
        u = {k:v.tolist() for k,v in zip(FIELDS,bootstrap)}
        s = copy.deepcopy(u)
        row = {'layer':name,'samples':len(x),'fallback_by_channel':reasons,
               'weight_normalization':'same per-channel normalized elements as channel search',
               'gradient_bootstrap':'shared channel bootstrap, frozen across granularity'}
        if bit == 8 and x.numel() and torch.count_nonzero(x):
            from ACT_WQ.clip_search_8bit import select
            pu, ps, detail = select(x, w)
            u = {k:[pu[k]] for k in FIELDS}
            s = {k:[ps[k]] for k in FIELDS}
            row.update(detail)
        elif (x<0).any() and (x>0).any():
            candidates = original_choices(x,bit)
            su = [score(e,torch.ones_like(x)) for p,e in candidates]
            ss = [score(e,w) for p,e in candidates]
            iu,is_ = min(range(18),key=su.__getitem__),min(range(18),key=ss.__getitem__)
            u = {k:[candidates[iu][0][k]] for k in FIELDS}
            s = {k:[candidates[is_][0][k]] for k in FIELDS}
            row.update(selected_U=iu,selected_S=is_)
        uniform[name],sensitive[name] = u,s
        diagnostics.append(row)
    return uniform,sensitive,diagnostics


