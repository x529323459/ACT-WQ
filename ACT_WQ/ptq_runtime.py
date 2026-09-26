"""Self-contained GPTQ deployment flow; no experimental driver dependencies."""
import copy,gc,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from rtdetr.model_solver import config_model,config_solver
from rtdetr.data.transforms import Compose
from rtdetr.data.coco.coco_dataset import CocoDetection
from rtdetr.data.dataloader import default_collate_fn
from rtdetr.nn.rtdetr_postprocessor import RTDETRPostProcessor
from rtdetr.utils.det_engine import evaluate
from utils.fuse import fuse_model
from ACT_WQ.quant_model import quant_model,collect_gptq_data
from ACT_WQ import act_calibration as cal
from ACT_WQ.sensitivity_calibration import calibrate_sensitivity,calibrate_unweighted,cleanup_legacy_hooks,dump,tensor_hash


def seed_all(seed):
    torch.set_num_threads(4)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def make_loader(cfg,split,workers):
    p=copy.deepcopy(cfg['dataloader'][split]);d=p['dataset']
    if p.get('shuffle',False):raise ValueError(split+' must use shuffle: false for stable replay')
    if p.get('collate_fn','default_collate_fn')!='default_collate_fn':raise ValueError('Unsupported collate_fn')
    ds=CocoDetection(Compose(**d['transforms']),d['img_folder'],d['ann_file'],False,False)
    return DataLoader(ds,batch_size=p['batch_size'],shuffle=False,
        num_workers=p['num_workers'] if workers<0 else workers,
        drop_last=p.get('drop_last',False),collate_fn=default_collate_fn)


def load_weights(fp,path,use_ema,partial=False):
    ck=torch.load(path,map_location='cpu')
    state=ck['ema']['module'] if use_ema and 'ema' in ck else ck.get('model',ck.get('state_dict',ck))
    if partial:
        current=fp.state_dict();matched={k:v for k,v in state.items() if k in current and current[k].shape==v.shape}
        print('TUNING matched',len(matched),'missing/mismatched',len(current)-len(matched),flush=True)
        fp.load_state_dict(matched,strict=False)
    else:fp.load_state_dict(state,strict=True)


def build(cfg,args):
    # Match ACT-WQ main.py preparation, including backbone loading, EMA
    # construction and config_solver's augmented train_dataset[0] access.
    # Work on a private config: config_solver/Compose mutate their arguments.
    model_cfg=copy.deepcopy(cfg)
    model_cfg['resume']=args.resume
    model_cfg['tuning']=None
    for split in model_cfg['dataloader'].values():
        split['num_workers']=0
    original_model=config_model(model_cfg)
    solver=config_solver(model_cfg,original_model)
    if args.resume:
        solver.resume(args.resume)
    fp=solver.ema.module if solver.ema is not None else solver.model
    if args.tuning:load_weights(fp,args.tuning,cfg.get('use_ema',True),partial=True)
    fuse_model(fp)
    model=copy.deepcopy(fp)
    quant_model(model,{},use_gptq=True,gptq_config={'bits':args.gptq_bits,'perchannel':getattr(args,'gptq_weight_perchannel',True),'sym':True})
    solver.setup(model)
    fp.eval().requires_grad_(False)
    print('PREPARATION ACT-WQ model -> full solver/train sample -> resume -> EMA -> fuse -> quantize -> setup; workers=0',flush=True)
    return fp,model


def runtime_flags(model):
    return {n:{'ready':bool(m.a_quantizer.ready()),'enabled':m.a_quantizer._enabled,
        'perchannel':m.a_quantizer._perchannel,'symmetric':m.a_quantizer._symmetric,
        'axis':m.a_quantizer._ch_axis,'weight_perchannel':m.w_quantizer.perchannel,
        'weight_symmetric':m.w_quantizer.sym} for n,m in cal._get_gptq_layers(model).items()}


calibrate_experiment = None  # Optional process-local experiment callback.


def run(cfg,args,out):
    seed_all(args.seed)
    loader=make_loader(cfg,'calib_dataloader',args.num_workers)
    fp,model=build(cfg,args)
    print('GRANULARITY weights:', 'per-channel' if getattr(args,'gptq_weight_perchannel',True) else 'per-tensor',
          'activations:', 'per-channel' if getattr(args,'gptq_activation_perchannel',True) else 'per-tensor',flush=True)
    print('CALIBRATION DATA images',len(loader.dataset),'batches',len(loader),'batch_size',loader.batch_size,flush=True)
    print('ACTIVATION MODE',args.activation_calibration if args.gptq_act else 'disabled (weight-only)',flush=True)
    if args.gptq_act:
        kwargs=dict(act_bits=args.gptq_act_bits,num_batches=args.gptq_act_batches,
            per_batch_samples=args.gptq_act_samples,max_channels=args.gptq_act_max_channels,device='cuda')
        if calibrate_experiment is not None:
            calibrate_experiment(fp,model,loader,output_dir=out,**kwargs)
        elif args.activation_calibration == 'minmax':
            from ACT_WQ.minmax_calibration import calibrate_minmax
            calibrate_minmax(model,sample_cache=args.gptq_act_sample_cache,output_dir=out,
                act_bits=args.gptq_act_bits,perchannel=args.gptq_activation_perchannel)
        elif not getattr(args,'gptq_activation_perchannel',True):
            from ACT_WQ.tensor_calibration import calibrate_tensor
            calibrate_tensor(fp,model,loader,output_dir=out,mode=args.activation_calibration,
                             sample_cache=args.gptq_act_sample_cache,**kwargs)
        elif args.activation_calibration=='sensitivity':
            calibrate_sensitivity(fp,model,loader,output_dir=out,**kwargs)
        elif args.activation_calibration=='unweighted':
            calibrate_unweighted(fp,model,loader,output_dir=out,
                sample_cache=args.gptq_act_sample_cache,**kwargs)
        else:
            start=time.perf_counter()
            with cleanup_legacy_hooks():params=cal.calibrate_gptq_activations(fp,model,loader,**kwargs)
            dump(out/'activation_params.json',params)
            dump(out/'calibration_summary.json',{'mode':'legacy','seconds':time.perf_counter()-start,'initialized_layers':len(params)})
    else:
        for m in cal._get_gptq_layers(model).values():
            m.a_quantizer._enabled=False;m.a_quantizer._ready=False
    fp.cpu();model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    print('WEIGHTS stage 1/2: original statistics collection',flush=True)
    start=time.perf_counter();model.cuda().eval()
    for m in cal._get_gptq_layers(model).values():m.device=torch.device('cuda')
    collect_gptq_data(model,loader,num_samples=args.gptq_weight_samples)
    model.cpu()
    for m in cal._get_gptq_layers(model).values():
        if m.H is not None:m.H=m.H.cpu()
        m.device=torch.device('cpu')
    gc.collect();torch.cuda.empty_cache();rows=[]
    from ACT_WQ.weight_range import FACTORS, quantize_layer
    range_enabled = getattr(args, 'gptq_weight_range_search', False)
    range_report = {'enabled':range_enabled, 'status':'running', 'weight_scale_factors':list(FACTORS),
        'objective':'sum(((Q-W) @ H) * (Q-W)); original uncentered calibration H, without damping',
        'selection_data':'same calibration statistics as GPTQ, not independent held-out data', 'layers':[]}
    if range_enabled:
        print('WEIGHT RANGE SEARCH enabled: 8-bit channel/tensor, factors', FACTORS, flush=True)
        dump(out/'weight_range_search.json',range_report)
    print('WEIGHTS stage 2/2: original GPTQ solver, one layer on GPU at a time',flush=True)
    for n,m in cal._get_gptq_layers(model).items():
        m.cuda();m.device=torch.device('cuda')
        if m.H is not None:m.H=m.H.cuda()
        h=tensor_hash(m.H) if m.H is not None else None;count=m.nsamples
        with torch.no_grad():
            _, detail = quantize_layer(m, range_enabled, blocksize=args.gptq_blocksize,percdamp=args.gptq_percdamp)
        if detail is not None:
            detail['layer']=n;range_report['layers'].append(detail)
            dump(out/'weight_range_search.json',range_report)
            print('WEIGHT_RANGE',n,'factor',detail.get('selected_factor'),'error',detail.get('selected_error'),flush=True)
        rows.append({'layer':n,'samples':count,'H_sha256':h,'offline_done':m._w_offline_done})
        m.H=None;m.cpu();torch.cuda.empty_cache();print('QUANTIZED',n,flush=True)
    dump(out/'weight_statistics.json',{'seconds':time.perf_counter()-start,'layers':rows})
    if range_enabled:
        range_report['status']='complete';dump(out/'weight_range_search.json',range_report)
    flags={n:{'offline_done':m._w_offline_done} for n,m in cal._get_gptq_layers(model).items()}
    torch.save({'state_dict':model.state_dict(),'flags':flags,'inference_flags':runtime_flags(model),'args':vars(args)},out/'model_quantized.pth')
    dump(out/'inference_flags.json',runtime_flags(model))
    if not args.skip_evaluation:
        val=make_loader(cfg,'val_dataloader',args.num_workers)
        if val.drop_last:raise ValueError('Validation drop_last must be false for full evaluation')
        post=RTDETRPostProcessor(**cfg['model']['RTDETRPostProcessor']).cuda().eval()
        start=time.perf_counter();model.cuda().eval()
        print('EVALUATE full configured validation set:',len(val.dataset),'images',flush=True)
        metrics,evaluator=evaluate(model,torch.nn.Identity(),post,val,val.dataset.coco,'cuda',out)
        result={'images':len(val.dataset),'metrics':metrics,'seconds':time.perf_counter()-start}
        torch.save(evaluator.coco_eval['bbox'].eval,out/'coco_eval.pth')
        dump(out/'metrics.json',result);print('FINAL METRICS',json.dumps(result),flush=True)
    print('RUN COMPLETE',out,flush=True)
    return fp,model
