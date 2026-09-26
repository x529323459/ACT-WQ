"""RT-DETR PTQ with sensitivity, paired unweighted, and historical activation calibration modes."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,contextlib,datetime,json,sys,traceback
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parent


def create_parser():
    p=argparse.ArgumentParser(description=__doc__)
    local_config = ROOT/'rtdetr/config.local.yml'
    p.add_argument('--config','-c',default=str(local_config) if local_config.is_file() else 'rtdetr/config.yml')
    p.add_argument('--resume','-r',default='pre_model/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth',help='pretrained checkpoint; EMA used when configured')
    p.add_argument('--tuning','-t',default=None,help='optional partial checkpoint applied before calibration')
    p.add_argument('--reconstruction','-rec',default=True,help='retained compatibility option; MQBench reconstruction does not apply to GPTQ')
    p.add_argument('--test-only',action='store_true',default=True,help='compatibility flag; this PTQ entry never trains')
    p.add_argument('--amp',action='store_true',help='unsupported by exact float32 calibration; explicit use raises an error')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--use-gptq',action='store_true',default=True,help='compatibility flag; GPTQ is the default backend')
    p.add_argument('--gptq-bits',type=int,default=4)
    p.add_argument('--gptq-blocksize',type=int,default=128)
    p.add_argument('--gptq-percdamp',type=float,default=.01)
    p.add_argument('--gptq-weight-samples',type=int,default=128,help='original per-layer statistics call limit')
    # Python 3.8 compatibility: explicit flags retain last-option-wins behavior.
    p.add_argument('--gptq-act',dest='gptq_act',action='store_true',default=True,help='enable activation quantization (default)')
    p.add_argument('--no-gptq-act',dest='gptq_act',action='store_false',help='disable activation quantization; use weights only')
    p.add_argument('--activation-calibration',choices=['sensitivity','unweighted','minmax','legacy'],default='sensitivity',help='sensitivity search, unweighted MSE search, paired Min-Max without search, or legacy')
    p.add_argument('--gptq-act-bits',type=int,default=4)
    p.add_argument('--gptq-act-batches',type=int,default=256,help='original batch/call cap; stops at end of configured calibration set')
    p.add_argument('--gptq-act-samples',type=int,default=16384)
    p.add_argument('--gptq-act-max-channels',type=int,default=-1,help='-1 covers all channels; positive values keep original random channel sampling')
    p.add_argument('--gptq-act-sample-cache',type=Path,help='Full-arm activation sample cache reused by the paired unweighted/Min-Max controls')
    p.add_argument('--num-workers',type=int,default=0,help='0 gives verified deterministic loader; -1 uses configured workers')
    p.add_argument('--output-dir',type=Path,help='new run directory; existing directory is rejected')
    p.add_argument('--skip-evaluation',action='store_true',help='export quantized model without validation (default evaluates all images)')
    return p


def resolve_input(value,launch):
    p=Path(value).expanduser()
    if p.is_absolute():return p.resolve()
    if (launch/p).exists():return (launch/p).resolve()
    return (ROOT/p).resolve()


class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,text):
        for stream in self.streams:stream.write(text);stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams:stream.flush()


def main(args):
    # Explicit experiment wrappers may retain their named granularity.
    from ACT_WQ.ptq_policy import configure
    configure(args)
    if args.gptq_act and not args.gptq_activation_perchannel and args.activation_calibration == 'legacy':
        raise ValueError('8-bit per-tensor supports sensitivity, unweighted or minmax calibration; legacy is per-channel only')
    if args.gptq_act and args.activation_calibration == 'minmax' and not args.gptq_act_sample_cache:
        raise ValueError('Paired Min-Max requires --gptq-act-sample-cache from Full')
    if args.amp:raise ValueError('--amp is incompatible with the verified float32 calibration; omit it')
    if not 2<=args.gptq_bits<=16 or not 2<=args.gptq_act_bits<=16:raise ValueError('Bit widths must be 2..16')
    if args.gptq_blocksize<=0 or args.gptq_percdamp<0 or args.gptq_weight_samples<=0:raise ValueError('Invalid GPTQ settings')
    if args.gptq_act_batches<=0 or args.gptq_act_samples<=0 or args.gptq_act_max_channels==0:raise ValueError('Invalid calibration sampling settings')
    launch=Path.cwd();args.config=str(resolve_input(args.config,launch));args.resume=str(resolve_input(args.resume,launch))
    if args.tuning:args.tuning=str(resolve_input(args.tuning,launch))
    if args.gptq_act_sample_cache:
        args.gptq_act_sample_cache=str(resolve_input(args.gptq_act_sample_cache,launch))
        if args.activation_calibration not in ('unweighted','minmax'):
            raise ValueError('--gptq-act-sample-cache is only valid with unweighted or minmax activation calibration')
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    # Existing config paths are project-root relative, independent of launch cwd.
    for p in cfg['dataloader'].values():
        for key in ('img_folder','ann_file'):
            value=Path(p['dataset'][key])
            if not value.is_absolute():p['dataset'][key]=str((ROOT/value).resolve())
    out=(launch/args.output_dir).resolve() if args.output_dir else (ROOT/cfg.get('output_dir','output')/('ptq_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    out.mkdir(parents=True,exist_ok=False)
    args.output_dir=str(out)
    with (out/'run.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
        try:
            print('OUTPUT',out,flush=True)
            if args.reconstruction is not True:print('NOTE --reconstruction only applied to the MQBench backend; GPTQ has never used it.',flush=True)
            from ACT_WQ.ptq_runtime import run
            from ACT_WQ.sensitivity_calibration import dump
            dump(out/'run_config.json',{'args':vars(args),'config':cfg,'launch_directory':str(launch),'project_root':str(ROOT)})
            return run(cfg,args,out)
        except BaseException:
            traceback.print_exc();raise


if __name__=='__main__':main(create_parser().parse_args())
