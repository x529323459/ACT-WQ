# ACT-WQ: GPTQ quantization code

论文配套精简代码。运行入口为 `QuantRTDetr/main.py`。

保留的流程：RT-DETR 模型加载与融合 → 前置激活校准/量化 → 沿量化激活路径收集 Hessian → GPTQ 权重量化 → COCO 评估和模型导出。量化算法、默认位宽策略和模型初始化顺序未更改。

## 环境

本次回归验证使用 Python 3.9.23、PyTorch 2.0.1+cu118、torchvision 0.15.2+cu118。完整量化需要 NVIDIA GPU；代码使用这一版 torchvision 的 `datapoints` 和 v2 transforms 接口。

先安装匹配 CUDA 环境的上述 PyTorch/torchvision，再安装其他依赖：

```sh
pip install -r requirements.txt
cd QuantRTDetr
```

## 数据与权重

代码包不包含数据、预训练权重、实验结果或历史备份。运行前准备：

```text
QuantRTDetr/
  pre_model/
    ResNet18_vd_pretrained_from_paddle.pth
    rtdetr_r18vd_dec3_6x_coco_from_paddle.pth
  data/coco/
    train2017/
    val2017/
    calib100/
    annotations/
      instances_train2017.json
      instances_val2017.json
      instances_calib100.json
```

可以修改 `rtdetr/config.yml` 中的路径；相对路径以 `QuantRTDetr` 为基准。`--resume` 可指定检测器权重。默认配置仍会加载 ResNet18 主干权重，并在模型准备时读取训练集样本，因此即使跳过评估，也需要相应数据。请使用论文实验所采用的校准子集；本包未附该子集的样本清单，不保证重新随机抽样得到相同结果。

本机可将完整配置另存为 `rtdetr/config.local.yml` 并填写本机数据路径。无 `--config` 参数时优先使用该本地配置；显式 `--config` 可覆盖。此文件、`pre_model/`、`data/` 和 `output/` 已由 `.gitignore` 排除，保留它们供本机运行不会增加 Git 上传内容。

## 位宽配置

| 设置 | 权重粒度 | 激活粒度 | 权重范围搜索 |
|---|---|---|---|
| W4A4 | 逐通道 | 逐通道 | 关闭 |
| W6A6（默认） | 逐通道 | 逐通道 | 关闭 |
| W8A8 | 逐张量 | 逐张量 | 开启 |

权重和激活位宽分别设置，支持混合位宽；各自为 8 bit 时采用逐张量，否则采用逐通道。参数接受 2–16 bit，但该范围不代表每种设置均完成论文实验验证。

```sh
python main.py --gptq-bits 4 --gptq-act-bits 4 --output-dir output/w4a4
python main.py --gptq-bits 6 --gptq-act-bits 6 --output-dir output/w6a6
python main.py --gptq-bits 8 --gptq-act-bits 8 --output-dir output/w8a8
```

输出目录必须尚不存在。`--no-gptq-act` 用于仅权重量化；`--skip-evaluation` 跳过最终评估。

前置激活校准保留 `sensitivity`（默认敏感度加权搜索）、`unweighted`、`minmax` 和 `legacy`。其中 `minmax` 需要通过 `--gptq-act-sample-cache` 提供此前 Full 流程生成的样本缓存；`legacy` 不支持逐张量激活配置。详细参数见 `python main.py --help`。

## 文件组织

- `ACT_WQ/`：仅保留 GPTQ、前置激活校准、8-bit 搜索和必要辅助代码。
- `rtdetr/`：量化流程使用的模型、数据加载、模型准备和评估依赖。
- `utils/fuse.py`：模型融合。
- 根目录 `THIRD_PARTY_LICENSE`：保留的第三方许可证；原 `GPTQ` 目录及语言模型示例已移除。

已移除 MQBench 的 AdaRound、QDrop、LSQ、PACT、DSQ、DoReFa、TQT、NNIE 等其他量化方法、相关 observer/reconstruction 实现，以及无关实验、绘图和审稿文件。保留源码中的原有版权声明。

## 验证

发布精简前已通过 8 项回归测试和 6 项激活校准检查；开发测试脚本不包含在此精简包中。通用工具文件中所需的两个定义已原样合并至 `ACT_WQ/quant_model.py`，通过语法树一致性核对。

本次未重新运行 COCO 全量校准或评估，因此不报告新的 AP 或量化性能结果。
