import matplotlib.pyplot as plt


def calculate_mse_error(list1,list2):
    assert len(list1) == len(list2), "两个列表长度不相等，无法计算。"
    mse_error = []
    for tensor1, tensor2 in zip(list1, list2):
        assert tensor1.shape == tensor2.shape, "张量形状不相等，无法计算MSE。"
        squared_diff = (tensor1 - tensor2) ** 2
        mse = squared_diff.mean()
        mse_error.append(mse.item())
    return mse_error

def plot(name_item,fp_out,quant_out):
    # 将tensor值展平为1维
    fp_out = fp_out.flatten()
    quant_out = quant_out.flatten()
    # 创建一个图表
    # plt.figure(figsize=(8, 6))
    # # 绘制fp_out的直方图
    # plt.hist(fp_out, bins=30, color='blue', alpha=0.7, edgecolor='black')
    # plt.title(name_item)
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # # 显示图表
    # plt.show()

    # 创建一个包含两个子图的图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # 绘制fp_out的直方图
    ax1.hist(fp_out, bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax1.set_title('FP Output Distribution')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    # 绘制quant_out的直方图
    ax2.hist(quant_out, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax2.set_title('Quantized Output Distribution')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    # 设置图表的总标题
    fig.suptitle(f"Histogram of {name_item}")
    # 显示图表
    plt.tight_layout()  # 调整子图之间的间距
    plt.show()

def sortprint(name,error,fp_out,quant_out):
    sorted_pairs = sorted(zip(name, error,fp_out,quant_out), key=lambda x: x[1], reverse=True)
    for name_item, error_item,fp_out,quant_out in sorted_pairs[:20]:
        print(f"{name_item}:{error_item}")
        plot(name_item,fp_out,quant_out)

def sensitivity_analysis(wrapped_modules,solver):
    #model = solver.ema.module if solver.ema else solver.model
    fp_out = []
    quant_out = []
    namelist = []
    for name, module in wrapped_modules.items():
        if module.quant_out == None or module.raw_out == None:
            continue
        namelist.append(name)
        fp_out.append(module.raw_out)
        quant_item = module.quant_out.to('cpu')
        quant_out.append(quant_item)

    errorlist = calculate_mse_error(fp_out,quant_out)
    sortprint(namelist,errorlist,fp_out,quant_out)