# PIAFusion_pytorch

9.13  加入多卡训练（model包含encoder，decoder，discriminator等多个结构，应当对每个结构都用torch.nn.DataParallel包裹，而不是只对model进行包裹）

9.12  加入coattention模块，加入指标评估计算部分
