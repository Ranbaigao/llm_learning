import torch
import torch.nn as nn

# ==========================================
# 1. 前置依赖 (沿用之前的核心逻辑)
# ==========================================
def sinkhorn_knopp(M, n_iters=20, tau=1.0):
    H = torch.exp(M / tau)
    for _ in range(n_iters):
        H = H / (H.sum(dim=-1, keepdim=True) + 1e-8)
        H = H / (H.sum(dim=-2, keepdim=True) + 1e-8)
    return H

class mHCBlock(nn.Module):
    def __init__(self, dim, n_streams=4):
        super().__init__()
        self.dim = dim
        self.n = n_streams
        total_dim = n_streams * dim
        
        self.norm = nn.RMSNorm(total_dim)
        self.proj_pre  = nn.Linear(total_dim, n_streams)
        self.proj_post = nn.Linear(total_dim, n_streams)
        self.proj_res  = nn.Linear(total_dim, n_streams * n_streams)
        
        self.alpha_pre  = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res  = nn.Parameter(torch.tensor(0.01))

    def forward(self, x, layer_func):
        B, L, N, C = x.shape
        x_global = x.view(B, L, -1)
        x_norm = self.norm(x_global)
        
        # 计算路由权重
        H_pre = torch.sigmoid(self.alpha_pre * self.proj_pre(x_norm)).view(B, L, N, 1)
        H_post = 2.0 * torch.sigmoid(self.alpha_post * self.proj_post(x_norm)).view(B, L, N, 1)
        H_res = sinkhorn_knopp(self.alpha_res * self.proj_res(x_norm).view(B, L, N, N))

        # A. 聚合
        x_main_in = (x * H_pre).sum(dim=2) 
        # B. 主干计算
        x_main_out = layer_func(x_main_in)
        # C. 残差互通
        x_res_mixed = torch.matmul(H_res, x)
        # D. 广播分发
        x_next = x_res_mixed + H_post * x_main_out.unsqueeze(2)
        
        return x_next

# ==========================================
# 2. 核心处理厂：标准的前馈神经网络 (FFN)
# ==========================================
class FeedForward(nn.Module):
    """标准的 MLP 处理模块，它完全不需要知道外面有 mHC 的存在"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. 宏观架构：基于 mHC 的分类器
# ==========================================
class mHC_SequenceClassifier(nn.Module):
    def __init__(self, vocab_size, dim, n_streams, num_layers, out_classes):
        super().__init__()
        self.dim = dim
        self.n = n_streams
        
        # 1. 输入层：将词元映射为向量 [B, L, C]
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # 2. 拓宽车道：把 1 条 C 维的车道，强行拆解成 n 条 C 维的车道
        self.expand = nn.Linear(dim, n_streams * dim)
        
        # 3. 堆叠 mHC 层和 FFN 层
        self.mhc_blocks = nn.ModuleList([mHCBlock(dim, n_streams) for _ in range(num_layers)])
        self.ffn_layers = nn.ModuleList([FeedForward(dim, dim * 4) for _ in range(num_layers)])
        
        # 4. 压缩车道：把 n 条车道的信息压平，合并回 1 条 C 维车道
        self.collapse = nn.Linear(n_streams * dim, dim)
        
        # 5. 输出层：简单的池化 + 分类头
        self.head = nn.Linear(dim, out_classes)

    def forward(self, x):
        B, L = x.shape
        
        # [B, L] -> [B, L, C]
        x = self.embedding(x)
        
        # [B, L, C] -> [B, L, N*C] -> [B, L, N, C]
        # 此时数据正式进入 4 条并行的残差流
        x = self.expand(x).view(B, L, self.n, self.dim)
        
        # 逐层穿过 mHC 模块
        for mhc, ffn in zip(self.mhc_blocks, self.ffn_layers):
            # 将 ffn 作为 callable 函数传给 mhc
            x = mhc(x, ffn)
            
        # 离开多数据流区域，将 [B, L, N, C] 压平为 [B, L, N*C]
        x = x.view(B, L, -1)
        
        # 合并回单主干道 [B, L, C]
        x = self.collapse(x)
        
        # 序列池化：取所有 Token 的平均特征 [B, C]
        x_pooled = x.mean(dim=1)
        
        # 输出预测结果 [B, out_classes]
        logits = self.head(x_pooled)
        return logits

# ==========================================
# 4. 测试与验证
# ==========================================
if __name__ == "__main__":
    # 超参数设置
    BATCH_SIZE = 8
    SEQ_LEN = 128
    VOCAB_SIZE = 1000
    DIM = 256
    N_STREAMS = 4
    NUM_LAYERS = 6
    NUM_CLASSES = 10

    # 实例化模型
    model = mHC_SequenceClassifier(
        vocab_size=VOCAB_SIZE, 
        dim=DIM, 
        n_streams=N_STREAMS, 
        num_layers=NUM_LAYERS, 
        out_classes=NUM_CLASSES
    )
    
    # 将模型扔到 GPU 上（如果你在 Ubuntu 环境中执行，可以开启）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 构造假输入数据
    dummy_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)
    
    # 前向传播
    print("🚀 开始前向传播...")
    outputs = model(dummy_input)
    print(f"✅ 输出张量维度: {outputs.shape} (预期: [{BATCH_SIZE}, {NUM_CLASSES}])")
    
    # 模拟一次反向传播
    print("\n🚀 模拟计算 Loss 并反向传播...")
    loss = outputs.sum()
    loss.backward()
    print("✅ 反向传播完成！")
    
    # 检查梯度是否正常（通过 Sinkhorn 约束，梯度应该非常稳定）
    grad_norm = model.mhc_blocks[-1].proj_res.weight.grad.norm().item()
    print(f"📊 最后一层 mHC proj_res 的梯度范数: {grad_norm:.4f}")
    
    grad_tensor = model.mhc_blocks[-1].proj_res.weight.grad
    print(f"📊 梯度最大绝对值: {grad_tensor.abs().max().item():.8f}")
    print(f"📊 梯度真实范数: {grad_tensor.norm().item():.8f}")