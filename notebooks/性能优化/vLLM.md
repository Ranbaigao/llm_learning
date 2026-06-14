
# VLLM

## PageAttention
[图解大模型计算加速系列之：vLLM核心技术PagedAttention原理](https://zhuanlan.zhihu.com/p/691038809)

## 实操

linux
```shell
CUDA_VISIBLE_DEVICES=0 nohup vllm serve E:\\Models\\Qwen3.5-2B-AWQ-4bit \
  --port 8009 \
  --tensor-parallel-size 1 \
  --max-model-len 32000 \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
  --served-model-name qwen3.5-2b \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.5
```

powershell:
```shell
$env:CUDA_VISIBLE_DEVICES="0"
vllm serve "E:\Models\Qwen3.5-2B-AWQ-4bit" `
  --port 8009 `
  --tensor-parallel-size 1 `
  --max-model-len 32000 `
  --speculative-config "{\`"method\`":\`"qwen3_next_mtp\`",\`"num_speculative_tokens\`":2}" `
  --served-model-name qwen3.5-2b `
  --reasoning-parser qwen3 `
  --gpu-memory-utilization 0.5
```


```shell
vllm serve "E:\Models\Qwen3.5-2B-AWQ-4bit" ^
  --port 8009 ^
  --tensor-parallel-size 1 ^
  --max-model-len 10000 ^
  --served-model-name qwen3.5-2b ^
  --reasoning-parser qwen3 ^
  --default-chat-template-kwargs "{\"enable_thinking\": true}" ^
  --gpu-memory-utilization 0.7
```