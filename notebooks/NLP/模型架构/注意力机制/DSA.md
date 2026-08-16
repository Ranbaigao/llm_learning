
[【手撕 DSA】 DeepSeek-V3.2 的 Sparse Attention 比 NSA 好在哪？](https://zhuanlan.zhihu.com/p/1957032283270812718)

Sparse Attention 的设计范式：
在计算注意力分数前，用低成本的方式筛选出与当前时刻 query 有价值的少量 KV， NSA 根据压缩注意力筛选 block-wise KV，DSA 使用索引( Indexer )网络筛选出 element-wise KV。