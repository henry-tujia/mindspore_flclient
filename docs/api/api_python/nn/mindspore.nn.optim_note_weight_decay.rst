在参数未分组时，优化器配置的 `weight_decay` 应用于名称含有"beta"或"gamma"的网络参数，通过网络参数分组可调整权重衰减策略。分组时，每组网络参数均可配置 `weight_decay` ，若未配置，则该组网络参数使用优化器中配置的 `weight_decay` 。

