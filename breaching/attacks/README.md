# Attack implementations

This module implements all attacks. A new attack should inherit from `base_attack._BaseAttacker` or at least follow the interface outlined there,
which requires that a `reconstruct` method is implemented which takes as input `server_payload` and  `shared_data` which are both lists of payloads and data outputs from `server.distribute_payload` and `user.compute_local_updates`. The attack should return a dictionary that contains the entries `data` and `labels`. Both should be PyTorch tensors that (possibly, depending on the success of the attack) approximate the immediate input to the user model.

Any new optimization-based attack can likely inherit a lot of functionality already present in `optimization_based_attack.OptimizationBasedAttacker`.

Implementing a new regularizer or objective requires no change to the main attacks, only another entry in the  `auxiliaries/regularizers.regularizer_lookup` or `auxiliaries/objectives.objective_lookup` interface respectively.

# 攻击实现

该模块实现了所有攻击。新的攻击应当继承自 `base_attack._BaseAttacker` 或至少遵循该接口，该接口要求实现一个 `reconstruct` 方法，输入参数为 `server_payload` 和 `shared_data`，这两个参数分别是从 `server.distribute_payload` 和 `user.compute_local_updates` 输出的负载和数据列表。该攻击应返回一个包含 `data` 和 `labels` 条目的字典。这两个条目应是 PyTorch 张量，且（根据攻击的成功与否）可能近似用户模型的即时输入。

任何新的基于优化的攻击可能会继承自 `optimization_based_attack.OptimizationBasedAttacker`，这样可以利用该类中已经存在的大量功能。

实现新的正则化器或目标函数不需要修改主要攻击，只需在 `auxiliaries/regularizers.regularizer_lookup` 或 `auxiliaries/objectives.objective_lookup` 接口中添加新的条目。