# Cases

This module implements the core functionality for each use case. `users.py` implements the user-side protocol for different types of users (`single_gradient`, multiple `local_updates` and aggregations of such users as `multiuser_aggregate`). `servers.py` implements a range of server threat models from `honest-but-curious` servers to `malicious-model` and `malicious-parameters` variants.

If you are looking specifically for the implementation of the modification necessary for "Robbing-the-Fed", take a look at `malicious_modifications/imprint.py`. If you are looking for "Decepticons", look at `malicious_modifications/analytic_transformer_utils.py`.

# 用例

这个模块实现了每个用例的核心功能。`users.py` 实现了不同类型用户的用户端协议（如 `single_gradient`、多个 `local_updates` 以及这些用户的聚合形式 `multiuser_aggregate`）。`servers.py` 实现了多个服务器威胁模型，从 `honest-but-curious`（诚实但好奇）到 `malicious-model`（恶意模型）和 `malicious-parameters`（恶意参数）变体。

如果你专门寻找为 "Robbing-the-Fed" 需要的修改实现，可以查看 `malicious_modifications/imprint.py`。如果你寻找的是 "Decepticons" 的实现，可以查看 `malicious_modifications/analytic_transformer_utils.py`。