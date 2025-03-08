当然可以。以下是`reconstruct`函数的详细步骤说明：

1. **初始化攻击准备**：
    ```python
    rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)
    ```
    - 调用`prepare_attack`函数，准备攻击所需的模型、标签和统计信息。
    - `rec_models`：从服务器负载中构建的模型。
    - `labels`：从共享数据中提取的标签信息。
    - `stats`：用于存储攻击过程中的统计信息。

2. **初始化重建循环**：
    ```python
    scores = torch.zeros(self.cfg.restarts.num_trials)
    candidate_solutions = []
    ```
    - 初始化用于存储每次试验得分的张量`scores`，大小为`num_trials`。
    - 初始化候选解决方案列表`candidate_solutions`，用于存储每次试验的重建结果。

3. **重建试验循环**：
    ```python
    try:
        for trial in range(self.cfg.restarts.num_trials):
            candidate_solutions += [
                self._run_trial(rec_models, shared_data, labels, stats, trial, initial_data, dryrun)
            ]
            scores[trial] = self._score_trial(candidate_solutions[trial], labels, rec_models, shared_data)
    except KeyboardInterrupt:
        print("Trial procedure manually interruped.")
        pass
    ```
    - 进行多次重建试验，每次试验调用`_run_trial`函数执行重建，并将结果存储在`candidate_solutions`中。
    - 调用`_score_trial`函数对每次试验的结果进行评分，并将得分存储在`scores`中。
    - 如果手动中断试验过程，捕获`KeyboardInterrupt`异常并打印提示信息。

4. **选择最佳重建结果**：
    ```python
    optimal_solution = self._select_optimal_reconstruction(candidate_solutions, scores, stats)
    ```
    - 调用`_select_optimal_reconstruction`函数，从所有试验中选择得分最高的重建结果。

5. **处理重建数据**：
    ```python
    reconstructed_data = dict(data=optimal_solution, labels=labels)
    if server_payload[0]["metadata"].modality == "text":
        reconstructed_data = self._postprocess_text_data(reconstructed_data)
    if "ClassAttack" in server_secrets:
        true_num_data = server_secrets["ClassAttack"]["true_num_data"]
        reconstructed_data["data"] = torch.zeros([true_num_data, *self.data_shape], **self.setup)
        reconstructed_data["data"][server_secrets["ClassAttack"]["target_indx"]] = optimal_solution
        reconstructed_data["labels"] = server_secrets["ClassAttack"]["all_labels"]
    ```
    - 将最佳重建结果和标签存储在`reconstructed_data`字典中。
    - 如果数据类型是文本，则调用`_postprocess_text_data`函数进行后处理。
    - 如果存在`ClassAttack`的服务器秘密，则根据秘密信息调整重建数据。

6. **返回重建结果和统计信息**：
    ```python
    return reconstructed_data, stats
    ```
    - 返回最终的重建数据和统计信息。

通过上述步骤，`reconstruct`函数实现了从共享数据和服务器负载中重建用户数据的过程。

# `prepare_attack`

### **函数解析：`prepare_attack`**
该函数用于 **攻击准备**，主要目的是**解析服务器发送的 `server_payload` 及客户端训练后的 `shared_data`，构建恢复模型（rec_models），并处理梯度和标签信息**。这是许多 **重建攻击（Reconstruction Attack）** 方法的基础步骤。

---

## **1. 代码结构**
函数的主要流程：
1. **初始化统计信息**
2. **浅拷贝 `server_payload` 和 `shared_data`**
3. **加载 `metadata` 预处理常数**
4. **加载服务器参数并构造恢复模型**
5. **转换 `shared_data`**
6. **文本数据处理**
7. **恢复标签信息**
8. **梯度归一化（如果需要）**
9. **返回 `rec_models`、`labels` 和 `stats`**

---

## **2. 详细代码解析**

### **(1) 初始化统计信息**
```python
stats = defaultdict(list)
```
- **目的**：存储攻击过程中的统计数据，如损失值、梯度信息等。
- **使用 `defaultdict(list)`**：确保每个键都有默认的 `list`，避免 `KeyError`。

---

### **(2) 浅拷贝 `server_payload` 和 `shared_data`**
```python
shared_data = shared_data.copy()  # Shallow copy is enough
server_payload = server_payload.copy()
```
- **浅拷贝（shallow copy）**：
  - **拷贝 `shared_data` 和 `server_payload`**，避免对原始数据修改。
  - **浅拷贝 vs 深拷贝**：
    - **浅拷贝**：拷贝外层结构，内部对象仍然共享引用。
    - **深拷贝**（`deepcopy()`）：完全复制对象，子对象也是新实例。

---

### **(3) 解析 `metadata`**
```python
metadata = server_payload[0]["metadata"]
self.data_shape = metadata.shape
```
- `metadata` 来自 `server_payload`，通常包含：
  - `shape`：数据的形状
  - `mean/std`：数据归一化参数
  - `modality`：数据类型（如 `text`、`image`）
  
```python
if hasattr(metadata, "mean"):
    self.dm = torch.as_tensor(metadata.mean, **self.setup)[None, :, None, None]
    self.ds = torch.as_tensor(metadata.std, **self.setup)[None, :, None, None]
else:
    self.dm, self.ds = torch.tensor(0, **self.setup), torch.tensor(1, **self.setup)
```
- **目的**：加载数据 **均值 (`mean`) 和标准差 (`std`)** 以用于归一化。
- **数据归一化方式**：
  - 若 `metadata` **包含 `mean` 和 `std`**：
    - `self.dm`：形状 `[1, C, 1, 1]`（用于 `C` 维通道归一化）。
    - `self.ds`：同理。
  - 若 `metadata` **不包含 `mean` 和 `std`**：
    - 设为 `0` 和 `1`（不进行归一化）。

---

### **(4) 构造恢复模型**
```python
rec_models = self._construct_models_from_payload_and_buffers(server_payload, shared_data)
```
- **调用 `_construct_models_from_payload_and_buffers`**：
  - **参数**：
    - `server_payload`（包含服务器的初始参数）
    - `shared_data`（包含客户端训练后的梯度信息）
  - **作用**：
    - 可能用于构造一个 **攻击恢复模型**，用于重建输入数据。
  - **你需要提供 `_construct_models_from_payload_and_buffers` 代码**，否则无法确定具体逻辑。

---

### **(5) 处理 `shared_data`**
```python
shared_data = self._cast_shared_data(shared_data)
```
- **调用 `_cast_shared_data`**：
  - 可能涉及：
    - 将数据转换为特定格式（如 `torch.Tensor`）。
    - 调整数据类型（float32、float16）。
    - 适配不同设备（如 `CPU` / `GPU`）。
  - **你需要提供 `_cast_shared_data` 代码**。

---

### **(6) 处理文本数据**
```python
if metadata.modality == "text":
    rec_models, shared_data = self._prepare_for_text_data(shared_data, rec_models)
```
- **`metadata.modality`**：
  - `"text"` 表示数据是文本（如 `NLP` 任务）。
  - **不同数据类型可能需要不同处理方式**：
    - **图像**（`image`）：梯度恢复像素值。
    - **文本**（`text`）：梯度恢复单词或 token。
- **调用 `_prepare_for_text_data`**：
  - **作用**：
    - 可能对 `rec_models` 和 `shared_data` 进行特殊处理（如 `tokenizer`、`embedding`）。
  - **你需要提供 `_prepare_for_text_data` 代码**。

---

### **(7) 处理标签信息**
```python
if shared_data[0]["metadata"]["labels"] is None:
    labels = self._recover_label_information(shared_data, server_payload, rec_models)
else:
    labels = shared_data[0]["metadata"]["labels"].clone()
```
- **检查 `labels` 是否可用**：
  - 若 `shared_data` 中 `labels == None`：
    - 调用 `_recover_label_information` 进行恢复。
  - 若 `labels` 存在：
    - 直接 `clone()`，避免修改原始数据。

```python
labels = self._recover_label_information(shared_data, server_payload, rec_models)
```
- **调用 `_recover_label_information`**：
  - **作用**：
    - 可能基于 `gradients`、`rec_models` 等 **逆推标签**。
  - **你需要提供 `_recover_label_information` 代码**。

---

### **(8) 梯度归一化（如果需要）**
```python
if self.cfg.normalize_gradients:
    shared_data = self._normalize_gradients(shared_data)
```
- **配置选项**：`self.cfg.normalize_gradients`
  - `True`：对 `shared_data` 梯度进行归一化。
  - `False`：不归一化。
- **调用 `_normalize_gradients`**：
  - **可能作用**：
    - 归一化梯度，避免数值爆炸/梯度消失问题。
    - 可能用于 `LPIPS` 归一化、标准化梯度分布。
  - **你需要提供 `_normalize_gradients` 代码**。

---

### **(9) 返回攻击准备数据**
```python
return rec_models, labels, stats
```
- **`rec_models`**：攻击恢复的模型（用于梯度逆推输入）。
- **`labels`**：可能恢复出的标签信息。
- **`stats`**：统计信息，如攻击过程中的损失、梯度变化等。

---

## **3. 总结**
### **核心功能**
- 解析 `server_payload` 和 `shared_data`。
- 归一化 `metadata` 相关的均值/标准差。
- 构造攻击恢复模型（`rec_models`）。
- 处理不同 `modality`（如文本 vs 图像）。
- 尝试恢复 `labels`。
- 归一化梯度（可选）。

### **涉及的外部函数（你需要提供代码）**
1. `_construct_models_from_payload_and_buffers(server_payload, shared_data)`
2. `_cast_shared_data(shared_data)`
3. `_prepare_for_text_data(shared_data, rec_models)`
4. `_recover_label_information(shared_data, server_payload, rec_models)`
5. `_normalize_gradients(shared_data)`

---

## **4. 适用场景**
这个函数**主要用于联邦学习攻击**，可能涉及：
- **梯度反演攻击（Gradient Inversion Attack）**：利用梯度信息恢复客户端的训练数据。
- **模型逆向攻击（Model Reconstruction Attack）**：恢复模型参数或输入数据。
- **文本数据的梯度攻击**：针对 `NLP` 任务的文本恢复。

你可以提供缺失的函数代码，以便进一步分析整个攻击流程。


### **函数解析：`_run_trial`**
该函数 **执行一次恢复（reconstruction）尝试**，主要用于 **攻击者在梯度信息（`shared_data`）的基础上恢复客户端的原始数据**。其核心目标是：
1. **初始化正则化项与损失函数**。
2. **创建数据恢复的初始状态**（可选地从 `initial_data` 继续优化）。
3. **优化迭代**：
   - 计算目标函数（`objective`）。
   - 使用 **梯度下降** 方法优化恢复的数据。
   - 施加 **边界约束**（`projection`），防止恢复数据超出范围。
   - 记录恢复过程中的 **最优数据**。
4. **终止条件**：
   - 达到最大迭代次数。
   - 目标函数值变得 **非有限**（如 `NaN`）。
   - `dryrun=True` 时提前终止。

---

## **1. 代码结构**
```python
def _run_trial(self, rec_model, shared_data, labels, stats, trial, initial_data=None, dryrun=False):
    """Run a single reconstruction trial."""
```
**函数参数**：
- **`rec_model`**：恢复模型（从 `_construct_models_from_payload_and_buffers` 得到）。
- **`shared_data`**：联邦学习客户端的 **梯度信息**，用于反推输入数据。
- **`labels`**：已恢复的标签信息（如果已知）。
- **`stats`**：存储恢复过程的统计数据。
- **`trial`**：当前尝试编号（用于多次实验）。
- **`initial_data`**（可选）：初始化数据（如已部分恢复的数据）。
- **`dryrun`**（默认 `False`）：如果为 `True`，则运行一次并立即停止（用于测试代码）。

---

## **2. 详细代码解析**

### **(1) 初始化损失函数和正则项**
```python
for regularizer in self.regularizers:
    regularizer.initialize(rec_model, shared_data, labels)
self.objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])
```
- **遍历 `self.regularizers`**：
  - **调用 `initialize(rec_model, shared_data, labels)`** 进行初始化。
  - `regularizer` 可能用于 **梯度正则化** 或 **重建数据的额外约束**（如 TV 先验）。
  
- **初始化目标函数 `self.objective`**：
  - `self.loss_fn`：损失函数（如 `MSELoss`、`CrossEntropyLoss`）。
  - `self.cfg.impl`：实现方式（可能决定计算方式，如 `PyTorch` vs `JIT`）。
  - `shared_data[0]["metadata"]["local_hyperparams"]`：本地超参数（如学习率、批次大小）。

❗ **你需要提供 `self.objective.initialize()` 和 `regularizer.initialize()` 的代码，才能进一步分析它们的作用。**

---

### **(2) 初始化恢复数据**
```python
candidate = self._initialize_data([shared_data[0]["metadata"]["num_data_points"], *self.data_shape])
if initial_data is not None:
    candidate.data = initial_data.data.clone().to(**self.setup)
```
- **调用 `_initialize_data()`**：
  - **创建一个 `candidate`（恢复数据的变量）**。
  - 形状为 `[num_data_points, *self.data_shape]`，即与客户端数据形状匹配。

- **如果 `initial_data` 存在**：
  - 直接复制已有数据（`initial_data.data.clone().to(**self.setup)`）。
  - **这样可以在已有的恢复结果上继续优化**。

📌 **你需要提供 `_initialize_data()` 代码，以确定它如何初始化数据（是否随机？是否基于 `shared_data`）**。

---

### **(3) 记录当前最优恢复结果**
```python
best_candidate = candidate.detach().clone()
minimal_value_so_far = torch.as_tensor(float("inf"), **self.setup)
```
- **`best_candidate`**：存储 **最优恢复数据**（用于选取最小损失时的 `candidate`）。
- **`minimal_value_so_far`**：初始化为 **正无穷**，用于跟踪最小损失。

---

### **(4) 初始化优化器和调度器**
```python
optimizer, scheduler = self._init_optimizer([candidate])
```
- **调用 `_init_optimizer([candidate])`**：
  - 可能返回：
    - `optimizer`（优化器，如 `Adam`、`SGD`）。
    - `scheduler`（学习率调度器）。
  
📌 **你需要提供 `_init_optimizer()` 代码，以确定优化器的配置方式**。

---

### **(5) 进入优化迭代**
```python
current_wallclock = time.time()
try:
    for iteration in range(self.cfg.optim.max_iterations):
```
- **迭代 `max_iterations` 次**，优化 `candidate`。
- **记录开始时间**（`current_wallclock`），用于计算时间消耗。

---

### **(6) 计算目标函数并优化**
```python
closure = self._compute_objective(candidate, labels, rec_model, optimizer, shared_data, iteration)
objective_value, task_loss = optimizer.step(closure), self.current_task_loss
scheduler.step()
```
- **调用 `_compute_objective()`**：
  - 计算恢复损失（可能基于 `MSE` 或 `KL 散度`）。
  - 返回 `closure` 供 `optimizer.step(closure)` 计算梯度。
- **优化步骤**：
  - `optimizer.step(closure)`：执行梯度更新。
  - `scheduler.step()`：调整学习率。

📌 **你需要提供 `_compute_objective()` 代码，以确定如何计算恢复损失**。

---

### **(7) 投影到合法值范围**
```python
with torch.no_grad():
    if self.cfg.optim.boxed:
        candidate.data = torch.max(torch.min(candidate, (1 - self.dm) / self.ds), -self.dm / self.ds)
```
- **作用**：
  - 若 `self.cfg.optim.boxed = True`，则 **限制恢复数据 `candidate` 在合法范围内**。
  - 例如：
    - `image` 数据通常在 `[0, 1]` 之间。
    - `text embedding` 可能有特定范围约束。

---

### **(8) 记录最优恢复数据**
```python
if objective_value < minimal_value_so_far:
    minimal_value_so_far = objective_value.detach()
    best_candidate = candidate.detach().clone()
```
- **如果当前 `objective_value` 更优**：
  - 记录最小损失 `minimal_value_so_far`。
  - 更新 `best_candidate`。

---

### **(9) 记录日志**
```python
if iteration + 1 == self.cfg.optim.max_iterations or iteration % self.cfg.optim.callback == 0:
    timestamp = time.time()
    log.info(
        f"| It: {iteration + 1} | Rec. loss: {objective_value.item():2.4f} | "
        f" Task loss: {task_loss.item():2.4f} | T: {timestamp - current_wallclock:4.2f}s"
    )
    current_wallclock = timestamp
```
- **每 `callback` 轮打印日志**：
  - **恢复损失 `objective_value`**。
  - **任务损失 `task_loss`**（可能是原任务的损失）。
  - **时间消耗 `T`**。

---

### **(10) 终止条件**
```python
if not torch.isfinite(objective_value):
    log.info(f"Recovery loss is non-finite in iteration {iteration}. Cancelling reconstruction!")
    break
```
- **若 `objective_value` 变为 `NaN` 或 `Inf`**，则提前终止。

---

### **(11) 记录统计信息**
```python
stats[f"Trial_{trial}_Val"].append(objective_value.item())
```
- 记录当前 `trial` 过程中的 `objective_value`。

---

### **(12) 处理 `dryrun`**
```python
if dryrun:
    break
```
- 若 `dryrun=True`，则只运行一次并退出。

---

### **(13) 处理中断**
```python
except KeyboardInterrupt:
    print(f"Recovery interrupted manually in iteration {iteration}!")
    pass
```
- 支持 **手动终止恢复**（`Ctrl+C`）。

---

### **(14) 返回最佳恢复结果**
```python
return best_candidate.detach()
```
- 返回 **最优恢复数据 `best_candidate`**。

---

## **3. 总结**
**`_run_trial` 的主要功能**：
- **优化恢复数据 `candidate`**，使其逼近原始客户端数据。
- **使用 `shared_data` 和 `labels`** 计算目标函数。
- **优化步骤**：
  - `optimizer.step(closure)` 计算梯度并更新 `candidate`。
  - `scheduler.step()` 调整学习率。
- **应用约束（如 `boxed` 限制）** 以确保恢复数据合理。
- **记录日志和恢复损失**，确保实验可复现。

📌 **你需要提供以下函数的代码，以完成整体分析**：
1. `self.objective.initialize()`
2. `_initialize_data()`
3. `_init_optimizer()`
4. `_compute_objective()`

---

### **函数解析：`_initialize_data`**
该函数用于 **初始化恢复数据 `candidate`**，在攻击（如梯度反演攻击）中，`candidate` 代表攻击者 **试图恢复的原始数据**。它的初始化方式会影响攻击的收敛速度和最终恢复的质量。

---

## **1. 代码结构**
```python
def _initialize_data(self, data_shape):
```
**参数**：
- **`data_shape`**：数据形状，通常为 `[batch_size, channels, height, width]`（图像）或 `[batch_size, seq_len]`（文本）。

**返回值**：
- **`candidate`**：一个 `torch.Tensor`，包含恢复数据的初始值，且 **开启梯度计算**（`requires_grad=True`）。

---

## **2. 详细代码解析**

### **(1) 读取初始化方式**
```python
init_type = self.cfg.init
```
- `self.cfg.init` 决定初始化方式，例如：
  - `randn`（正态分布）
  - `rand`（均匀分布）
  - `zeros`（全零）
  - `patterned`（特定模式）
  - `wei`（Wei 等人提出的方法）
  - `red`, `green`, `blue`, `dark`, `light`（颜色通道特定初始化）

---

### **(2) 常见随机初始化**
```python
if init_type == "randn":
    candidate = torch.randn(data_shape, **self.setup)
elif init_type == "randn-trunc":
    candidate = (torch.randn(data_shape, **self.setup) * 0.1).clamp(-0.1, 0.1)
elif init_type == "rand":
    candidate = (torch.rand(data_shape, **self.setup) * 2) - 1.0
elif init_type == "zeros":
    candidate = torch.zeros(data_shape, **self.setup)
```
- **`randn`**（正态分布）：`N(0,1)`
- **`randn-trunc`**（截断正态分布）：`N(0, 0.1)`，裁剪到 `[-0.1, 0.1]` 范围
- **`rand`**（均匀分布）：`[-1,1]`
- **`zeros`**（全零）：`0`

这些方式主要用于**图像或文本嵌入恢复攻击**，通过梯度优化调整 `candidate` 逼近真实数据。

---

### **(3) 颜色通道特定初始化**
```python
elif any(c in init_type for c in ["red", "green", "blue", "dark", "light"]):
    candidate = torch.zeros(data_shape, **self.setup)
    if "light" in init_type:
        candidate = torch.ones(data_shape, **self.setup)
    else:
        nonzero_channel = 0 if "red" in init_type else 1 if "green" in init_type else 2
        candidate[:, nonzero_channel, :, :] = 1
    if "-true" in init_type:
        candidate = (candidate - self.dm) / self.ds
```
- **适用于 RGB 图像数据**：
  - `"red"`：红色通道设为 `1`，其他通道 `0`。
  - `"green"`：绿色通道设为 `1`，其他通道 `0`。
  - `"blue"`：蓝色通道设为 `1`，其他通道 `0`。
  - `"dark"`：初始化为 **全零**。
  - `"light"`：初始化为 **全一**。

- **`"-true"` 选项**：
  - 若 `init_type` 结尾包含 `-true`，则 **应用 `self.dm` 和 `self.ds` 进行归一化**。

---

### **(4) 特定模式初始化**
```python
elif "patterned" in init_type:
    pattern_width = int("".join(filter(str.isdigit, init_type)))
    if "randn" in init_type:
        seed = torch.randn([data_shape[0], 3, pattern_width, pattern_width], **self.setup)
    elif "rand" in init_type:
        seed = (torch.rand([data_shape[0], 3, pattern_width, pattern_width], **self.setup) * 2) - 1
    else:
        seed = torch.randn([data_shape[0], 3, pattern_width, pattern_width], **self.setup)
    
    x_factor, y_factor = (
        torch.as_tensor(data_shape[2] / pattern_width).ceil(),
        torch.as_tensor(data_shape[3] / pattern_width).ceil(),
    )
    candidate = (
        torch.tile(seed, (1, 1, int(x_factor), int(y_factor)))[:, :, : data_shape[2], : data_shape[3]]
        .contiguous()
        .clone()
    )
```
- **适用于图像数据**
- **思路**：
  1. 生成一个 **`pattern_width x pattern_width`** 大小的随机噪声图案（`seed`）。
  2. **重复扩展 `seed`** 使其匹配 `data_shape`。
  3. **使用 `.tile()`** 方法进行扩展，使其具有 **规则模式**。

- **适用场景**：
  - **隐私保护评估**（Wei et al., 2020 提出的评估框架）。
  - **可视化攻击效果**，通过模式初始化观察梯度反演效果。

---

### **(5) `wei` 初始化**
```python
elif "wei" in init_type:
    pattern_width = int("".join(filter(str.isdigit, init_type)))
    if "rand" in init_type:
        seed = (torch.rand([data_shape[0], 3, pattern_width, pattern_width], **self.setup) * 2) - 1
    else:
        seed = torch.randn([data_shape[0], 3, pattern_width, pattern_width], **self.setup)
    
    x_factor, y_factor = (
        torch.as_tensor(data_shape[2] / pattern_width).ceil(),
        torch.as_tensor(data_shape[3] / pattern_width).ceil(),
    )
    candidate = (
        torch.tile(seed, (1, 1, int(x_factor), int(y_factor)))[:, :, : data_shape[2], : data_shape[3]]
        .contiguous()
        .clone()
    )
```
- **类似 `patterned` 初始化**，但可能用于不同实验。
- **目的**：
  - 生成 **局部一致的初始化**，测试不同攻击方案对初始化的敏感性。

---

### **(6) 处理未知初始化方式**
```python
else:
    raise ValueError(f"Unknown initialization scheme {init_type} given.")
```
- **若 `init_type` 不是已知类型**，抛出错误。

---

### **(7) 使 `candidate` 可训练**
```python
candidate.to(memory_format=self.memory_format)
candidate.requires_grad = True
candidate.grad = torch.zeros_like(candidate)
```
- **确保 `candidate` 在 GPU/CPU 上的正确格式**。
- **开启梯度计算**：
  - `candidate.requires_grad = True`，这样梯度反演攻击可以优化它。
- **初始化梯度**：
  - `candidate.grad = torch.zeros_like(candidate)` 确保梯度不会是 `None`。

---

## **3. 总结**
### **主要功能**
- **初始化恢复数据 `candidate`**，作为攻击目标。
- **支持多种初始化方式**：
  - **随机初始化**（`randn`, `rand`, `zeros`）
  - **RGB 颜色初始化**（`red`, `green`, `blue`）
  - **模式化初始化**（`patterned`, `wei`）
- **保证 `candidate` 可训练**：
  - 设定 `requires_grad=True`，支持梯度优化。

### **适用场景**
- **梯度反演攻击**：尝试恢复图像/文本输入。
- **隐私保护评估**：测试不同初始化对攻击效果的影响。
- **数据恢复优化**：结合 `_run_trial()` 进行优化，使 `candidate` 逼近真实数据。

📌 **你可以提供 `self.cfg.init` 的可能值，确认是否有其他特殊初始化方式。**