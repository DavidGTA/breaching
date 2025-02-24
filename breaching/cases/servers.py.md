## **📌 代码解析：Honest-but-Curious 服务器实现**

### **🔹 代码作用**
该代码实现了 **联邦学习（Federated Learning, FL）服务器**，其中包含：
- **服务器类型选择 (`construct_server`)**：根据不同的攻击模型（Honest 或 Malicious）选择服务器类型。
- **`HonestServer` 类**：实现**诚实但好奇（Honest-but-Curious）**服务器。
  - 负责 **发送模型参数** 给用户（`distribute_payload`）。
  - 负责 **执行联邦学习协议**，模拟用户更新数据（`run_protocol`）。
  - **不会恶意篡改数据**，但可能会分析用户上传的信息（**好奇但不主动攻击**）。

---

## **1. `construct_server()`：服务器类型选择**
```python
def construct_server(
    model, loss_fn, cfg_case, setup=dict(device=torch.device("cpu"), dtype=torch.float), external_dataloader=None
):
    """服务器接口函数，根据不同的服务器类型创建服务器实例。"""
```
- **输入**
  - `model`：用于联邦学习的神经网络模型
  - `loss_fn`：损失函数
  - `cfg_case`：当前实验的配置
  - `setup`：设备（CPU/GPU）和数据类型（float）
  - `external_dataloader`：外部数据加载器（若 `cfg_case.server.has_external_data=True`）

- **逻辑**
  ```python
  if cfg_case.server.name == "honest_but_curious":
      server = HonestServer(model, loss_fn, cfg_case, setup, external_dataloader=dataloader)
  elif cfg_case.server.name == "malicious_model":
      server = MaliciousModelServer(model, loss_fn, cfg_case, setup, external_dataloader=dataloader)
  elif cfg_case.server.name == "class_malicious_parameters":
      server = MaliciousClassParameterServer(model, loss_fn, cfg_case, setup, external_dataloader=dataloader)
  elif cfg_case.server.name == "malicious_transformer_parameters":
      server = MaliciousTransformerServer(model, loss_fn, cfg_case, setup, external_dataloader=dataloader)
  else:
      raise ValueError(f"Invalid server type {cfg_case.server} given.")
  return server
  ```
  - **如果 `server.name` 是 `"honest_but_curious"`**，则返回 `HonestServer` 实例（**当前使用的服务器类型**）。
  - **如果是其他恶意服务器类型**（`malicious_model` 等），则返回不同的 `MaliciousServer` 变体（**当前未使用**）。

---

## **2. `HonestServer` 类：诚实但好奇的服务器**
```python
class HonestServer:
    """实现诚实但好奇（Honest-but-curious）服务器协议。

    该服务器会：
    1. 选择并加载初始模型，并将其发送给用户
    2. 运行联邦学习协议，接收用户梯度更新
    3. 由于是 "诚实但好奇" 的服务器，不会恶意修改数据，但可能会分析上传的信息
    """
```

### **🔹 主要属性**
```python
THREAT = "Honest-but-curious"
```
- `THREAT`：服务器的 **威胁模型**，即 **“诚实但好奇”**（Honest-but-Curious）。
- **它不会主动攻击用户数据，但会分析用户上传的梯度信息**（即潜在的隐私风险）。

```python
def __init__(self, model, loss, cfg_case, setup=dict(dtype=torch.float, device=torch.device("cpu")), external_dataloader=None):
```
- **初始化服务器**
  - `model`：深度学习模型
  - `loss`：损失函数
  - `cfg_case`：实验配置
  - `setup`：设备（CPU/GPU）
  - `external_dataloader`：外部数据加载器（如果 `server.has_external_data=True`）

```python
self.model = model
self.model.eval()
self.loss = loss
self.setup = setup
self.num_queries = cfg_case.server.num_queries
self.cfg_data = cfg_case.data
self.cfg_server = cfg_case.server
self.external_dataloader = external_dataloader
self.secrets = dict()  # 该服务器不存储任何秘密信息
```
- **模型初始化**
  - `self.model.eval()`：将模型设为 **评估模式**（`eval`），避免梯度更新。
  - `self.num_queries`：服务器允许的 **最大查询次数**（即训练轮数）。
  - `self.secrets = dict()`：**由于服务器是诚实的，它不存储任何秘密数据**。

---

### **📌 `__repr__` 方法的作用**
在 Python 中，`__repr__` 方法用于 **返回对象的字符串表示**，通常用于 **调试和日志记录**，帮助开发者更方便地查看对象的状态。

---

### **🔹 作用**
1. **当调用 `print(server)` 或 `repr(server)` 时，会返回 `__repr__` 方法的字符串**。
2. **方便调试**：可以快速查看服务器的 **类型、威胁模型、查询次数、模型信息、隐私数据等** 关键信息。
3. **提高可读性**：比默认的 `<HonestServer object at 0x1234>` 更清晰易懂。

---

### **📌 `__repr__` 方法解析**
```python
def __repr__(self):
    return f"""Server (of type {self.__class__.__name__}) with settings:
    Threat model: {self.THREAT}
    Number of planned queries: {self.num_queries}
    Has external/public data: {self.cfg_server.has_external_data}

    Model:
        model specification: {str(self.model.name)}
        model state: {self.cfg_server.model_state}
        {f'public buffers: {self.cfg_server.provide_public_buffers}' if len(list(self.model.buffers())) > 0 else ''}

    Secrets: {self.secrets}
    """
```

---

### **🔹 代码解析**
#### **1. 输出服务器类型**
```python
Server (of type {self.__class__.__name__}) with settings:
```
- **`self.__class__.__name__`** 获取当前实例的类名，如 `HonestServer`。
- **示例输出**：
  ```plaintext
  Server (of type HonestServer) with settings:
  ```

#### **2. 显示服务器的威胁模型**
```python
Threat model: {self.THREAT}
```
- **`self.THREAT`** 是服务器的威胁类型，例如 `"Honest-but-curious"`。
- **示例输出**：
  ```plaintext
  Threat model: Honest-but-curious
  ```

#### **3. 服务器允许的最大查询次数**
```python
Number of planned queries: {self.num_queries}
```
- **`self.num_queries`** 指定该服务器在 **一个实验中可以处理的最大查询次数**。
- **示例输出**：
  ```plaintext
  Number of planned queries: 10
  ```

#### **4. 服务器是否使用外部数据**
```python
Has external/public data: {self.cfg_server.has_external_data}
```
- **`self.cfg_server.has_external_data`** 是一个布尔值，指示服务器是否有额外的数据集可供使用。
- **示例输出**：
  ```plaintext
  Has external/public data: False
  ```

#### **5. 输出模型信息**
```python
Model:
    model specification: {str(self.model.name)}
    model state: {self.cfg_server.model_state}
```
- **`self.model.name`**：模型的名称，如 `"resnet18"`。
- **`self.cfg_server.model_state`**：模型的当前状态（如 `"trained"`、`"untrained"`）。
- **示例输出**：
  ```plaintext
  Model:
      model specification: resnet18
      model state: trained
  ```

#### **6. 是否提供 BatchNorm 的 `buffers`**
```python
{f'public buffers: {self.cfg_server.provide_public_buffers}' if len(list(self.model.buffers())) > 0 else ''}
```
- **如果 `self.model` 具有 `buffers`（如 BatchNorm 的 `running_mean`），则显示 `public buffers` 选项**。
- **示例输出**：
  ```plaintext
  public buffers: True
  ```
  如果 `model.buffers()` 为空，则不会打印这行内容。

#### **7. 显示服务器存储的“秘密”**
```python
Secrets: {self.secrets}
```
- **由于 `HonestServer` 是诚实的，它的 `self.secrets` 为空 `dict()`**。
- **示例输出**：
  ```plaintext
  Secrets: {}
  ```

---

### **📌 运行示例**
假设我们有一个 `HonestServer` 对象：
```python
server = HonestServer(model, loss, cfg_case)
print(server)
```
如果 `server` 的配置如下：
```python
self.num_queries = 10
self.THREAT = "Honest-but-curious"
self.cfg_server.has_external_data = False
self.model.name = "resnet18"
self.cfg_server.model_state = "trained"
self.cfg_server.provide_public_buffers = True
self.secrets = {}
```
那么 `print(server)` 的输出如下：
```plaintext
Server (of type HonestServer) with settings:
    Threat model: Honest-but-curious
    Number of planned queries: 10
    Has external/public data: False

    Model:
        model specification: resnet18
        model state: trained
        public buffers: True

    Secrets: {}
```

---

### **📌 结论**
✅ **`__repr__` 使 `HonestServer` 的信息更易读**  
✅ **有助于调试，快速查看服务器的配置信息**  
✅ **在 `print(server)` 或 `repr(server)` 时，会自动调用 `__repr__` 方法**  

💡 **总结：这个 `__repr__` 方法提供了服务器的** **"自述" 信息**，有助于理解当前服务器的状态，尤其是在联邦学习场景下！ 🚀

## **3. `reconfigure_model()`：根据不同状态重新配置模型**
```python
def reconfigure_model(self, model_state, query_id=0):
```
- **作用**：根据 `model_state` 重新初始化模型
- **支持的 `model_state`**
  - `"untrained"`：重置模型参数
  - `"trained"`：保持预训练状态
  - `"linearized"`：对 `BatchNorm` 和 `Conv2d` 层进行特殊修改
  - `"orthogonal"`：使用**正交初始化**模型参数
  - `"unchanged"`：保持不变

**部分代码解析**
```python
if model_state == "untrained":
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
elif model_state == "linearized":
    with torch.no_grad():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.weight.data = module.running_var.data.clone()
            module.bias.data = module.running_mean.data.clone() + 10
        if isinstance(module, torch.nn.Conv2d) and hasattr(module, "bias"):
            module.bias.data += 10
elif model_state == "orthogonal":
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    if "conv" in name or "linear" in name:
        torch.nn.init.orthogonal_(module.weight, gain=1)
```
- `linearized`：对 `BatchNorm2d` 进行特殊初始化
- `orthogonal`：**使用正交初始化**，有助于保持梯度稳定

---

## **4. `distribute_payload()`：服务器向用户发送模型参数**
```python
def distribute_payload(self, query_id=0):
```
- **作用**：服务器在 **每轮训练开始前** 发送模型参数给用户。
- **步骤**
  - 重新配置模型（`reconfigure_model`）。
  - 发送模型参数和（可选的）BatchNorm 缓冲区。

**核心代码**
```python
honest_model_parameters = [p for p in self.model.parameters()]  # 发送模型参数
if self.cfg_server.provide_public_buffers:
    honest_model_buffers = [b for b in self.model.buffers()]
else:
    honest_model_buffers = None
return dict(parameters=honest_model_parameters, buffers=honest_model_buffers, metadata=self.cfg_data)
```
- **如果 `provide_public_buffers=True`，则发送 `BatchNorm` 缓冲区**（例如 `running_mean`）。
- **返回的数据**
  ```python
  {
      "parameters": [模型参数列表],
      "buffers": [可选的 BatchNorm 缓冲区],
      "metadata": [数据集配置]
  }
  ```

---

## **5. `run_protocol()`：服务器运行 FL 训练协议**
```python
def run_protocol(self, user):
```
- **作用**：模拟联邦学习协议，向用户发送模型，并获取用户更新。
- **核心流程**
  ```python
  shared_user_data = []
  payloads = []
  for query_id in self.queries():
      server_payload = self.distribute_payload(query_id)
      shared_data_per_round, true_user_data = user.compute_local_updates(server_payload)
      payloads.append(server_payload)
      shared_user_data.append(shared_data_per_round)
  return shared_user_data, payloads, true_user_data
  ```
  - **发送模型 (`distribute_payload`)**
  - **用户计算本地更新 (`user.compute_local_updates`)**
  - **收集用户的更新数据 `shared_user_data`**
  - **收集服务器的 `payloads`**

---

## **📌 总结**
✅ **`HonestServer`（诚实但好奇）**
- **不会主动篡改模型参数**
- **会收集用户上传的梯度更新**，但不进行恶意利用

✅ **`construct_server()`**
- 选择不同的服务器（`HonestServer` 或 `MaliciousServer`）

✅ **`distribute_payload()`**
- 服务器向用户发送 **模型参数**

✅ **`run_protocol()`**
- **执行联邦学习协议**，获取用户上传的更新

💡 **总结：该服务器是“诚实但好奇”的，不会主动攻击用户，但仍可能分析梯度，造成隐私泄露风险！** 🚀