## **📌 `_build_dataset_vision()` 代码解析**

### **🔹 作用**
该函数用于 **构造计算机视觉（vision）数据集**，并返回：
1. **已加载的 `dataset`**
2. **数据批处理函数 `collate_fn`**（用于 `DataLoader`）

---

## **📌 代码结构**
```python
def _build_dataset_vision(cfg_data, split, can_download=True):
```
- **输入参数**
  - `cfg_data`：包含数据集名称、路径、是否标准化等配置信息。
  - `split`：数据集划分（如 `"training"`、`"validation"`）。
  - `can_download`：是否允许自动下载数据集。

- **返回值**
  - `dataset`：加载的数据集（`torchvision.datasets`）。
  - `collate_fn`：数据批处理函数。

---

## **1. 设置默认的 `ToTensor()` 转换**
```python
_default_t = torchvision.transforms.ToTensor()
```
- 该转换 **将 PIL 图片转换为 `torch.Tensor`**。
- **如果后续未指定额外的图像增强转换，则默认使用 `ToTensor()`。**

---

## **2. 处理 `cfg_data.path`**
```python
cfg_data.path = os.path.expanduser(cfg_data.path)
```
- **展开 `~` 为用户目录路径**，保证路径正确。

---

## **3. 选择不同的数据集**
### **📌 3.1 处理 `CIFAR10`**
```python
if cfg_data.name == "CIFAR10":
    dataset = torchvision.datasets.CIFAR10(
        root=cfg_data.path, train=split == "training", download=can_download, transform=_default_t,
    )
    dataset.lookup = dict(zip(list(range(len(dataset))), dataset.targets))
```
- **使用 `torchvision.datasets.CIFAR10` 直接加载 CIFAR-10**。
- `train=split == "training"`：
  - **如果 `split="training"`**，则加载训练集；
  - 否则加载测试集（`train=False`）。
- **数据 `lookup` 机制**
  ```python
  dataset.lookup = dict(zip(list(range(len(dataset))), dataset.targets))
  ```
  - **构造索引到类别标签的映射**，方便后续处理。

---

### **📌 3.2 处理 `CIFAR100`**
```python
elif cfg_data.name == "CIFAR100":
    dataset = torchvision.datasets.CIFAR100(
        root=cfg_data.path, train=split == "training", download=can_download, transform=_default_t,
    )
    dataset.lookup = dict(zip(list(range(len(dataset))), dataset.targets))
```
- 逻辑与 `CIFAR10` **完全相同**，只是数据集换成 `CIFAR100`。

---

### **📌 3.3 处理 `ImageNet`**
```python
elif cfg_data.name == "ImageNet":
    dataset = torchvision.datasets.ImageNet(
        root=cfg_data.path, split="train" if "train" in split else "val", transform=_default_t,
    )
    dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset.samples]))
```
- `split="train"` 或 `"val"` 确定加载训练集或验证集。
- **`dataset.samples` 是 `(图片路径, 类别标签)` 的列表**。
- `lookup` 记录 **索引到类别的映射**：
  ```python
  dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset.samples]))
  ```

---

### **📌 3.4 处理 `ImageNetAnimals`**
```python
elif cfg_data.name == "ImageNetAnimals":
    dataset = torchvision.datasets.ImageNet(
        root=cfg_data.path, split="train" if "train" in split else "val", transform=_default_t,
    )
    dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset.samples]))
    indices = [idx for (idx, label) in dataset.lookup.items() if label < 397]
    dataset.classes = dataset.classes[:397]
    dataset.samples = [dataset.samples[i] for i in indices]
    dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset.samples]))
```
- **基于 ImageNet 过滤 `label < 397` 的类别**，用于 **子集实验**（仅保留前 397 类）。
- **调整 `dataset.classes` 和 `dataset.samples`**，使其与 `lookup` 一致。

---

### **📌 3.5 处理 `TinyImageNet`**
```python
elif cfg_data.name == "TinyImageNet":
    dataset = TinyImageNet(
        root=cfg_data.path, split=split, download=can_download, transform=_default_t, cached=True,
    )
    dataset.lookup = dict(zip(list(range(len(dataset))), dataset.targets))
```
- **TinyImageNet** 是一个更小的 ImageNet 变体（200 类，每类 500 张图）。
- `cached=True` 可能 **启用缓存，提高加载效率**。

---

### **📌 3.6 处理 `Birdsnap`**
```python
elif cfg_data.name == "Birdsnap":
    dataset = Birdsnap(root=cfg_data.path, split=split, download=can_download, transform=_default_t)
    dataset.lookup = dict(zip(list(range(len(dataset))), dataset.labels))
```
- **Birdsnap 是鸟类分类数据集**，`labels` 存储类别索引。

---

### **📌 3.7 数据集不存在时抛出错误**
```python
else:
    raise ValueError(f"Invalid dataset {cfg_data.name} provided.")
```
- 如果 `cfg_data.name` 不是支持的数据集，则抛出 `ValueError`。

---

## **4. 计算数据均值和标准差**
```python
if cfg_data.mean is None and cfg_data.normalize:
    data_mean, data_std = _get_meanstd(dataset)
    cfg_data.mean = data_mean
    cfg_data.std = data_std
```
- **如果 `cfg_data.mean` 未定义且需要标准化，则自动计算均值和标准差**。
- `mean` 和 `std` 可用于后续 **图像标准化预处理**。

---

## **5. 解析数据增强**
```python
transforms = _parse_data_augmentations(cfg_data, split)
```
- **解析数据增强**（如 `RandomCrop`、`HorizontalFlip`）。
- **若 `cfg_data.augmentations` 配置了数据增强，则返回对应的 `transforms`**。

---

## **6. 应用数据变换**
```python
dataset.transform = transforms if transforms is not None else None
```
- **如果 `transforms` 存在，则应用到 `dataset.transform`**。
- **否则，默认使用 `ToTensor()`**。

---

## **7. 存储均值和标准差**
```python
if cfg_data.normalize:
    dataset.mean = cfg_data.mean
    dataset.std = cfg_data.std
else:
    dataset.mean = [0]
    dataset.std = [1]
```
- **如果 `normalize=True`，存储 `mean` 和 `std`**。
- **否则，设置 `mean=0`，`std=1`（无标准化）。**

---

## **8. 处理数据子集**
```python
if cfg_data.size < len(dataset):
    dataset = Subset(dataset, torch.arange(0, cfg_data.size))
```
- **如果 `cfg_data.size` 小于数据集总大小，则** **裁剪数据集**（用于小规模实验）。

---

## **9. 返回数据集和 `collate_fn`**
```python
collate_fn = _torchvision_collate
return dataset, collate_fn
```
- **`collate_fn` 处理数据批次（默认 `torch.utils.data.default_collate`）。**
- **返回 `dataset` 和 `collate_fn`**。

---

## **📌 代码流程总结**
1. **初始化 `ToTensor()` 作为默认变换**。
2. **加载不同数据集**（`CIFAR10`、`ImageNet` 等）。
3. **计算数据均值和标准差（如果 `normalize=True`）**。
4. **解析数据增强**（如 `RandomCrop`）。
5. **应用变换 `dataset.transform`**。
6. **存储 `mean` 和 `std`**。
7. **如果 `cfg_data.size` 小于数据总量，则裁剪数据**。
8. **返回 `dataset` 和 `collate_fn`**。

---

## **📌 结论**
✅ **支持 `CIFAR10`、`ImageNet` 等多个数据集**。  
✅ **可自动计算 `mean/std`，并支持标准化**。  
✅ **支持数据增强、子集选择、下载等功能**。  

💡 **总结：该函数是计算机视觉数据加载的核心组件，为联邦学习和隐私攻击任务提供数据支持！🚀**



## **📌 `_torchvision_collate()` 代码解析**

### **🔹 作用**
- 该函数是一个 **自定义的数据批次合并函数（collate function）**，用于 `torch.utils.data.DataLoader`。
- **修改默认 `collate_fn` 的行为**，使其返回 **字典格式**：
  ```python
  return dict(inputs=..., labels=...)
  ```
- **确保张量在多线程数据加载时使用共享内存，避免额外的拷贝，提高数据加载效率**。

---

## **📌 代码结构**
```python
def _torchvision_collate(batch):
```
- **输入参数**
  - `batch`：一个 **批次样本列表**，格式如下：
    ```python
    batch = [(img1, label1), (img2, label2), (img3, label3), ...]
    ```
    其中：
    - `img_i` 是一个 `torch.Tensor`（图像数据）
    - `label_i` 是一个整数（类别标签）

- **返回值**
  - **`dict(inputs=张量, labels=张量)`**
  - **格式示例**：
    ```python
    {
        "inputs": torch.Size([batch_size, C, H, W]),  # 图像数据
        "labels": torch.Size([batch_size])  # 目标标签
    }
    ```

---

## **1. `transposed = list(zip(*batch))`：将 batch 转置**
```python
transposed = list(zip(*batch))
```
- **作用**：将 `batch` **拆分成两个列表**：
  - `transposed[0]`：图像 `img_i` 列表
  - `transposed[1]`：标签 `label_i` 列表

🔹 **示例**
```python
batch = [(img1, label1), (img2, label2), (img3, label3)]
transposed = list(zip(*batch))
```
**结果**
```python
transposed[0] = (img1, img2, img3)  # 图像数据
transposed[1] = (label1, label2, label3)  # 标签
```

---

## **2. `_stack_tensor(tensor_list)`：合并 `inputs` 张量**
```python
def _stack_tensor(tensor_list):
```
- **作用**：将 `tensor_list` **堆叠** 成一个 `torch.Tensor`，并 **在多线程模式下使用共享内存**。

### **📌 代码解析**
```python
elem = tensor_list[0]
elem_type = type(elem)
out = None
```
- 获取 `tensor_list` 第一个元素，检查其类型（应为 `torch.Tensor`）。

```python
if torch.utils.data.get_worker_info() is not None:
```
- **检查当前是否在 `DataLoader` 的子进程中**（即多线程数据加载）。
- **如果在子进程中，分配共享内存，提高效率**：
  ```python
  numel = sum(x.numel() for x in tensor_list)
  storage = elem.storage()._new_shared(numel)
  out = elem.new(storage)
  ```
  - **计算 `tensor_list` 中所有张量的元素数量 `numel`**。
  - **创建共享内存 `storage`**，避免不必要的数据拷贝，提高数据加载效率。

```python
return torch.stack(tensor_list, 0, out=out)
```
- **最终使用 `torch.stack()` 合并张量**，并返回共享内存中的 `out`。

---

## **3. `labels = torch.tensor(transposed[1])`**
```python
return dict(inputs=_stack_tensor(transposed[0]), labels=torch.tensor(transposed[1]))
```
- **将 `labels` 列表转换为 `torch.Tensor`**：
  ```python
  labels = torch.tensor(transposed[1])
  ```

- **最终返回**
  ```python
  return {
      "inputs": batch_size 张量,
      "labels": batch_size 张量
  }
  ```

---

## **📌 代码流程总结**
1. **将 `batch` 转置**：
   ```python
   transposed = list(zip(*batch))
   ```
   - `transposed[0]`：所有图像
   - `transposed[1]`：所有标签

2. **合并 `inputs`**
   ```python
   inputs = _stack_tensor(transposed[0])
   ```
   - **多线程模式下** 使用 **共享内存** 加速数据加载。

3. **合并 `labels`**
   ```python
   labels = torch.tensor(transposed[1])
   ```

4. **返回字典**
   ```python
   return dict(inputs=inputs, labels=labels)
   ```

---

## **📌 示例**
### **🎯 输入**
```python
batch = [
    (torch.randn(3, 32, 32), 0),  # 图像 & 标签
    (torch.randn(3, 32, 32), 1),
    (torch.randn(3, 32, 32), 2)
]
```

### **🎯 代码执行**
```python
result = _torchvision_collate(batch)
```

### **🎯 输出**
```python
{
    "inputs": torch.Size([3, 3, 32, 32]),  # 3 张图片，3 通道，32x32 大小
    "labels": torch.Size([3])  # 3 个标签
}
```

---

## **📌 结论**
✅ **该函数自定义 `collate_fn`，返回字典格式 `{inputs, labels}`**  
✅ **使用 `torch.stack()` 组合 `inputs`，提高 GPU 训练效率**  
✅ **支持多线程数据加载，在子进程中使用共享内存，避免拷贝**  

💡 **总结：`_torchvision_collate()` 是一个优化版的 `collate_fn`，特别适用于 `DataLoader` 进行高效数据加载！🚀**

---
---
---

## **📌 `_split_dataset_vision()` 代码解析**

### **🔹 作用**
- **对视觉数据集 `dataset` 进行用户级别划分（Partitioning）**，适用于 **联邦学习（Federated Learning, FL）** 场景。
- **不同的 `partition` 方案决定了数据如何分配给用户**。
- **支持不同的数据划分策略**：
  - `balanced`（均衡划分）
  - `unique-class`（唯一类别）
  - `mixup`（混合划分）
  - `feat_est`（特定类别划分）
  - `random-full`（随机但允许重复）
  - `random`（随机但不重复）
  - `none`（所有用户使用相同数据）

---

## **📌 代码结构**
```python
def _split_dataset_vision(dataset, cfg_data, user_idx=None, return_full_dataset=False):
```
- **输入参数**
  - `dataset`：完整数据集（`torchvision.datasets`）。
  - `cfg_data`：数据集的配置信息，包含划分方式 `partition`。
  - `user_idx`：当前用户索引（指定该用户应该使用的数据）。
  - `return_full_dataset`：是否返回完整数据集（如用于整体分析）。

- **返回值**
  - `dataset`：划分后的子数据集。

---

## **1. 处理 `return_full_dataset=True` 的情况**
```python
if not return_full_dataset:
```
- **如果 `return_full_dataset=True`，则直接返回完整数据集**，不进行划分。

---

## **2. 选择 `user_idx`**
```python
if user_idx is None:
    user_idx = torch.randint(0, cfg_data.default_clients, (1,))
else:
    if user_idx > cfg_data.default_clients:
        raise ValueError("This user index exceeds the maximal number of clients.")
```
- **如果 `user_idx` 为空**，则随机分配一个用户索引。
- **如果 `user_idx` 超出 `cfg_data.default_clients`（总用户数）**，抛出错误。

---

## **3. 数据划分策略**
根据 **`cfg_data.partition`** 选择不同的 **数据划分方式**。

---

### **📌 3.1 `balanced`（均衡划分）**
```python
if cfg_data.partition == "balanced":
    data_per_class_per_user = len(dataset) // len(dataset.classes) // cfg_data.default_clients
    if data_per_class_per_user < 1:
        raise ValueError("Too many clients for a balanced dataset.")
    data_ids = []
    for class_idx, _ in enumerate(dataset.classes):
        data_with_class = [idx for (idx, label) in dataset.lookup.items() if label == class_idx]
        data_ids += data_with_class[
            user_idx * data_per_class_per_user : data_per_class_per_user * (user_idx + 1)
        ]
    dataset = Subset(dataset, data_ids)
```
- **每个用户获得** **相同数量的每个类别的样本**（均衡划分）。
- 计算 **每个用户每个类别的样本数**：
  ```python
  data_per_class_per_user = len(dataset) // len(dataset.classes) // cfg_data.default_clients
  ```
- **数据选择方式**
  ```python
  for class_idx in dataset.classes:
      data_with_class = [idx for (idx, label) in dataset.lookup.items() if label == class_idx]
      data_ids += data_with_class[user_idx * data_per_class_per_user : (user_idx + 1) * data_per_class_per_user]
  ```
  - **遍历所有类别**，为 `user_idx` 选择属于该类别的 `data_per_class_per_user` 张图片。

---

### **📌 3.2 `unique-class`（唯一类别）**
```python
elif cfg_data.partition == "unique-class":
    data_ids = [idx for (idx, label) in dataset.lookup.items() if label == user_idx]
    dataset = Subset(dataset, data_ids)
```
- **每个用户只获得单个类别的数据**（类别 ID = 用户 ID）。
- **适用于“每个用户只学一个类别”的 FL 任务**。

---

### **📌 3.3 `mixup`（数据混合划分）**
```python
elif cfg_data.partition == "mixup":
    if "mixup_freq" in cfg_data:
        mixup_freq = cfg_data.mixup_freq
    else:
        mixup_freq = 2
    data_per_user = len(dataset) // cfg_data.default_clients
    last_id = len(dataset) - 1
    data_ids = []
    for i in range(data_per_user):
        data_ids.append(user_idx * data_per_user + i)
        data_ids.append(last_id - user_idx * data_per_user - i)
    dataset = Subset(dataset, data_ids)
```
- **数据被双向分配**：
  - 一部分从 `user_idx * data_per_user` 开始
  - 另一部分从 `last_id - user_idx * data_per_user` 倒序取
- **确保每个用户的数据同时来自头部和尾部，增加数据多样性**。

---

### **📌 3.4 `feat_est`（特定类别划分）**
```python
elif cfg_data.partition == "feat_est":
    num_data_points = cfg_data.num_data_points if "num_data_points" in cfg_data else 1
    target_label = cfg_data.target_label if "target_label" in cfg_data else 0
    data_ids = [idx for (idx, label) in dataset.lookup.items() if label == target_label]
    data_ids = data_ids[user_idx * num_data_points : (user_idx + 1) * num_data_points]
    dataset = Subset(dataset, data_ids)
```
- **仅为 `target_label` 指定的类别分配数据**。
- **适用于特定类别攻击（Feature Estimation）**。

---

### **📌 3.5 `random-full`（随机但允许重复）**
```python
elif cfg_data.partition == "random-full":
    data_per_user = len(dataset) // cfg_data.default_clients
    data_ids = torch.randperm(len(dataset))[:data_per_user]
    dataset = Subset(dataset, data_ids)
```
- **随机选择数据，允许不同用户共享数据**。

---

### **📌 3.6 `random`（随机但不重复）**
```python
elif cfg_data.partition == "random":
    data_per_user = len(dataset) // cfg_data.default_clients
    generator = torch.Generator()
    generator.manual_seed(233)
    data_ids = torch.randperm(len(dataset))[user_idx * data_per_user : data_per_user * (user_idx + 1)]
    dataset = Subset(dataset, data_ids)
```
- **随机划分数据，每个用户数据不重复**（**保证不同用户数据唯一性**）。
- **使用固定随机种子 `233` 以保证可复现**。

---

### **📌 3.7 `none`（所有用户共享完整数据集）**
```python
elif cfg_data.partition == "none":
    pass
```
- **所有用户共享完整数据集**（即无划分）。
- **用于 sanity check（完整性测试）**。

---

### **📌 3.8 抛出未实现的划分策略**
```python
else:
    raise ValueError(f"Partition scheme {cfg_data.partition} not implemented.")
```
- **如果 `partition` 未在上述策略中定义，则抛出错误**。

---

## **📌 代码流程总结**
1. **检查是否返回完整数据集**
   - `return_full_dataset=True` → 直接返回完整数据集。

2. **确定用户索引**
   - 若 `user_idx=None`，则随机选择一个。

3. **选择数据划分策略**
   - `balanced`：所有用户获得相同比例的类别数据
   - `unique-class`：每个用户只获得单个类别
   - `mixup`：数据混合，提升多样性
   - `feat_est`：针对特定类别进行特征估计
   - `random-full`：随机数据，可能重复
   - `random`：随机数据，不重复
   - `none`：所有用户共享完整数据集

---

## **📌 结论**
✅ **支持多种数据划分策略，适用于联邦学习**  
✅ **保证用户数据独立性，支持均衡、随机、唯一类别等划分方式**  
✅ **可复现（固定随机种子），支持 FL 训练**  

💡 **总结：该函数是联邦学习中“数据划分”的核心，确保每个用户获取正确的数据集！🚀**