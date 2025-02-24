## breaching/breaching/cases/models/model_preparation.py 中 _construct_vision_model 函数中 resnet部分 详细解释

### **📌 代码解析**
该代码用于**解析 `cfg_model` 变量，并根据其值创建一个 ResNet 模型**。  
`cfg_model` 是一个字符串，如 `"resnet18"`，代码的主要作用是：
1. **解析 `cfg_model` 以确定 ResNet 版本（深度 `depth` 和宽度 `width`）**。
2. **调用 `resnet_depths_to_config(depth)` 获取对应的 ResNet 结构**（残差块类型 `block` 和层数 `layers`）。
3. **实例化 ResNet 模型**，并设置：
   - 归一化方式
   - 非线性激活
   - 是否使用 BatchNorm
   - 是否使用 `zero_init_residual`

---

## **1. 判断 `cfg_model` 是否为 ResNet**
```python
elif "resnet" in cfg_model.lower():
```
- **`cfg_model.lower()`**：将 `cfg_model` 转换为小写，确保大小写无关（例如 `"ResNet18"` 和 `"resnet18"` 都匹配）。
- **作用**：如果 `cfg_model` **包含 `"resnet"`**，则进入此逻辑。

---

## **2. 解析 ResNet 深度（`depth`）和宽度（`width`）**
```python
if "-" in cfg_model.lower():  # Hacky way to separate ResNets from wide ResNets which are e.g. 28-10
    depth = int("".join(filter(str.isdigit, cfg_model.split("-")[0])))
    width = int("".join(filter(str.isdigit, cfg_model.split("-")[1])))
else:
    depth = int("".join(filter(str.isdigit, cfg_model)))
    width = 1
```

### **🔹 `if "-" in cfg_model.lower()`**
- 这里的 `"-"` 是用于区分 **标准 ResNet 和 Wide ResNet（WRN）** 的方式：
  - **标准 ResNet** 的命名方式如 `"resnet18"`、`"resnet50"`
  - **Wide ResNet（WRN）** 的命名方式如 `"resnet28-10"`，其中 `28` 是网络深度，`10` 是扩展系数（宽度倍数）

### **🔹 代码解析**
1. **对于 `"resnet28-10"` 这样的 Wide ResNet**：
   ```python
   depth = int("".join(filter(str.isdigit, cfg_model.split("-")[0])))  # depth = 28
   width = int("".join(filter(str.isdigit, cfg_model.split("-")[1])))  # width = 10
   ```
   - `split("-")[0]` 获取 `"resnet28-10"` 中的 `"resnet28"`，提取其中的数字 `28` 作为深度。
   - `split("-")[1]` 获取 `"10"`，提取数字 `10` 作为宽度扩展因子。

2. **对于 `"resnet18"` 这样的标准 ResNet**：
   ```python
   depth = int("".join(filter(str.isdigit, cfg_model)))  # depth = 18
   width = 1  # 标准 ResNet 宽度因子为 1
   ```
   - 直接提取 `18` 作为深度，默认宽度 `width=1`。

### **示例输入**
| `cfg_model` | 解析后的 `depth` | 解析后的 `width` |
|-------------|--------------|--------------|
| `"resnet18"` | `18` | `1` |
| `"resnet50"` | `50` | `1` |
| `"resnet28-10"` | `28` | `10` |

---

## **3. 获取 ResNet 结构**
```python
block, layers = resnet_depths_to_config(depth)
```
- **作用**：将解析出的 `depth` 传入 `resnet_depths_to_config(depth)`，获取对应的：
  - `block`：使用的残差块类型（如 `BasicBlock` 或 `Bottleneck`）
  - `layers`：每个阶段的残差块数量（如 `[2, 2, 2, 2]` 对应 ResNet-18）

resnet_depths_to_config(depth)函数对应表格：

| **`depth` (网络深度)** | **`block` (残差块类型)** | **`layers` (每个阶段的层数)** |
|-----------------|-----------------|----------------|
| `20`  | `BasicBlock`  | `[3, 3, 3]`  |
| `32`  | `BasicBlock`  | `[5, 5, 5]`  |
| `56`  | `BasicBlock`  | `[9, 9, 9]`  |
| `110` | `BasicBlock`  | `[18, 18, 18]` |
| `18`  | `BasicBlock`  | `[2, 2, 2, 2]`  |
| `34`  | `BasicBlock`  | `[3, 4, 6, 3]`  |
| `50`  | `Bottleneck`  | `[3, 4, 6, 3]`  |
| `101` | `Bottleneck`  | `[3, 4, 23, 3]` |
| `152` | `Bottleneck`  | `[3, 8, 36, 3]` |

---

### **📌 解析**
- **使用 `BasicBlock` 的 ResNet**
  - `20、32、56、110`：这些是 **小型 ResNet**（通常用于 CIFAR-10/100 数据集）。
  - `18、34`：这是 **经典 ResNet**（通常用于 ImageNet 任务）。
  
- **使用 `Bottleneck` 的 ResNet**
  - `50、101、152`：这些是 **更深的 ResNet**，用于更大规模的数据集，如 ImageNet。

💡 **`depth` 决定了网络的层数，而 `block` 决定了使用 `BasicBlock`（适用于较浅网络）还是 `Bottleneck`（适用于深层网络）！** 🚀
---

## **4. 构建 ResNet 模型**
```python
model = ResNet(
    block,
    layers,
    channels,
    classes,
    stem="CIFAR",
    convolution_type="Standard",
    nonlin="ReLU",
    norm="BatchNorm2d",
    downsample="B",
    width_per_group=(16 if len(layers) < 4 else 64) * width,
    zero_init_residual=False,
)
```

### **📌 关键参数**
| 参数 | 作用 |
|------|------|
| `block` | ResNet 的基本单元（`BasicBlock` 或 `Bottleneck`） |
| `layers` | 每个阶段的层数，如 `[2, 2, 2, 2]` |
| `channels` | 输入通道数 |
| `classes` | 任务的类别数 |
| `stem="CIFAR"` | 是否使用 CIFAR-10 的特殊卷积层 |
| `convolution_type="Standard"` | 使用标准卷积 |
| `nonlin="ReLU"` | 使用 `ReLU` 作为激活函数 |
| `norm="BatchNorm2d"` | 归一化方式，使用 `BatchNorm2d` |
| `downsample="B"` | 残差连接中的降采样方法 |
| `width_per_group=(16 if len(layers) < 4 else 64) * width` | **计算通道扩展系数** |
| `zero_init_residual=False` | 是否初始化残差块的 `gamma` 为 0 |

---

## **5. `width_per_group` 计算方式**
```python
width_per_group=(16 if len(layers) < 4 else 64) * width
```
- **如果 `layers` 长度小于 4**（ResNet-18/34），使用 `16` 作为基准通道数。
- **如果 `layers` 长度大于等于 4**（ResNet-50+），使用 `64` 作为基准通道数。
- 结果再 **乘以 `width`，即 `width` 作为扩展系数**（适用于 Wide ResNet）。

### **示例**
| `depth` | `layers` | `width_per_group`（默认 `width=1`） |
|---------|---------|--------------------------------|
| `18` | `[2, 2, 2, 2]` | `16` |
| `34` | `[3, 4, 6, 3]` | `16` |
| `50` | `[3, 4, 6, 3]` | `64` |
| `28-10` | `[3, 3, 3]`（WideResNet） | `16 × 10 = 160` |

---

## **🎯 总结**
✅ **解析 `cfg_model`，区分标准 ResNet 和 Wide ResNet**：
- `cfg_model="resnet18"` → `depth=18`，`width=1`
- `cfg_model="resnet28-10"` → `depth=28`，`width=10`

✅ **根据 `depth` 确定 `block` 和 `layers` 结构**：
- `resnet_depths_to_config(18)` → `BasicBlock` + `[2, 2, 2, 2]`
- `resnet_depths_to_config(50)` → `Bottleneck` + `[3, 4, 6, 3]`

✅ **创建 `ResNet` 模型**
- **标准 ResNet：** `width_per_group=16`（ResNet-18/34） 或 `64`（ResNet-50+）
- **Wide ResNet：** `width_per_group = 基准通道数 × width`

💡 **最终，代码动态解析 `cfg_model` 并创建对应的 ResNet 网络！** 🚀