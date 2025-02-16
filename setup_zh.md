# 如何上传到 PyPI（适用于小白，也就是我）

### 步骤：

1. 检查并更新清单文件：
   ```bash
   check-manifest -u -v
   ```

2. 构建 Python 包：
   ```bash
   python -m build
   ```

3. 上传到 TestPyPI：
   ```bash
   twine upload --repository testpypi dist/*
   ```

**每次出错后，记得增加计数器！😊**

---

### 测试安装：

1. 从 TestPyPI 安装：
   ```bash
   pip install -i https://test.pypi.org/simple/ breaching==0.1.0  # 可能不会安装依赖？
   ```

2. 直接安装本地构建的分发包：
   ```bash
   pip install dist/breaching-0.1.1.tar.gz
   ```