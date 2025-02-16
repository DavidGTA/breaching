# Analysis

A range of metrics are implemented here. The main entry point should be `analysis.report`, which automatically discovers
the kind of data that is present and evaluates metrics accordingly.

Several metrics require additional packages:
* R-PSNR : `kornia`
* CW-SSIM: `git+https://github.com/fbcotter/pytorch_wavelets`
* LPIPS: `lpips`
* IIP (Image Identifiability Precision scores): `lpips`
* BLEU: `datasets` (from huggingface)
* Rouge: `datasets` and `rouge-score`
* sacrebleu: `datasets` and `sacrebleu`

# 分析

这里实现了一系列的指标。主要的入口点应为 `analysis.report`，该函数会自动识别数据的类型并相应地评估指标。

一些指标需要额外的包：
* R-PSNR: `kornia`
* CW-SSIM: `git+https://github.com/fbcotter/pytorch_wavelets`
* LPIPS: `lpips`
* IIP（图像可识别精度分数）：`lpips`
* BLEU: `datasets`（来自 huggingface）
* Rouge: `datasets` 和 `rouge-score`
* sacrebleu: `datasets` 和 `sacrebleu`