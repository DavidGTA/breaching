# Examples

This folder contains jupyter notebooks that show how to use this framework, and visualize what (some) current attacks can do.

Several of them visualize and replicate our recent work:
Attack             | Type | Dataset | Update | Threat Model  | Publication
:-----------------:|:----:|-------:|:------:|:-------:|:------------:
[Robbing The Fed (Default)](Robbing%20The%20Fed%20-%20%20Analytic%20Attack%20-%20Malicious%20Model%20on%20ImageNet.ipynb)  | Analytic | ImageNet | fedSGD | Malicious (Model) | [LINK](https://openreview.net/forum?id=fwzUgo0FM9v)
[Robbing The Fed (Text)](Robbing%20The%20Fed%20-%20%20Analytic%20Attack%20-%20Malicious%20Model%20on%20Wikitext.ipynb)  | Analytic | wikitext | fedSGD | Malicious (Model) | [LINK](https://openreview.net/forum?id=fwzUgo0FM9v)
[Robbing The Fed (Local Updates)](Robbing%20The%20Fed%20-%20%20Analytic%20Attack%20-%20Malicious%20Model%20on%20ImageNet%20%20-%20Federated%20Averaging.ipynb)  | Analytic | ImageNet | fedAVG | Malicious (Model) | [LINK](https://openreview.net/forum?id=fwzUgo0FM9v)
[Robbing The Fed (One Shot against arbitrary aggregation)](Robbing%20The%20Fed%20-%20%20Analytic%20Attack%20-%20Malicious%20Model%20on%20ImageNet%20-%20One%20Shot%20Attack%20against%20arbitrary%20aggregation.ipynb)  | Analytic | ImageNet | fedSGD | Malicious (Model) | [LINK](https://openreview.net/forum?id=fwzUgo0FM9v)
[Decepticons (FL transformer)](Decepticons%20-%20%20Analytic%20Attack%20-%20Transformer%20Model%20on%20Wikitext.ipynb)  | Analytic | wikitext | fedSGD | Malicious (Parameters) | [LINK](https://arxiv.org/abs/2201.12675)
[Decepticons (BERT)](Decepticons%20-%20%20Analytic%20Attack%20-%20BERT%20on%20Wikitext.ipynb)  | Analytic | wikitext | fedSGD | Malicious (Parameters) | [LINK](https://arxiv.org/abs/2201.12675)
[Decepticons (small GPT2)](Decepticons%20-%20%20Analytic%20Attack%20-%20small%20GPT2%20on%20Wikitext.ipynb)  | Analytic | wikitext | fedSGD | Malicious (Parameters) | [LINK](https://arxiv.org/abs/2201.12675)
[Decepticons (small GPT2)](Decepticons%20-%20%20Analytic%20Attack%20-%20small%20GPT2%20on%20custom%20text.ipynb)  | Analytic | custom | fedSGD | Malicious (Parameters) | [LINK](https://arxiv.org/abs/2201.12675)
[Fishing for User Data (Binary Attack)](Fishing%20for%20User%20Data%20-%20Meta%20Optimization-based%20Attack%20-%20Feature%20Fishing%20Cross-Silo.ipynb)  | (Meta) Optimization | ImageNet | fedSGD (cross-silo) | Malicious (Parameters) | [LINK](https://arxiv.org/abs/2202.00580)
[Fishing for User Data (Binary Attack)](Fishing%20for%20User%20Data%20-%20Meta%20Analytic%20Attack%20-%20Feature%20Fishing%20Cross-Silo.ipynb)  | (Meta) Analytic | ImageNet | fedSGD (cross-silo) | Malicious (Parameters) | [LINK](https://arxiv.org/abs/2202.00580)
[Fishing for User Data (Feature Attack)](Fishing%20for%20User%20Data%20-%20Meta%20Optimization-based%20Attack%20-%20Feature%20Fishing%20Cross%20Device.ipynb)  | (Meta) Optimization | ImageNet | fedSGD (cross-device) | Malicious (Parameters) | [LINK](https://arxiv.org/abs/2202.00580)


But we also replicate a number of other work in the literature.
These other examples are (in chronological order):

Attack             | Type | Dataset | Update | Threat Model  | Publication
:-----------------:|:----:|-------:|:------:|:-------:|:------------:
[Beyond Inferring Class Representatives](Beyond%20Inferring%20Class%20Representatives%20-%20Optimization-based%20Attack%20-%20ConvNet%20CIFAR-10.ipynb)  | Optimization | CIFAR10 | fedSGD | Honest | [LINK](https://ieeexplore.ieee.org/abstract/document/8737416)
[Deep Leakage from Gradients](Deep%20Leakage%20from%20Gradients%20-%20Optimization-based%20Attack%20-%20ConvNet%20CIFAR-10.ipynb)  | Optimization | CIFAR10 | fedSGD | Honest (Also optimizes labels)| [LINK](https://papers.nips.cc/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html)
[Inverting Gradients (Basic)](Inverting%20Gradients%20-%20Optimization-based%20Attack%20-%20ResNet18%20on%20ImageNet.ipynb)  | Optimization | ImageNet | fedSGD | Honest| [LINK](https://proceedings.neurips.cc/paper/2020/hash/c4ede56bbd98819ae6112b20ac6bf145-Abstract.html)
[Inverting Gradients (Large Batch)](Inverting%20Gradients%20-%20Optimization-based%20Attack%20-%20Large%20Batch%20CIFAR-100.ipynb)  | Optimization | CIFAR100 | fedSGD | Honest| [LINK](https://proceedings.neurips.cc/paper/2020/hash/c4ede56bbd98819ae6112b20ac6bf145-Abstract.html)
[Inverting Gradients (Local Updates)](Inverting%20Gradients%20-%20Optimization-based%20Attack%20-%20ResNet18%20on%20ImageNet%20-%20Federated%20Averaging.ipynb)  | Optimization | CIFAR10 | fedAVG | Honest| [LINK](https://proceedings.neurips.cc/paper/2020/hash/c4ede56bbd98819ae6112b20ac6bf145-Abstract.html)
[Analytic Attack (Linear Model)](Analytic%20Attack%20-%20Linear%20Model%20on%20ImageNet.ipynb)  | Analytic | ImageNet | fedSGD | Honest| Several
[R-GAP](R-GAP%20%20-%20Recursive%20Attack%20-%20Small%20ConvNet%20on%20CIFAR-10.ipynb)  | Recursive | CIFAR10 | fedSGD | Honest| [LINK](https://openreview.net/forum?id=RSU17UoKfJF)
[See Through Gradients](See%20through%20Gradients%20-%20Optimization-based%20Attack%20-%20ResNet50%20on%20ImageNet.ipynb)  | Optimization | ImageNet | fedSGD | Honest (but a MOCO parameter vector)| [LINK](https://openaccess.thecvf.com/content/CVPR2021/html/Yin_See_Through_Gradients_Image_Batch_Recovery_via_GradInversion_CVPR_2021_paper.html)
[TAG (Classification)](TAG%20-%20Optimization-based%20Attack%20-%20BERT%20for%20COLA%20Sentence%20Classification.ipynb)  | Optimization | COLA | fedSGD | Honest| [LINK](https://aclanthology.org/2021.findings-emnlp.305/)
[TAG (Language Modeling)](TAG%20-%20Optimization-based%20Attack%20-%20FL-Transformer%20for%20Causal%20LM.ipynb)  | Optimization | wikitext | fedSGD | Honest| [LINK](https://aclanthology.org/2021.findings-emnlp.305/)
[Curious Abandon Honesty (Images)](Curious%20Abandon%20Honesty%20-%20%20Analytic%20Attack%20-%20Malicious%20Model%20on%20ImageNet.ipynb)  | Analytic | ImageNet | fedSGD | Malicious | [LINK](https://arxiv.org/abs/2112.02918)
[Curious Abandon Honesty (Text)](Curious%20Abandon%20Honesty%20-%20%20Analytic%20Attack%20-%20Malicious%20Model%20on%20Wikitext.ipynb)  | Analytic | wikitext | fedSGD | Malicious | [LINK](https://arxiv.org/abs/2112.02918)
[APRIL (Analytic Attack against ViT)](APRIL%20%20-%20Analytic%20Attack%20-%20Vision%20Transformer%20on%20ImageNet.ipynb)  | Analytic | ImageNet | fedSGD | Honest | [LINK](https://arxiv.org/abs/2112.14087)
[Modernized Inverting Gradients](Modern%20Hyperparameters%20-%20Optimization-based%20Attack%20-%20ResNet18%20on%20ImageNet.ipynb)  | Optimization | ImageNet | fedSGD | Honest| -


What is not quite visibile in this summary is that these attacks are improved by works that study
* Optimal initializations for these attacks ([A Framework for Evaluating Gradient Leakage Attacks in Federated Learning
](https://arxiv.org/abs/2004.10397))
* Label recovery algorithms ([User-Level Label Leakage from Gradients in Federated Learning
](https://arxiv.org/abs/2105.09369))
* Commentary, metrics and additional empirical evaluations of these attacks.

There is also an interesting direction of work using generative models to aid reconstruction (for example [Gradient Inversion with Generative Image Prior](https://fl-icml.github.io/2021/papers/FL-ICML21_paper_75.pdf), but also going back all the way to [Deep Models Under the GAN: Information Leakage from Collaborative Deep Learning
](https://arxiv.org/abs/1702.07464)) which is not included in this framework, as well as other FL scenarios (such as vertical FL, e.g. in [CAFE: Catastrophic Data Leakage in Vertical Federated Learning
](https://arxiv.org/abs/2110.15122)) that are not covered here.


# 示例

此文件夹包含Jupyter笔记本，展示了如何使用该框架，并可视化当前（部分）攻击的效果。

其中一些示例可视化并复制了我们最近的工作：

| 攻击 | 类型 | 数据集 | 更新方式 | 威胁模型 | 发表文章 |
|:-----------------:|:----:|-------:|:------:|:-------:|:------------:|
| [Robbing The Fed (默认)](Robbing%20The%20Fed%20-%20%20Analytic%20Attack%20-%20Malicious%20Model%20on%20ImageNet.ipynb)  | 分析型 | ImageNet | fedSGD | 恶意（模型） | [链接](https://openreview.net/forum?id=fwzUgo0FM9v) |
| [Robbing The Fed (文本)](Robbing%20The%20Fed%20-%20%20Analytic%20Attack%20-%20Malicious%20Model%20on%20Wikitext.ipynb)  | 分析型 | wikitext | fedSGD | 恶意（模型） | [链接](https://openreview.net/forum?id=fwzUgo0FM9v) |
| [Robbing The Fed (本地更新)](Robbing%20The%20Fed%20-%20%20Analytic%20Attack%20-%20Malicious%20Model%20on%20ImageNet%20%20-%20Federated%20Averaging.ipynb)  | 分析型 | ImageNet | fedAVG | 恶意（模型） | [链接](https://openreview.net/forum?id=fwzUgo0FM9v) |
| [Robbing The Fed (对抗任意聚合的一次性攻击)](Robbing%20The%20Fed%20-%20%20Analytic%20Attack%20-%20Malicious%20Model%20on%20ImageNet%20-%20One%20Shot%20Attack%20against%20arbitrary%20aggregation.ipynb)  | 分析型 | ImageNet | fedSGD | 恶意（模型） | [链接](https://openreview.net/forum?id=fwzUgo0FM9v) |
| [Decepticons (FL转换器)](Decepticons%20-%20%20Analytic%20Attack%20-%20Transformer%20Model%20on%20Wikitext.ipynb)  | 分析型 | wikitext | fedSGD | 恶意（参数） | [链接](https://arxiv.org/abs/2201.12675) |
| [Decepticons (BERT)](Decepticons%20-%20%20Analytic%20Attack%20-%20BERT%20on%20Wikitext.ipynb)  | 分析型 | wikitext | fedSGD | 恶意（参数） | [链接](https://arxiv.org/abs/2201.12675) |
| [Decepticons (小型GPT2)](Decepticons%20-%20%20Analytic%20Attack%20-%20small%20GPT2%20on%20Wikitext.ipynb)  | 分析型 | wikitext | fedSGD | 恶意（参数） | [链接](https://arxiv.org/abs/2201.12675) |
| [Decepticons (小型GPT2)](Decepticons%20-%20%20Analytic%20Attack%20-%20small%20GPT2%20on%20custom%20text.ipynb)  | 分析型 | custom | fedSGD | 恶意（参数） | [链接](https://arxiv.org/abs/2201.12675) |
| [Fishing for User Data (二进制攻击)](Fishing%20for%20User%20Data%20-%20Meta%20Optimization-based%20Attack%20-%20Feature%20Fishing%20Cross-Silo.ipynb)  | (元) 优化型 | ImageNet | fedSGD (跨孤岛) | 恶意（参数） | [链接](https://arxiv.org/abs/2202.00580) |
| [Fishing for User Data (二进制攻击)](Fishing%20for%20User%20Data%20-%20Meta%20Analytic%20Attack%20-%20Feature%20Fishing%20Cross-Silo.ipynb)  | (元) 分析型 | ImageNet | fedSGD (跨孤岛) | 恶意（参数） | [链接](https://arxiv.org/abs/2202.00580) |
| [Fishing for User Data (特征攻击)](Fishing%20for%20User%20Data%20-%20Meta%20Optimization-based%20Attack%20-%20Feature%20Fishing%20Cross%20Device.ipynb)  | (元) 优化型 | ImageNet | fedSGD (跨设备) | 恶意（参数） | [链接](https://arxiv.org/abs/2202.00580) |

但我们也复制了文献中一些其他的工作。
这些其他的示例如下（按时间顺序排列）：

|                                                                                     攻击                                                                                      | 类型 | 数据集 | 更新方式 | 威胁模型 | 发表文章 |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----:|-------:|:------:|:-------:|:------------:|
|      [超越推断类代表(Beyond Inferring Class Representatives)](Beyond%20Inferring%20Class%20Representatives%20-%20Optimization-based%20Attack%20-%20ConvNet%20CIFAR-10.ipynb)       | 优化型 | CIFAR10 | fedSGD | 诚实 | [链接](https://ieeexplore.ieee.org/abstract/document/8737416) |
|                  [梯度深度泄漏(Deep Leakage from Gradients)](Deep%20Leakage%20from%20Gradients%20-%20Optimization-based%20Attack%20-%20ConvNet%20CIFAR-10.ipynb)                  | 优化型 | CIFAR10 | fedSGD | 诚实（也优化标签） | [链接](https://papers.nips.cc/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html) |
|                   [反转梯度（基础版）(Inverting Gradients (Basic))](Inverting%20Gradients%20-%20Optimization-based%20Attack%20-%20ResNet18%20on%20ImageNet.ipynb)                    | 优化型 | ImageNet | fedSGD | 诚实 | [链接](https://proceedings.neurips.cc/paper/2020/hash/c4ede56bbd98819ae6112b20ac6bf145-Abstract.html) |
|                [反转梯度（大批量）(Inverting Gradients (Large Batch))](Inverting%20Gradients%20-%20Optimization-based%20Attack%20-%20Large%20Batch%20CIFAR-100.ipynb)                | 优化型 | CIFAR100 | fedSGD | 诚实 | [链接](https://proceedings.neurips.cc/paper/2020/hash/c4ede56bbd98819ae6112b20ac6bf145-Abstract.html) |
| [反转梯度（本地更新）(Inverting Gradients (Local Updates))](Inverting%20Gradients%20-%20Optimization-based%20Attack%20-%20ResNet18%20on%20ImageNet%20-%20Federated%20Averaging.ipynb) | 优化型 | CIFAR10 | fedAVG | 诚实 | [链接](https://proceedings.neurips.cc/paper/2020/hash/c4ede56bbd98819ae6112b20ac6bf145-Abstract.html) |
|                                 [分析攻击（线性模型）(Analytic Attack (Linear Model))](Analytic%20Attack%20-%20Linear%20Model%20on%20ImageNet.ipynb)                                  | 分析型 | ImageNet | fedSGD | 诚实 | 若干 |
|                                        [R-GAP(R-GAP)](R-GAP%20%20-%20Recursive%20Attack%20-%20Small%20ConvNet%20on%20CIFAR-10.ipynb)                                        | 递归型 | CIFAR10 | fedSGD | 诚实 | [链接](https://openreview.net/forum?id=RSU17UoKfJF) |
|                       [穿透梯度(See Through Gradients)](See%20through%20Gradients%20-%20Optimization-based%20Attack%20-%20ResNet50%20on%20ImageNet.ipynb)                       | 优化型 | ImageNet | fedSGD | 诚实（但使用MOCO参数向量） | [链接](https://openaccess.thecvf.com/content/CVPR2021/html/Yin_See_Through_Gradients_Image_Batch_Recovery_via_GradInversion_CVPR_2021_paper.html) |
|                      [TAG（分类）(TAG (Classification))](TAG%20-%20Optimization-based%20Attack%20-%20BERT%20for%20COLA%20Sentence%20Classification.ipynb)                       | 优化型 | COLA | fedSGD | 诚实 | [链接](https://aclanthology.org/2021.findings-emnlp.305/) |
|                         [TAG（语言建模）(TAG (Language Modeling))](TAG%20-%20Optimization-based%20Attack%20-%20FL-Transformer%20for%20Causal%20LM.ipynb)                          | 优化型 | wikitext | fedSGD | 诚实 | [链接](https://aclanthology.org/2021.findings-emnlp.305/) |
|            [好奇放弃诚实（图像）(Curious Abandon Honesty (Images))](Curious%20Abandon%20Honesty%20-%20%20Analytic%20Attack%20-%20Malicious%20Model%20on%20ImageNet.ipynb)             | 分析型 | ImageNet | fedSGD | 恶意 | [链接](https://arxiv.org/abs/2112.02918) |
|             [好奇放弃诚实（文本）(Curious Abandon Honesty (Text))](Curious%20Abandon%20Honesty%20-%20%20Analytic%20Attack%20-%20Malicious%20Model%20on%20Wikitext.ipynb)              | 分析型 | wikitext | fedSGD | 恶意 | [链接](https://arxiv.org/abs/2112.02918) |
|                 [APRIL（针对ViT的分析攻击）(APRIL (Analytic Attack against ViT))](APRIL%20%20-%20Analytic%20Attack%20-%20Vision%20Transformer%20on%20ImageNet.ipynb)                 | 分析型 | ImageNet | fedSGD | 诚实 | [链接](https://arxiv.org/abs/2112.14087) |
|                                [现代化反转梯度(Modernized Inverting Gradients)](Modern%20Hyperparameters%20-%20Optimization-based%20Attack%20-%20ResNet18%20on%20ImageNet.ipynb)                                 | 优化型 | ImageNet | fedSGD | 诚实 | - |

在这个总结中不太明显的是，这些攻击得到了以下研究工作的改进：

* 这些攻击的**最优初始化**研究（[《评估联邦学习中梯度泄漏攻击的框架》](https://arxiv.org/abs/2004.10397)）
* **标签恢复算法**（[《联邦学习中用户级标签泄漏》](https://arxiv.org/abs/2105.09369)）
* 对这些攻击的评论、指标和额外的实证评估。

还有一个有趣的研究方向是利用生成模型来辅助重建（例如，[《使用生成图像先验进行梯度反转》](https://fl-icml.github.io/2021/papers/FL-ICML21_paper_75.pdf)，以及追溯到[《GAN下的深度模型：来自协作深度学习的信息泄漏》](https://arxiv.org/abs/1702.07464)），但是这些内容没有包含在这个框架中。还有其他的联邦学习场景（例如，纵向联邦学习，如[《CAFE：纵向联邦学习中的灾难性数据泄漏》](https://arxiv.org/abs/2110.15122)），这些在这里没有涉及。