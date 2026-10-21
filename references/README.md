# 第一题（3）的参考论文

下载日期：2026-10-09。正文只引用与加法、乘法误差界直接相关的两篇论文。

1. `Valenzano_2014_non_reopening.pdf`
   - Richard Valenzano, Nathan R. Sturtevant, Jonathan Schaeffer. *Worst-Case Solution Quality Analysis When Not Re-Expanding Nodes in Best-First Search*. AAAI 2014.
   - 来源：https://www.cs.du.edu/~sturtevant/papers/non_reopening.pdf
   - 阅读位置：PDF 第 2 页的假设和不一致量定义、第 3 页的算法约定、第 4–5 页的定理 4.4–4.6。
   - 本文引用定理 4.6 及前缀误差引理，并据此推导一致性、每边加法放宽、每边乘法放宽三个推论。
   - 定理假设非负边代价、非负启发函数，目标处为 0，可达目标的节点启发值有限；不要求可采纳。求和从最优路径第二条边开始。

2. `Likhachev_2003_ARAstar.pdf`
   - Maxim Likhachev, Geoff Gordon, Sebastian Thrun. *ARA*: Anytime A* with Provable Bounds on Sub-Optimality*. NIPS 16, 2003.
   - 来源：https://papers.neurips.cc/paper_files/paper/2003/file/ee8fe9093fbbb687bef15a38facc44d2-Paper.pdf
   - 阅读位置：PDF 第 2 页第 2.1 节、第 3 页第 2.2 节末段。
   - 原文使用一致的基础启发函数和正边代价，明确指出加权后限制每个状态展开至多一次，仍保留权重倍数的解质量保证。

同名 `.txt` 是为检索生成的文字提取版本；公式和条件已结合 PDF 原页核对。
