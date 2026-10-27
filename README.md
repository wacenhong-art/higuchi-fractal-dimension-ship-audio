# CI-Higuchi 船舶音频分类 / CI-Higuchi Ship-Audio Classification

## 项目简介 / Project Overview

本项目归档了基于船舶音频信号的 MATLAB 实现，用于研究随机共振预处理、分形维数特征提取和 KNN 分类的组合流程。已实现的特征算法包括置信区间改进 Higuchi（CI-Higuchi）、传统 Higuchi（作为 `ci_hfd.m` 内部对比函数）、Katz 分形维数和计盒维数（FX）。随机共振版本通过四阶 Runge--Kutta（RK4）求解器处理每个音频片段，然后提取一个标量分形维数特征。

本仓库不包含原始音频、提取后的特征、运行日志、指标或性能结论。代码的存在仅表示实现已归档；除非本 README 的状态表明确给出可核验依据，否则不表示实验已执行或验证。

## 项目目录 / Project Structure

```text
.
├── ci_hfd.m                    # 正式主程序：四类、随机共振 + CI-Higuchi + 1-NN
├── ci_hfd_no_sr.m              # 四类对比：不使用随机共振的 CI-Higuchi
├── ci_fd_katz.m                # 四类对比：不使用随机共振的 Katz
├── ci_fd_fx.m                  # 四类对比：不使用随机共振的计盒维数
├── Rung_kutta_JDSR_k.m         # 随机共振版本调用的 RK4 求解器
├── v2/
│   ├── ci_hfd.m                # 五类、随机共振 + CI-Higuchi
│   ├── ci_hfd_no_sr.m          # 五类、无随机共振 CI-Higuchi
│   ├── ci_fd_katz.m            # 五类、无随机共振 Katz
│   ├── ci_fd_katz_sr.m         # 五类、随机共振 Katz
│   ├── ci_fd_fx.m              # 五类、无随机共振计盒维数
│   └── Rung_kutta_JDSR_k.m     # RK4 副本
├── 对比算法/
│   ├── Katz_FD.m               # Katz 实现（保留文件内原作者署名）
│   └── FX.m                    # 计盒维数实现
├── PARAMETERS.md               # 从源代码提取的参数表
├── EXPERIMENTS.md              # 实验/归档状态说明
└── AGENTS.md                   # 本项目 Git 与 README 长期维护规则
```

## 2026-10-27：v3_ds 十尺度 Petrosian 分类 / 10-Scale Petrosian Classification

新增 `v3_ds/ds_petrosian.m`，使用同目录原有的 `Petrosian_FD.m`。该函数保持原始 Petrosian 定义不变：以一阶差分符号变化数 `N_delta` 计算 `D = log10(N) / (log10(N) + log10(N / (N + 0.4*N_delta)))`；主程序只按其既有单输入、单标量输出接口调用，未修改其公式。

程序优先使用 `v3_ds/DeepShip-main/` 中的四个指定输入：Cargo `27.wav`、Passengership `1.wav`、Tanker `2.wav` 和 Tug `9.wav`。每类通过 `audioread` 读取，多声道按 `mean(x,2)` 转为单声道；从每类信号前部依次截取 200 个不重叠的 2048 点片段，索引为 `idx = (i-1)*2048 + (1:2048)`，长度不足则报错。

每个原始 2048 点片段独立按 `tau=1:10` 做非重叠均值粗粒化，尾部不足一组的点舍弃。因此 10 个尺度长度依次为 2048、1024、682、512、409、341、292、256、227、204；每个粗粒化序列计算一个 Petrosian 分形维数，构成 `800×10` 的 `features`，同时生成 `800×1` 的 `labels`。程序对每个特征检查有限实数；异常会报告类别、样本、尺度及函数调用原因。

分类保持四类既有规则：`rng(42)` 后每类随机选 100 个训练片段，其余 100 个测试；`fitcknn` 使用全部 10 列特征、`NumNeighbors=1` 和欧氏距离。输出 Cargo、Passengership、Tanker、Tug 各类识别率、总体识别率，以及“行=真实类别、列=预测类别”、类别顺序与输入一致的 4×4 混淆矩阵。

```mermaid
flowchart TD
    A[四个指定 DeepShip WAV] --> B[audioread；多声道 mean(x,2)]
    B --> C[每类前 200 个非重叠 2048 点片段]
    C --> D[每片段独立 tau=1:10 非重叠均值粗粒化]
    D --> E[每尺度调用原有 Petrosian_FD]
    E --> F[800x10 特征；800x1 标签]
    F --> G[rng(42)；每类 100 训练和 100 测试]
    G --> H[1-NN；欧氏距离；全部 10 列]
    H --> I[四类和总体识别率；4x4 混淆矩阵]
```

更新时间（UTC+08:00）：2026-10-27 20:34。操作：新增并运行十尺度 Petrosian 分类实验；影响文件：`v3_ds/ds_petrosian.m`、`README.md`；验证状态：`EXECUTED`。输出 `features=800×10`、训练集 `400×10`、测试集 `400×10`；Cargo、Passengership、Tanker、Tug 识别率和总体识别率均为 100.00%，混淆矩阵为 `diag([100,100,100,100])`。该新增流程未调用随机共振、RK4 或相关函数。

未纳入公开版本控制的本地目录包括 `DeepShip-main/`、`v2/音频/`、`backup/`、`OG/` 和 `材料/`。其中前两者含原始或本地音频；`backup/` 与 `OG/` 是历史副本，不是正式入口或独立实验版本。

## 程序流程图 / Program Workflow

根目录四类脚本使用代码中写死的相对 WAV 路径；`v2` 脚本在五个类别目录中递归寻找且要求恰有一个受支持音频文件。两类脚本均先调用 `audioread`，多声道则按列均值转单声道，检查每类至少有 `200 × 2048` 个采样点，再按类别循环切分前 200 个非重叠片段。

```mermaid
flowchart TD
    S[选择根目录四类脚本或 v2 五类脚本] --> I{输入文件定位}
    I -->|根目录四类| R1[固定相对 WAV 路径]
    I -->|v2 五类| R2[FindSupportedAudioFile<br/>每类目录必须恰有一个受支持音频]
    R1 --> A[audioread]
    R2 --> A
    A --> M{多声道?}
    M -->|是| MONO[mean(x, 2) 转为单声道]
    M -->|否| LEN[保持原信号]
    MONO --> LEN[检查每类长度至少 200 x 2048]
    LEN --> SEG[按类循环 200 次<br/>截取非重叠 2048 点片段]
    SEG --> ALG{所选特征版本}
    ALG -->|CI-Higuchi + SR| SR[RK4: Rung_kutta_JDSR_k(a,b,h,x)]
    SR --> Z[按均值和标准差标准化]
    Z --> CI[CI_Higuchi<br/>kmax=20, alpha=0.95]
    ALG -->|CI-Higuchi 无 SR| CI0[CI_Higuchi<br/>直接使用原始片段]
    ALG -->|Katz + SR，仅 v2| KSR[RK4 后标准化]
    KSR --> K[Katz_FD 行向量]
    ALG -->|Katz 无 SR| K0[Katz_FD 行向量]
    ALG -->|FX 无 SR| FX[FX 片段, cellmax=2048]
    CI --> F[每片段一个标量特征<br/>并写入类别标签]
    CI0 --> F
    K --> F
    K0 --> F
    FX --> F
    F --> SPLIT[rng(42)<br/>每类随机选 100/200 片段训练]
    SPLIT --> KNN[fitcknn: 1 邻居、欧氏距离]
    KNN --> OUT[predict<br/>输出各类/总体识别率和混淆矩阵]
```

分支差异完全来自当前代码：根目录含四类 CI-Higuchi（有/无随机共振）、Katz（无随机共振）和 FX（无随机共振）；`v2` 扩展为五类，并额外含 `ci_fd_katz_sr.m`。所有脚本将结果显示在 MATLAB 命令窗口和工作区；代码没有写出结果文件的语句。

## 算法与参数 / Algorithms & Parameters

| 项目 | 实际设置与含义 | 源代码入口 |
|---|---|---|
| CI-Higuchi | `kmax=20`，计算尺度 `1:20`；`alpha=0.95`，用于按 t 分布置信区间剔除正长度序列中的离群值 | `ci_hfd.m`、`ci_hfd_no_sr.m`、`v2/ci_hfd.m`、`v2/ci_hfd_no_sr.m` |
| 传统 Higuchi | `Traditional_Higuchi(x,kmax)` 定义在 CI-Higuchi 脚本中；当前主流程不将其写入分类特征 | 同上 |
| 随机共振与 RK4 | `a=10`、`b=1`、`h=0.01`；RK4 迭代项为 `a*x - b*x^3 + input`，初始输出向量为零 | `ci_hfd.m`、`v2/ci_hfd.m`、`v2/ci_fd_katz_sr.m`、`Rung_kutta_JDSR_k.m` |
| Katz | 调用前转为行向量；`Katz_FD.m` 没有生效的 `Kmax` 参数 | `ci_fd_katz.m`、`v2/ci_fd_katz*.m`、`对比算法/Katz_FD.m` |
| 计盒维数 | `FX(x,cellmax)`，`cellmax=2048`；代码要求其不小于输入长度 | `ci_fd_fx.m`、`v2/ci_fd_fx.m`、`对比算法/FX.m` |
| 音频预处理 | `audioread`；多声道 `mean(x,2)`；未重采样，采样率由 `~` 接收且未参与后续计算 | 所有分类脚本 |
| 分段与特征 | 每类前 200 个连续、不重叠的 2048 点片段；每片段生成一个标量特征。根目录为 `800×1`（4×200），`v2` 为 `1000×1`（5×200） | 所有分类脚本 |
| 分类器 | `rng(42)`；每类随机选 100 个片段训练、其余 100 个测试；`fitcknn(...,'NumNeighbors',1,'Distance','euclidean')` | 所有分类脚本 |

完整的逐项参数索引见 [PARAMETERS.md](PARAMETERS.md)。代码的片段级划分会使同一源录音的片段同时进入训练和测试；它不是源文件级独立评估的声明。

`v2/ci_fd_katz.m`、`v2/ci_fd_katz_sr.m` 与 `v2/ci_fd_fx.m` 实际构造的查找路径是 `v2/对比算法`，而归档实现位于仓库根目录 `对比算法/`。这是静态检查发现的本地配置边界；本次未修改历史算法脚本，运行前须由使用者明确配置 MATLAB 路径或目录布局并自行验证。

## 已完成的操作 / Completed Work

状态含义：`IMPLEMENTED`=实现存在；`EXECUTED`=有可核验运行记录；`VERIFIED`=有可核验验证结果；`PENDING`=尚待执行或验证；`ABORTED`=已中止；`SUPERSEDED`=被替代。当前 Git 历史仅含本地代码归档提交，未含 MATLAB 运行记录或实验结果。

| 项目 | 状态 | 可核验依据与边界 |
|---|---|---|
| 四类 CI-Higuchi、随机共振、RK4、KNN 实现 | `IMPLEMENTED`; `PENDING`（执行/验证） | 入口 `ci_hfd.m` 与 `Rung_kutta_JDSR_k.m` 已跟踪；没有提交的运行输出或结果文件。 |
| 四类无 SR、Katz、FX 对比实现 | `IMPLEMENTED`; `PENDING`（执行/验证） | `ci_hfd_no_sr.m`、`ci_fd_katz.m`、`ci_fd_fx.m` 已跟踪；没有可核验运行记录。 |
| 五类 `v2` CI-Higuchi、Katz（含/不含 SR）、FX 实现 | `IMPLEMENTED`; `PENDING`（执行/验证） | `v2/` 中对应脚本已跟踪；比较算法路径需人工配置。 |
| 性能指标、分类结果和结果文件 | `PENDING` | 仓库明确排除原始数据与结果，不编造准确率或性能。 |
| `backup/`、`OG/` 历史副本 | `SUPERSEDED`（归档边界） | 已在 `.gitignore` 排除；不是公开主线或独立实验。 |
| 已中止实验 | 无可核验条目 | 当前 Git 历史、公开文档和已跟踪代码中未发现可据实标记为 `ABORTED` 的记录。 |

## 运行与复现 / Reproduction

1. 使用 MATLAB，并确保可用 `audioread`、`tinv`、`fitcknn` 和 `confusionmat`（通常需要 Statistics and Machine Learning Toolbox）。
2. 仅使用你有权使用的音频。根目录四类脚本需要代码所列的 `DeepShip-main/<类别>/<文件>.wav` 相对路径；`v2` 需要五个 `v2/音频/<类别>/` 目录，并且每个目录递归后恰有一个受支持音频文件。
3. 确保每类音频至少含 `200×2048` 个采样点；脚本不会重采样，且不使用采样率变量。两个已更新的 `v2` 脚本将每段独立按 `tau=1:10` 粗粒化；Higuchi 在最短的 204 点尺度 10 输入上仍以 `kmax=20` 计算，FX 则使用 2048、1024、512、512、256、256、256、256、128、128 点截取及同值 `cellmax`。
4. 选择一个入口：`ci_hfd.m` 为根目录随机共振 CI-Higuchi 主程序；其余脚本是上表所列对比版本。不要把多个脚本的工作区结果混作同一次实验。
5. 执行前核对参数表和 `v2` 对比算法路径说明。执行后，当前脚本只在命令窗口显示识别率/混淆矩阵，并在工作区保留变量；不自动生成结果文件。

2026-10-27 的多尺度更新已实际运行 MATLAB。两条执行路径都直接从音频片段进行粗粒化与特征提取，未调用 `Rung_kutta_JDSR_k` 或其他随机共振求解函数，也未保留随机共振参数。MATLAB `checkcode` 未报告错误（仅有原有未调用 `Traditional_Higuchi` 的提示）；运行时逐特征拒绝非有限或复数值。FX 运行中记录到原始 `FX.m` 的若干 `polyfit` 秩亏警告，但程序完成且未有 NaN/Inf 进入 KNN。

## 更新日志 / Development Changelog

| 更新时间（UTC+08:00） | 操作类型 | 更新内容 | 影响文件 | 状态 |
|---|---|---|---|---|
| 2026-10-27 15:59 | docs | 建立持续维护的 GitHub 首页：基于已跟踪 MATLAB 源码补充真实目录、Mermaid 数据流、参数、复现边界、实验状态与 Git 更新日志；新增项目维护规则。未运行实验。 | `README.md`, `AGENTS.md` | 已完成（文档）；实验待验证 |
| 2026-10-27 15:49 | docs | 创建公开代码归档、参数/实验说明与排除规则；该时间来自 commit `c363805`。 | `.gitignore`, `README.md`, `PARAMETERS.md`, `EXPERIMENTS.md`, MATLAB 源文件 | 已完成（本地归档）；远端备份待验证 |

## 本次 v2 多尺度更新 / v2 Multiscale Update

2026-10-27 19:51（UTC+08:00）实际修改 `v2/ci_fd_fx.m` 和 `v2/ci_hfd_no_sr.m`：每个独立的 2048 点片段均按 `tau=1:10` 作非重叠均值粗粒化，`y_tau(j)=mean(x((j-1)*tau+1:j*tau))`，尾部不足一个窗口的点舍弃。每片段由 1 维分形特征变为 10 维，五类共 1000 个样本的 `features` 为 `1000×10`，标签仍为 `1000×1`。

FX 使用原有 `FX.m`：尺度 1 仍直接传入原始 2048 点并使用 `cellmax=2048`；其他尺度使用不超过粗粒化长度的最大 2 的整数次幂截取，并同步设置 `cellmax`。CI-Higuchi 保持原有 `kmax=20`（算法内部参数，而非粗粒化尺度数）和 `alpha=0.95`。两条路径均保持 `rng(42)`、每类 100/100 训练测试划分、1-NN 和欧氏距离；未调用随机共振求解函数或参数。

MATLAB `checkcode` 未报告错误（仅有原有未调用 `Traditional_Higuchi` 的提示）。两条程序均完成实际运行，且逐特征有限实数检查未报错：FX 总体识别率 79.80%，CI-Higuchi 总体识别率 77.20%。FX 运行时有原始 `FX.m` 的 `polyfit` 秩亏警告；程序未产生 NaN/Inf 并完成 KNN 分类。

```mermaid
flowchart TD
    A[五类音频] --> B[audioread；多声道 mean(x,2)]
    B --> C[每类前 200 个不重叠 2048 点片段]
    C --> D[每个片段独立进行 tau=1:10 非重叠均值粗粒化]
    D --> E{算法}
    E --> F[FX：截取最大 2 的幂并同步 cellmax]
    E --> G[CI-Higuchi：每尺度 kmax=20]
    F --> H[1000x10 特征与 1000x1 标签]
    G --> H
    H --> I[rng(42)；每类 100 训练 / 100 测试]
    I --> J[1-NN；欧氏距离；识别率和混淆矩阵]
```
## v3_ds 十尺度更新 / v3_ds 10-Scale Update

更新时间（UTC+08:00）：2026-10-27 19:52。`v3_ds/ds_fx.m`、`v3_ds/ds_higuchi.m` 和 `v3_ds/ds_katz.m` 处理实际的四类音频（Cargo、Passengership、Tanker、Tug）。保留 `audioread`、多声道 `mean(x,2)`、每类前 200 个非重叠 2048 点片段、`rng(42)`、每类 100/100 训练测试划分、1-NN 欧氏距离及原有准确率输出。

每个独立片段直接按 `tau=1:10` 作非重叠分组均值：`y_tau(j)=mean(x((j-1)*tau+1:j*tau))`，尾部不足一组的点舍弃。尺度长度依次为 2048、1024、682、512、409、341、292、256、227、204；每个样本得到同一算法的 10 维特征，三个脚本的 `features` 均为 `800×10`，`labels` 为 `800×1`。FX、CI-Higuchi、Katz 的原有数学定义与参数保持不变；FX 对每尺度取不小于输入长度的最小 2 的幂 `cellmax`，尺度 1 仍为 2048。完整信号路径不调用 `Rung_kutta_JDSR_k`、双稳态随机共振或其数值积分。

```mermaid
flowchart LR
 A[四类原始音频] --> B[原有预处理与 2048 点分段]
 B --> C[tau=1:10 独立非重叠粗粒化]
 C --> D[10 个同算法分形维数]
 D --> E[每片段 10 维特征]
 E --> F[原有划分与 1-NN KNN]
 F --> G[四类与总体识别率]
```

已实际验证：MATLAB 静态分析无语法问题（CI-Higuchi 保留原有未调用 `Traditional_Higuchi` 提示）；独立 2048 点检查确认尺度 1 逐元素一致、尺度 10 为 204 点，10 个尺度的 FX 与 Katz 结果均为有限实数。`v3_ds` 的预期本地 `DeepShip-main/` 路径不存在，故未执行完整分类，未填写准确率前后比较。本次提交记录：`feat: add 10-scale coarse-grained FD features to ds classifiers`。