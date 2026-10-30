# FLDE 配图审查与本项目实现映射

参考：`论文/FLDE.docx`。直接提取四个内嵌 EMF，并查看白底 PNG 渲染；对照图注及实验正文。本文是配图审查记录，不是第二份 README。

## 观察方法与精度

原图为矢量 EMF，原始像素尺寸不能等同于最终印刷尺寸。观察渲染统一宽1800像素：柱图约1800×867，散点综合图约1800×2492。逐个解析EMF的字体记录，四图均明确使用 Times New Roman。Word XML中图1—4的嵌入尺寸依次为8.788×4.230、8.788×12.164、8.788×4.228、8.788×12.169厘米。字号、线宽、透明度和RGB为视觉匹配参数，并非已恢复的原始绘图源码参数；原图字号尚未精确还原。

## 图1与图3：平均识别率

- 单面板宽幅，白底，无图内总标题，无图例；数据集名及 Figure 编号在 Word 图注中。
- 四边细深灰边框，刻度向内，上、右边可见对应刻度；浅灰横纵网格。
- 纵轴原文 `Recognition rate (%)`，范围70至约103，刻度70、75、80、85、90、95、100。横轴只有方法缩写，没有轴标题；文字约逆时针30度。
- 八个等距柱，宽约相邻中心距的0.72；填充为低饱和砖红、灰绿、浅绿、灰紫、灰蓝、卡其、棕、蓝灰，深灰细边。
- 黑色竖向误差棒，有水平端帽。柱顶标注 `均值±标准差`，均为两位小数，不另带百分号；文字居中，位于误差棒上方。
- 方法顺序：FO-LDE、MS-LDE、FO-SLOE、MS-SLOE、FO-GE、MS-GE、FO-LZC、MS-LZC。

| 方法 | 图1 DeepShip：均值±标准差 | 图3 NPS：均值±标准差 |
|---|---|---|
| FO-LDE | 99.80±0.19 | 98.44±0.56 |
| MS-LDE | 96.06±0.84 | 97.50±0.77 |
| FO-SLOE | 98.39±0.56 | 75.19±1.73 |
| MS-SLOE | 96.71±0.77 | 74.42±1.65 |
| FO-GE | 83.49±1.35 | 84.72±1.26 |
| MS-GE | 82.01±1.49 | 96.28±0.67 |
| FO-LZC | 96.78±0.74 | 96.97±0.75 |
| MS-LZC | 95.45±0.60 | 94.65±0.77 |

## 图2与图4：t-SNE

- 四行两列，左FO、右MS，依次LDE、SLOE、GE、LZC。白底，子图无网格，有四边边框和向内刻度。
- 每个面板标题为 `方法名 (SC=0.000)`，常规粗细衬线字体，SC三位小数，负数保留负号。
- 无横纵坐标名称，无物理单位；各子图范围和刻度独立，并非所有面板共用范围。不可将二维方向或绝对位置解释为物理量。
- 类别标记依次为蓝色空心圆、橙色加号、黄色叉号、紫色空心方形；点没有连接线，圆和方形不填色。类别符号、颜色跨面板一致。
- 图底部居中放一条带矩形边框的横排图例，不覆盖子图；无图内数据集总标题。
- 图2图例原文：`Cargo`、`Passenger Ship`、`Tanker`、`Tug`。
- 图4图例原文：`Ferry`、`Cargo Ship`、`Small Diesel Engine`、`Outboard Motor`。

| 面板方法 | 图2 SC | 图4 SC |
|---|---:|---:|
| FO-LDE | 0.649 | 0.523 |
| MS-LDE | 0.482 | 0.168 |
| FO-SLOE | 0.347 | 0.082 |
| MS-SLOE | 0.264 | -0.015 |
| FO-GE | 0.219 | 0.177 |
| MS-GE | 0.117 | 0.241 |
| FO-LZC | 0.532 | 0.428 |
| MS-LZC | 0.411 | 0.198 |

图2 FO-LDE四类明显分开，部分其他方法存在类别交叠；图4 FO-LDE仍有局部接近，SLOE交叠较明显。仅依据散点形状不替代分类结果。参考正文没有明确SC计算空间，不能断言原作者是在二维或十维计算。

## Word图注原文（不在图片内部）

1. Figure 1. Average SVM recognition rates of the eight feature extraction methods on the DeepShip dataset.
2. Figure 2. t-SNE visualization of the eight feature groups on the DeepShip dataset.
3. Figure 3. Average SVM recognition rates of the eight feature extraction methods on the NPS dataset.
4. Figure 4. t-SNE visualization of the eight feature groups on the NPS dataset.

## 本项目实施与有意差异

- v2 = National Park，八类；v3_ds = DeepShip，四类。真实类别及顺序以原脚本音频路径为准，National Park不是参考论文中的四类子集。
- 使用四个仍存在的算法 Katz/Higuchi/FX/R/S，固定顺序，不恢复用户已删除的Petrosian/Dubuc，不伪造八种熵算法。
- 原始算法、音频、2048点、每类200段、均值粗粒化1:10保持不变。分类仍为原来的无标准化欧氏1-NN，未改为参考SVM。v2各方法10维；DeepShip Katz/Higuchi/FX保留第一列，R/S十列。
- 当前采用原脚本单次rng(42)划分，每类100训练/100测试，所有柱高逐一复现原预测与混淆矩阵。单次无标准差，不画误差棒，数字三位小数，柱状图无总标题，不在图内添加seed等实验设置文字。此前20次均值图已归档；98.375%与97.70625%分别属于单次与20次均值，不能混用。片段级划分不等同独立录音测试。
- t-SNE使用全部十维特征，无标准化，种子1042，perplexity30，不搜索更好看的种子，不按类别变换坐标。SC明确采用原始十维特征欧氏距离；标题SC不是二维嵌入质量分。
- 四方法用2×2子图，延续参考的字体、无网格、独立坐标范围、标题及底部带框图例；八类增加四种颜色/符号，图例为两行四列。
- 新结果单独存于各目录 `results_flde_style/`，不覆盖旧比较图。旧Higuchi脚本运行时的附带导出会临时备份并恢复原文件。
- PNG 300 dpi；PDF/SVG矢量；FIG可编辑。真实结果自动填写，不沿用上表论文数值。
- 两个原入口调用根目录共享 `run_flde_comparison`；full重提特征，results从features.mat重新单次分类/降维，plot仅从experiment.mat重新绘图并拒绝旧20次缓存。

## 本项目可用英文图注

1. Single-run 1-NN recognition rates (seed = 42; no between-run error bars) of four fractal-dimension methods on the National Park dataset.
2. t-SNE visualization of ten-dimensional fractal features on the National Park dataset. SC is computed in the original ten-dimensional feature space using Euclidean distance.
3. Single-run 1-NN recognition rates (seed = 42; no between-run error bars) of four fractal-dimension methods on the DeepShip dataset. Katz, Higuchi and FX retain the original first-scale classifier input; R/S uses ten scales.
4. t-SNE visualization of ten-dimensional fractal features on the DeepShip dataset. SC is computed in the original ten-dimensional feature space using Euclidean distance.

当前额外验收标准：图中八根柱的实际YData、三位小数标签及CSV必须等于原脚本单次预测准确率；不得存在误差棒；SC必须通过独立a(i)/b(i)距离公式复算；每一子图的逐类散点坐标必须等于保存的t-SNE坐标。National Park负SC保留。
