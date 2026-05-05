# 机器学习实验：基于 Word2Vec 的情感预测

## 1. 学生信息

- **姓名**：李佳霖
- **学号**：112304260101
- **班级**：

> 注意：姓名和学号必须填写，否则本次实验提交无效。

---

## 2. 实验任务

本实验基于给定文本数据，使用 Word2Vec 将文本转为向量特征，再结合分类模型完成情感预测任务，并将结果提交到 Kaggle 平台进行评分。

本实验重点包括：

- 文本预处理（去 HTML、清洗、转小写等）
- Word2Vec 词向量训练与句向量/文档向量表示
- 分类模型训练（Logistic Regression 等）
- 更强基线：TF-IDF（词/字符 n-gram）+ 线性模型
- 集成：stacking / 融合

---

## 3. 比赛与提交信息

- **比赛名称**：Word2Vec NLP Tutorial（Bag of Words Meets Bags of Popcorn）
- **比赛链接**：https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview
- **提交日期**：2026-04-24（Asia/Shanghai）
- **GitHub 仓库地址**：https://github.com/Li929-study/112304260101lijialin
- **GitHub README 地址**：https://github.com/Li929-study/112304260101lijialin/blob/main/README.md

> 注意：GitHub 仓库首页或 README 页面中，必须能看到"姓名 + 学号"，否则无效。

---

## 4. Kaggle 成绩

- **Public Score**：0.96964
- **Private Score（如有）**：0.96964
- **排名（如能看到可填写）**：无
- **最终推荐提交文件**：Word2Vec_Embedding_Logistic.csv

---

## 5. Kaggle 截图

请在下方插入 Kaggle 提交结果截图，要求能清楚看到分数信息。

建议将截图保存在 images 文件夹中。

截图文件名示例：`112304260101_李佳霖_kaggle_score.png`

![Kaggle截图](./images/112304260101_李佳霖_kaggle_score.png)

---

## 6. 实验方法说明

### （1）文本预处理

请说明你对文本做了哪些处理，例如：

- 分词
- 去停用词
- 去除标点或特殊符号
- 转小写

**我的做法：**

1. 使用 BeautifulSoup 去除 HTML 标签（保留纯文本）
2. 将非字母字符替换为空格（仅保留 A-Z / a-z）
3. 统一转小写并合并多余空白
4. **保留否定词**：不移除 "not"、"don't" 等否定词，因为否定词对情感分类至关重要
5. 在部分实验中，对否定词后的词添加 `NOT_` 前缀标记（如 "not good" → "NOT_good"），增强否定语义的表达

---

### （2）Word2Vec 特征表示

请说明你如何使用 Word2Vec，例如：

- 是自己训练 Word2Vec，还是使用已有模型
- 词向量维度是多少
- 句子向量如何得到（平均、加权平均、池化等）

**我的做法：**

- 使用 gensim 训练 Word2Vec 词向量（合并 labeledTrainData + unlabeledTrainData + testData 语料，共约 75000 条评论）
- 词向量维度：**300 维**
- 训练参数：`vector_size=300, window=10, min_count=2, sg=1, epochs=5, seed=42`（Skip-gram 模型）
- 文档向量表示：对文档内词向量取平均（average pooling）得到固定维度向量
- 在最终方案中，Word2Vec 特征作为基础特征之一参与融合

---

### （3）分类模型

请说明你使用了什么分类模型，例如：

- Logistic Regression
- Random Forest
- SVM
- XGBoost

并说明最终采用了哪一个模型。

**我的做法：**

本实验最终采用 **TF-IDF + NBSVM + Word2Vec + OOF Stacking 融合方案**，而非单一模型。具体如下：

#### TF-IDF 基础模型（11个）

| 模型 | 特征 | 参数 | 5折CV AUC |
|------|------|------|-----------|
| NBSVM | TF-IDF word+char (1,3)+(2,5) | alpha=0.01, C=50 | 0.97341 |
| NBSVM | TF-IDF word+char (1,3)+(2,5) | alpha=0.01, C=30 | 0.97285 |
| NBSVM | TF-IDF word+char (1,3)+(2,5) | alpha=1.0, C=10 | 0.97012 |
| LR | TF-IDF word+char (1,3)+(2,5) | C=5 | 0.96712 |
| LR | TF-IDF word+char (1,3)+(2,5) | C=10 | 0.96833 |
| LR | TF-IDF word+char (1,3)+(2,5) | C=30 | 0.96655 |
| SGDClassifier | TF-IDF word+char (1,3)+(2,5) | log_loss, alpha=1e-5 | 0.96502 |
| SGDClassifier | TF-IDF word+char (1,3)+(2,5) | log_loss, alpha=1e-6 | 0.96428 |
| MultinomialNB | TF-IDF word+char (1,3)+(2,5) | alpha=0.01 | 0.95870 |
| MultinomialNB | TF-IDF word+char (1,3)+(2,5) | alpha=0.1 | 0.95520 |
| BernoulliNB | TF-IDF word+char (1,3)+(2,5) | alpha=0.01 | 0.95340 |

#### Word2Vec 模型

| 模型 | 特征 | 参数 | 5折CV AUC |
|------|------|------|-----------|
| LR | Word2Vec Average Embedding (300维) | C=5 | ~0.88 |

#### 融合方式

- **Stacking 元学习器**：用 LR 对 TF-IDF 11个模型 + Word2Vec 模型的 OOF 预测概率做 Stacking → OOF AUC **0.97304**
- **加权融合（Weighted Blend）**：按各模型 CV AUC 加权平均 → OOF AUC **0.97103**
- **rank_mean 融合**：对概率值做排名平均 → 稳健融合方案

#### 最终选择

最终采用**元学习器 Stacking**方案，Kaggle Public Score = **0.96964**。

---

## 7. 实验流程

1. 从 Kaggle 下载并解压数据（labeledTrainData.tsv、testData.tsv、unlabeledTrainData.tsv）
2. 文本预处理（去 HTML、清洗、保留否定词、添加 NOT_ 标记）
3. 训练 Word2Vec 词向量（使用全部语料），生成均值 Embedding 句向量
4. 提取多种 TF-IDF 特征：word (1,3) + char_wb (2,5)，合并为词级+字符级组合特征
5. 使用分层5折交叉验证，分别训练11个 TF-IDF 基础模型和1个 Word2Vec LR 模型
6. 收集各模型的 OOF（Out-of-Fold）预测概率和测试集预测概率
7. 对所有模型进行元学习器 Stacking 融合，生成最终预测概率
8. 使用 blend_submissions.py 进行 rank_mean/logit_mean/mean 多种融合对比
9. 在测试集上输出正类概率，生成 Word2Vec_Embedding_Logistic.csv
10. 上传 Kaggle 获取 ROC AUC 分数，迭代优化并做融合

---

## 8. 文件说明

**我的项目结构：**

```text
E:\机器3\
├─ labeledTrainData.tsv\              # 训练数据（25000条）
├─ testData.tsv\                      # 测试数据（25000条）
├─ unlabeledTrainData.tsv\            # 无标签数据（50000条）
├─ src\
│  ├─ make_submission_tfidf.py        # 传统/集成方案（TF-IDF word+char, 11个模型, stacking）
│  ├─ make_submission_w2v.py          # Word2Vec 特征方案（训练+均值向量+LR）
│  └─ blend_submissions.py            # 融合多个 submission（mean/rank_mean/logit_mean）
├─ artifacts\                         # 传统模型缓存（word2vec / tfidf / oof预测）
├─ images\                            # README 截图（Kaggle 成绩截图）
│  └─ 112304260101_李佳霖_kaggle_score.png
├─ optimized_pipeline.py              # 主实验代码（TF-IDF + NBSVM + Stack融合）
├─ final_fusion_pipeline.py           # 完整融合代码（NBSVM + TF-IDF LR + W2V + OOF融合）
├─ Word2Vec_Embedding_Logistic.csv    # 最终提交文件（25000行，概率值）
├─ submission_final.csv               # 提交文件副本
├─ sub_*.csv                          # 各种提交文件（Kaggle 上传用）
├─ sampleSubmission.csv               # 提交样例
└─ README.md                          # 实验报告
```

---

## 9. 复现方式（命令）

### 9.1 安装依赖

```bash
cd E:\机器3
pip install pandas numpy scikit-learn gensim beautifulsoup4 scipy joblib
```

### 9.2 传统强基线（TF-IDF + NBSVM + Stacking）

```bash
python .\optimized_pipeline.py
```

输出文件：`sub_tfidf_stack.csv`（Stacking 融合）、`sub_tfidf_weighted.csv`（加权融合）

### 9.3 Word2Vec 特征方案

```bash
python .\src\make_submission_w2v.py
```

输出文件：`sub_w2v_avg.csv`

### 9.4 完整融合 Pipeline（TF-IDF + Word2Vec + Stacking）

```bash
python .\final_fusion_pipeline.py
```

输出文件：`Word2Vec_Embedding_Logistic.csv`（最终推荐提交）

### 9.5 融合提交（AUC 常用 rank_mean）

```bash
python .\src\blend_submissions.py --method rank_mean --out .\submission_blend.csv --inputs .\sub_tfidf_stack.csv .\sub_w2v_avg.csv
```
