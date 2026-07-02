# SMS Spam Classifier — AI Assignment (Supervised ML)

## 项目结构
```
spam-classifier-project/
├── data/
│   └── SMSSpamCollection      ← 你需要自己下载放这里 (见下方 Step 1)
├── src/
│   ├── prepare_data.py        ← 跑一次:清洗+切分+向量化,两人共用
│   ├── train_naive_bayes.py   ← 你负责
│   ├── train_svm.py           ← 队友负责
│   ├── compare_models.py      ← 汇总两人结果做对比表
│   └── app.py                 ← Streamlit UI (prototype demo用)
├── models/                    ← 训练完自动生成 (模型文件、图表、对比表)
└── requirements.txt
```

## Step 1: 下载数据集
1. 打开 https://archive.ics.uci.edu/dataset/228/sms+spam+collection
2. 点 "Download" 下载 zip,解压
3. 把解压出来的 `SMSSpamCollection` 文件放进 `data/` 文件夹

## Step 2: 安装依赖
在项目根目录打开终端:
```bash
pip install -r requirements.txt
```

## Step 3: 跑流程(按顺序)
```bash
# 1. 准备数据(只需跑一次，之后两人不用重跑，除非改了预处理逻辑)
python src/prepare_data.py

# 2. 你训练 Naive Bayes
python src/train_naive_bayes.py

# 3. 队友训练 SVM
python src/train_svm.py

# 4. 汇总对比 (两人的模型都跑完之后再跑这个)
python src/compare_models.py

# 5. 启动网页 Demo
streamlit run src/app.py
```

## 跑完会得到什么
- `models/naive_bayes_model.joblib`, `models/svm_model.joblib` — 训练好的模型
- `models/confusion_matrix_nb.png`, `models/confusion_matrix_svm.png` — 各自的混淆矩阵图
- `models/comparison_table.csv` — 两个模型的 accuracy/precision/recall/F1 对比表 → **直接贴进 Documentation 的 Results 部分**
- `models/comparison_chart.png` — 对比柱状图 → **同样贴进报告**
- `streamlit run src/app.py` 打开的网页 → **demo 展示用**

## 注意事项
- `prepare_data.py` 里的 `RANDOM_STATE = 42` 千万别改,两人必须用同一个才能公平对比
- 如果想加 TF-IDF 版本做对比，把 `CountVectorizer` 换成 `TfidfVectorizer`（sklearn 用法几乎一样），可以另存一份 `prepare_data_tfidf.py` 做加分对比
- AI Disclosure Statement 记得写：用 Claude 协助搭建了项目结构和代码框架
