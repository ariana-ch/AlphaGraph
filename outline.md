# MSc Thesis Outline: A Multi-Modal Deep Learning Framework for Dynamic Portfolio Optimization

**Title:** Integrating Temporal Knowledge Graphs and Market Data for Dynamic Equity Portfolio Construction

---

### **Abstract**
**(Approx. 0.5 pages)**

A concise summary of your entire project. State the problem (limitations of unimodal portfolio models), your proposed solution (a hybrid model fusing news-derived knowledge graphs and price data), the methodology (PatchTST/LSTM/NBEATSx with a KGTransformer and cross-attention), key findings (e.g., improved Sharpe ratio, controlled cardinality), and the main contribution (a novel, end-to-end differentiable framework for constructing sparse, tradable portfolios).

---

### **Chapter 1: Introduction**
**(Approx. 3-4 pages)**

**1.1. Background and Motivation**:
* Start with the foundational challenge of portfolio construction: balancing risk and return.
* Introduce the limitations of traditional models that rely solely on structured price data.
* Motivate the need for incorporating unstructured data (news) to capture market sentiment and event-driven dynamics that prices alone cannot reflect.

**1.2. Problem Statement**:
* Clearly state the primary problem: How to systematically integrate unstructured news data with structured market data to construct portfolios that are not only profitable but also tradable (e.g., with controlled cardinality and rebalancing frequency).

**1.3. Research Objectives**:
* Construct sparse equity portfolios (~40 stocks) rebalanced every 5 days.
* Develop a novel, end-to-end differentiable model that fuses time-series price data with a dynamic knowledge graph from news.
* Design and implement a custom loss function to learn portfolio sparsity, control volatility, and optimize a mean-variance utility.
* Empirically validate the performance of the hybrid model against strong price-only baselines.

**1.4. Contributions**:
* Novelty of the hybrid architecture (KGTransformer + Price Model + Cross-Attention).
* The design of a sophisticated, differentiable loss function for portfolio construction.
* An end-to-end pipeline from raw news/price data to optimized portfolio weights.

**1.5. Thesis Structure**:
* Briefly outline the subsequent chapters.

---

### **Chapter 2: Literature Review**
**(Approx. 6-8 pages)**

**2.1. Foundations of Portfolio Theory**:
* Briefly cover Markowitz's Mean-Variance Optimization and the Sharpe Ratio.

**2.2. Machine Learning for Portfolio Construction**:
* Discuss how ML models (LSTMs, Transformers) are used for return prediction and direct portfolio optimization.
* Review relevant time-series forecasting models you used as baselines: **PatchTST**, **Bidirectional LSTMs**, and **NBEATSx**.

**2.3. Unstructured Data in Finance**:
* Review literature on using news sentiment for market prediction.
* Introduce the concept of **Knowledge Graphs (KGs)** in finance.
* Discuss **temporal/dynamic KGs** and review the key inspiration for your model: **EvoKG**.

**2.4. Multi-Modal Fusion Techniques**:
* Discuss methods for combining heterogeneous data, focusing on **attention mechanisms**, specifically cross-attention, which you've used to link the two data streams.

---

### **Chapter 3: Data and Preprocessing**
**(Approx. 4-5 pages)**

**3.1. Data Collection**:
* **Structured Data**: Detail the source, date range (Jan 2020 - Aug 2025), and scope (512 liquid NASDAQ stocks) of the price and volume data.
* **Unstructured Data**: Describe the source and collection process for the news articles.

**3.2. Feature Engineering (Structured Data)**:
* Explain the creation of the **45 technical indicators** from raw price/volume data. List a few key examples (e.g., RSI, MACD, Moving Averages).

**3.3. Knowledge Graph Construction (Unstructured Data)**:
* Detail the process of using **Gemini** to extract knowledge triplets.
* Describe the schema and prompt engineering used to ensure consistent and relevant extractions. This is a key methodological step.

**3.4. Dataset Preparation**:
* Explain how you created the training, validation, and testing sets.
* Crucially, detail the difference in batching: **sequential overlapping periods for training** and **non-overlapping periods for validation/testing**.

---

### **Chapter 4: Methodology**
**(Approx. 8-10 pages)**

**4.1. Problem Formulation**:
* Mathematically define the task: mapping a sequence of market and news data to a vector of portfolio weights.

**4.2. Phase 1: Price-Only Models (Baselines)**:
* Detail the architecture of the **PatchTST**, **Bi-LSTM**, and **NBEATSx** models.
* Input Tensor Shape: `batch x 30 x 512 x d_features`.
* Output Tensor Shape: `512 x 1`.

**4.3. Phase 2: Hybrid Model Architecture**:
* **4.3.1. The KGTransformer**:
    * Explain the architecture in detail: static and dynamic embeddings, the role of **RGCN** for structural convolution, and the **RNN/GRU cell** for temporal evolution.
    * Mention the decay mechanism for dynamic embeddings.
* **4.3.2. Cross-Attention Fusion**:
    * This is the core of your model. Explain how the price model's hidden state attends to the KG embeddings.
    * Specify the query, key, and value tensors in your attention mechanism. Emphasize the intuition: "prices learn about the market from the news."
* **4.3.3. Output Layer**:
    * Describe the final MLP and the use of **entmax15 activation** to enforce differentiable sparsity.

**4.4. Custom Loss Function**:
* This is a major contribution. Detail each component:
    * **Mean-Variance Utility**: Explain the formula and the role of the risk aversion coefficient.
    * **Differentiable Cardinality Penalty**: Explain how you designed this function to penalize deviations from the target of ~40 stocks.
    * **Volatility Control Penalty**: The term with a target volatility.
    * **Small Weight Penalty**: The term penalizing negligible weights.
* Explain why you moved away from the negative Sharpe ratio (instability).

**4.5. Training and Evaluation Strategy**:
* **Training Loop**: Describe the training process.
* **Early Stopping**: Mention that it's based on the validation Sharpe ratio.
* **Evaluation Metrics**: List and define all metrics used: number of components, turnover, cumulative return, Sharpe, max drawdown, and Calmar ratio.

---

### **Chapter 5: Experiments and Results**
**(Approx. 6-8 pages)**

**5.1. Experimental Setup**:
* Hardware, software libraries (PyTorch, etc.), and key hyperparameters for all models.

**5.2. Baseline Performance**:
* Present the results of the three price-only models on the test set using your evaluation metrics. Use clear tables and charts (e.g., a cumulative return plot).

**5.3. Hybrid Model Performance**:
* Present the results of your full hybrid model.
* Directly compare its performance against the baselines, highlighting the "alpha" or improvement gained from incorporating news data.

**5.4. Portfolio Analysis**:
* Show how the portfolio composition evolves over time.
* Analyze specific periods (e.g., high market volatility) to see what stocks the model selected and how it performed. This provides qualitative insight into your model's behavior.

---

### **Chapter 6: Discussion and Future Work**
**(Approx. 3-4 pages)**

**6.1. Analysis of Results**:
* Interpret the findings. Was the hybrid model significantly better? Why?
* Discuss the effectiveness of your custom loss function. Did it successfully control the number of stocks and volatility?
* Discuss the role of the cross-attention mechanism.

**6.2. Ablation Studies (Proposed)**:
* As you mentioned, you can advise on this. Suggest studies that could be performed to isolate the impact of different components:
    * Removing the KG input to quantify its value.
    * Removing individual components of the loss function (e.g., the cardinality penalty).
    * Testing different activation functions instead of `entmax15`.

**6.3. Limitations**:
* Acknowledge limitations, such as the dependency on a single news source and the computational complexity. Mention the issue of not having enough news coverage to build a portfolio from the graph alone.

**6.4. Future Work**:
* Suggest extensions: incorporating more data sources, exploring different graph network architectures, or applying the framework to other asset classes.

---

### **Chapter 7: Conclusion**
**(Approx. 1-2 pages)**

* Succinctly summarize the thesis. Reiterate the problem, your innovative solution, the key results, and the main takeaways. End with a strong concluding statement about the potential of multi-modal deep learning in quantitative finance.

---

### **References**
* List all academic papers, books, and software libraries cited.

---

### **Appendices**
*(Does not count towards the 40-page limit)*

* **Appendix A: Technical Indicators**: A full list and brief description of the 45 indicators used.
* **Appendix B: Hyperparameter Details**: Detailed tables of all hyperparameters used for training each model.
* **Appendix C: Gemini Prompting for KG Extraction**: The exact prompt and schema used for triplet extraction.
* **Appendix D: Additional Plots and Tables**: Any supplementary results or visualizations that are useful but not essential for the main body.
* **Appendix E: Code Snippets**: Key snippets of your code, especially the custom loss function or the cross-attention module implementation.