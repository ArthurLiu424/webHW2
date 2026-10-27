# HW2 報告書：以多元線性回歸預測洪水機率（CRISP-DM）

- **資料集**：Kaggle — Flood Prediction Dataset（naiyakhalid）
  <https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset>
  建模用 `train.csv`（即 Kaggle Playground Series S4E5，<https://www.kaggle.com/competitions/playground-series-s4e5>），隨機抽樣 50,000 筆。
- **特徵數**：20（符合 13 < 特徵數 < 40）
- **目標變數**：`FloodProbability`（連續值）
- **程式**：`analysis.py`；**網頁**：<https://arthurliu424.github.io/webHW2/>

---

## 1. Business Understanding（商業理解）

洪水是最常見、損失最大的天然災害之一。地方政府在分配防災預算時，以及保險公司在訂定保費時，都需要估計一個區域的洪水發生機率。這個估計必須**可解釋**：決策者想知道「改善排水系統 1 分，可以降低多少洪水機率」。

- **問題**：給定 20 項區域風險因子的評分，預測該區域的洪水機率。
- **方法**：多元線性回歸。係數就是各因子的邊際效果，容易溝通。
- **成功標準**：測試集 R² ≥ 0.80；95% 預測區間的實際覆蓋率接近 95%。

## 2. Data Understanding（資料理解）

| 項目 | 值 |
|---|---|
| 樣本數 | 50,000（抽樣自 train.csv 的 1,117,957 筆） |
| 特徵 | 20 個整數評分（0–18）：MonsoonIntensity、TopographyDrainage、RiverManagement、Deforestation、Urbanization、ClimateChange、DamsQuality、Siltation、AgriculturalPractices、Encroachments、IneffectiveDisasterPreparedness、DrainageSystems、CoastalVulnerability、Landslides、Watersheds、DeterioratingInfrastructure、PopulationScore、WetlandLoss、InadequatePlanning、PoliticalFactors |
| 目標 | FloodProbability，平均約 0.504 |
| 缺失值 / 重複列 | 0 / 0 |
| 特徵間最大 \|r\| | 0.025（幾乎互相獨立） |
| 各特徵與目標的 r | 全部為正，約 0.17–0.20 |

**重要發現**：一開始使用原始的 `flood.csv`，OLS 的 R² = **1.000000**。也就是說，目標幾乎是特徵的確定性線性函數，沒有雜訊，信賴區間和預測區間會縮成零寬度，失去統計推論的意義。因此改用同一資料集中、加入雜訊的合成版 `train.csv`。

圖：`figures/01_target_hist.png`、`02_corr_heatmap.png`、`03_corr_target.png`

## 3. Data Preparation（資料準備）

1. 移除 `id` 欄位，並從 train.csv 隨機抽樣 50,000 筆（seed = 2024），存成 `data/flood_s4e5_sample50k.csv`。
2. 沒有缺失值，也沒有類別變數，所以不需要補值或編碼。
3. 切分 80% 訓練（40,000 筆）/ 20% 測試（10,000 筆），seed = 42。
4. 特徵選擇與交叉驗證**只使用訓練集**，避免資料洩漏。
5. 只用訓練集的平均與標準差做 z-score 標準化，供 Lasso 與標準化係數使用。
6. 建立工程特徵 `FeatureSum`（20 個因子總和），作為對照組。

## 4. Modeling（建模與特徵選擇）

### 4.1 特徵選擇（四種方法）

| 方法 | 做法 | 結果 |
|---|---|---|
| Filter | 單變量 F-test（= `SelectKBest(f_regression)`） | 20 個 F 值都很大，差距小 |
| Wrapper | 前向逐步選擇，準則為 5-fold CV RMSE | CV RMSE 從 k=1 的 0.0501 一路降到 k=20 的 0.0200，**最佳 k = 20** |
| Embedded | Lasso 路徑（座標下降，標準化尺度） | 20 個係數幾乎同時進入，大小相近 |
| Backward | 逐一刪除 p > 0.05 的變數 | 20 個全部 p ≈ 0，**全部保留** |

**結論**：四種方法都指出 20 個特徵缺一不可。原因是每個特徵單獨只能解釋約 3–4% 的變異，但它們彼此獨立、權重相近，所以每少一個特徵，誤差都會明顯上升。

圖：`figures/04_forward_selection.png`、`05_lasso_path.png`

### 4.2 模型比較（測試集）

| 模型 | 特徵數 | Test RMSE | Test R² |
|---|---|---|---|
| Simple LR（單一最佳原始特徵 DamsQuality） | 1 | 0.05041 | 0.0354 |
| **Simple LR（FeatureSum）** | 1 | 0.02006 | 0.8472 |
| MLR Top-5 | 5 | 0.04616 | 0.1912 |
| MLR Top-10 | 10 | 0.04065 | 0.3730 |
| MLR Top-15 | 15 | 0.03252 | 0.5986 |
| **MLR 全部 20 個（最終模型）** | 20 | **0.02006** | **0.8472** |

最終模型：

FloodProbability ≈ −0.0549 + Σ βᵢ · xᵢ，其中 20 個 βᵢ 都落在 0.0055–0.0058 之間。

也就是說，任何一個風險因子每增加 1 分，洪水機率約增加 0.56 個百分點。由於係數幾乎相等，用一個 `FeatureSum` 的簡單線性回歸就能得到相同的表現，這是本作業最有意思的發現。

## 5. Evaluation（評估）

### 5.1 指標

| 指標 | 訓練集 | 測試集 |
|---|---|---|
| R² | 0.8468 | 0.8472 |
| Adjusted R² | 0.8468 | 0.8469 |
| RMSE | — | 0.02006 |
| MAE | — | 0.01573 |
| MSE | — | 0.000402 |
| F 統計量 | 11,052（p ≈ 0） | — |

訓練集與測試集的 R² 幾乎一樣，表示**沒有過擬合**。

### 5.2 預測圖（含信賴區間與預測區間）

- `figures/06_prediction_intervals.png`：隨機 300 筆測試資料，依預測值排序，畫出 95% CI（深色）與 95% PI（淺色）。
- `figures/07_actual_vs_pred.png`：實際值 vs 預測值。
- `figures/08_partial_effect_CI_PI.png`：單一特徵的邊際效果，含 CI 與 PI。

**95% PI 在測試集的實際覆蓋率為 94.2%**，接近名目的 95%，表示預測區間可信。CI 很窄（樣本數大，平均反應估得很準）；PI 較寬（約 ±0.04），反映單筆資料本身的雜訊。

### 5.3 回歸假設檢查

| 假設 | 檢定 | 結果 | 判定 |
|---|---|---|---|
| 線性 | 殘差 vs 擬合值 | 殘差平均約為 0，沒有曲線趨勢 | 大致成立 |
| 獨立性 | Durbin–Watson | 2.009 | 成立 |
| 同質變異 | Breusch–Pagan | LM = 1610，p ≈ 0 | 輕微違反（兩端變異較大） |
| 常態性 | Jarque–Bera | skew = 0.71，kurtosis = 0.95 | 輕微違反（右偏） |
| 無多重共線性 | VIF | 最大 1.004 | 成立 |

由於樣本數很大（n = 40,000），Breusch–Pagan 和 Jarque–Bera 對很小的偏離也會顯著。依中央極限定理，係數推論仍然穩健；如果要更嚴謹，可以改用 HC3 穩健標準誤。

圖：`figures/09_residual_diagnostics.png`、`10_coefficients.png`

## 6. Deployment（部署）

- **網頁**：<https://arthurliu424.github.io/webHW2/>（GitHub Pages），包含完整的 CRISP-DM 分析、互動圖表，以及一個**互動預測器**：調整 20 個滑桿後，即時算出預測值、95% CI 與 95% PI。
- **模型檔**：`outputs/model.json` 存有 β、(XᵀX)⁻¹、σ²、t 臨界值，任何語言都能載入。預測公式：
  - ŷ = x₀ᵀβ
  - CI = ŷ ± t · √(s² · x₀ᵀ(XᵀX)⁻¹x₀)
  - PI = ŷ ± t · √(s² · (1 + x₀ᵀ(XᵀX)⁻¹x₀))
- **可重現性**：`python analysis.py` 會重新產生所有圖、表與網頁資料。

### 限制與未來改進

1. 資料為合成資料，因子評分的實際意義有限，不宜直接用於真實決策。
2. 殘差有輕微異質變異。可以改用加權最小平方法（WLS）或穩健標準誤。
3. Kaggle 排行榜前段的解法加入列統計量（平均、標準差、分位數等）並搭配梯度提升樹，R² 約可達 0.869；線性回歸的上限約為 0.847。

---

## GPT 輔助內容

> 以下內容由 ChatGPT 對話產生。完整對話 PDF：`ai_logs/chatgpt_dataset_selection.pdf`

- 提問：「list kaggle multiple linear regression problems, choose features > 13, < 40」
- GPT 推薦：Predict the Flood（20 特徵）、Flood Prediction Dataset（20）、Istanbul Apartment Prices（≤ 29）、Swiss Housing（30）、Human–AI Interaction（20），以及 House Prices（79 個特徵，需要特徵選擇）。
- GPT 建議首選 Flood 資料集，因為 20 個數值特徵直接符合條件，也不需要處理類別編碼。
- GPT 建議的流程：CRISP-DM 六階段；特徵選擇比較 SelectKBest、RFE、SFS、Lasso；評估 MAE、MSE、RMSE、R²、Adj R²，並檢查回歸假設。

程式碼、圖表與網頁由 AI 程式助理（Claude Code）協助撰寫。所有數字都由 `analysis.py` 實際執行得出，並經人工檢查。

## NotebookLM 摘要

> 【請以 NotebookLM 實際產生的摘要取代本段。操作方式見 `ai_logs/README.md`】

（初稿，供比對）在 Kaggle「Regression with a Flood Prediction Dataset」（Playground S4E5）中，評分指標為 R²。多數公開解法發現，20 個風險因子與目標之間近似等權重的線性關係，因此單純的線性回歸就能達到約 0.84–0.85 的 R²。進一步的改進多半來自特徵工程：加入 20 個特徵的總和、平均、標準差、最大最小值與排序後分位數等列統計量，再搭配 LightGBM、XGBoost、CatBoost 或神經網路做集成，可將 R² 提升到約 0.869。也有討論指出，原始 flood.csv 幾乎沒有雜訊，而競賽資料是由生成模型合成，因此帶有額外雜訊。本作業的結果與這些觀察一致：只用 FeatureSum 的簡單線性回歸，R² 就與 20 個特徵的多元回歸相同。
