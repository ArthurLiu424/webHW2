# AI 對話紀錄

請把以下檔案放在這個資料夾：

1. `chatgpt_dataset_selection.pdf`：與 ChatGPT 的對話（用 pdfCrowd 或瀏覽器「列印 → 另存 PDF」匯出），內容是挑選 Kaggle 資料集與規劃 CRISP-DM 流程。
2. `notebooklm_summary.pdf` 或截圖：NotebookLM 的研究摘要。
3. （選用）`claude_code_session.pdf`：產生程式碼與網頁的 AI 協助紀錄。

## NotebookLM 建議操作

1. 開新筆記本，加入以下來源：
   - <https://www.kaggle.com/competitions/playground-series-s4e5/discussion>
   - <https://www.kaggle.com/competitions/playground-series-s4e5/code>（挑 3–5 篇高票 Notebook）
   - <https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset/code>
2. 提問：「請用繁體中文、150 字左右，摘要這些解法使用的模型、特徵工程與 R² 表現，並比較線性回歸與其他方法。」
3. 把輸出貼進 `REPORT.md` 的「NotebookLM 摘要」段落，以及 `docs/index.html` 中 `id="nlm-text"` 的段落。
