# CrisisForge LinkedIn Post

## Recommended English version

**Can a complex generative financial-risk system beat strong traditional baselines?**

I built CrisisForge, a research system that combines:

- regime-switching latent factors;
- one-shot temporal diffusion;
- stochastic factor-to-asset mapping;
- asset-level VaR, Expected Shortfall, and co-crash risk;
- CVaR and Wasserstein-robust portfolio decisions; and
- a structural counterfactual stress-testing extension.

The most interesting result was not a victory for AI.

Across 74 non-overlapping validation origins, simple time-dependence-preserving
baselines beat my switching-factor model on energy and variogram scores. The joint
VaR–ES result was statistically inconclusive. Historical CVaR also delivered lower
realized Expected Shortfall, lower drawdown, and far less turnover than portfolios
built from the more complex scenario generator.

The one-shot diffusion model ran end to end, but it remains a four-origin
engineering pilot—not evidence of superiority. A known-SCM counterfactual engine
recovered oracle paths almost exactly, then deteriorated sharply when I deliberately
misspecified the financial transmission structure.

My biggest takeaway:

> A model can be more sophisticated, generate plausible scenarios, and still make
> worse risk decisions.

The data pipeline constructs and quality-checks the complete panel, but every model
stage excludes post-2019 rows from estimation, tuning, scoring, and decisions. I
retained the negative results and saved the full Python code, configurations,
tests, data lineage, experiment receipts, report, and figures.

For me, the project became less about asking “Can AI simulate the next crisis?”
and more about asking:

**What evidence would make us trust a generated crisis scenario enough to act on
it?**

Code, report, tests, and research log:
<https://github.com/davie0624/CrisisForge>

Independent research project; not peer-reviewed and not investment advice.

What would you require before using generated stress scenarios in a real risk
decision?

#QuantFinance #RiskManagement #GenerativeAI #FinancialEngineering #Research

## 中文版本

**生成式 AI 真的能打敗傳統金融風險模型嗎？**

我建立了一套名為 CrisisForge 的研究系統，整合市場狀態轉換、潛在因子、
one-shot temporal diffusion、factor-to-asset mapping、VaR／Expected Shortfall、
CVaR／Wasserstein DRO，以及結構式反事實壓力測試。

但最有價值的結果不是「AI 全面獲勝」。

在 74 個不重疊的驗證期間中，保留時間依賴的傳統基準在 energy score 與
variogram score 上優於 switching-factor 模型；joint VaR–ES 的差異則沒有
可靠證據。使用歷史情境做 CVaR 的投資組合，也比複雜情境生成器得到更低的
Expected Shortfall、更小的最大回撤與更低的換手率。

Diffusion pipeline 確實已完整跑通，但目前只有四個 reporting origins，
因此我把它誠實標示為 engineering pilot，而不是勝負結論。反事實模組在
已知結構方程下幾乎完美復原，但只要故意改錯傳導結構，結果就明顯惡化。

這份研究帶給我最大的結論是：

> 模型可以更複雜、情境可以更逼真，但最終的風險決策仍可能更差。

資料管線會建立並檢查完整面板，但所有模型階段都把 2020 年之後的資料排除
於估計、調參、評分與決策之外；我也保留了負面結果，以及完整 Python 程式、
設定、測試、資料 lineage、實驗紀錄、研究報告與圖表。

現在我更想回答的問題不是「AI 能不能模擬下一次金融危機」，而是：

**我們需要什麼證據，才應該相信一個生成式危機情境並據此做決策？**

完整程式、報告、測試與研究紀錄：
<https://github.com/davie0624/CrisisForge>

這是一份獨立研究專案，尚未經過同儕審查，也不構成投資建議。

如果要把生成式壓力情境用在真實風險決策中，你會要求哪些證據？

#QuantFinance #RiskManagement #GenerativeAI #FinancialEngineering #Research
