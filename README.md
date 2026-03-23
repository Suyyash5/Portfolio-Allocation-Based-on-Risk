# Portfolio Allocation Based on Risk using Sharpe Ratio

## Overview
This project focuses on **allocating a user's investment across different stocks based on their risk profile**.  
The allocation strategy is driven by **Sharpe Ratio**, which balances return and risk to identify optimal stock weightage.

The goal of this project is to provide a **simple, data-driven approach** to portfolio allocation without introducing unnecessary complexity.


## Problem Statement
Investors often struggle to decide:
- Which stocks to invest in?
- How much capital to allocate to each stock?

This project solves that by:
- Measuring **risk (volatility)** using standard deviation  
- Measuring **return** from historical stock data  
- Using **Sharpe Ratio** to determine optimal weightage  



## Dataset
- **Number of Companies:** 46  
- **Time Period:** 2011 – 2023  
- **Features Used:**
  - Open
  - High
  - Low
  - Close  



## Methodology

### 1. Return Calculation
Returns are calculated using closing prices:
Return= Pt - Pt−1 / Pt-1



### 2. Risk Calculation
Risk is measured using **standard deviation of returns**:
Risk = σ (Returns)



### 3. Sharpe Ratio
The Sharpe Ratio is used to evaluate risk-adjusted returns:
Sharpe Ratio = (Return - R_f) / σ

*(Assuming risk-free rate R_f ≈ 0 for simplicity)*



### 4. Weight Allocation
- Stocks with higher Sharpe Ratio get higher weight  
- Weights are normalized so total allocation = 100%  



## Project Workflow
1. Load stock dataset  
2. Compute returns for each company  
3. Calculate standard deviation (risk)  
4. Compute Sharpe Ratio  
5. Rank stocks based on Sharpe Ratio  
6. Assign weights accordingly  
7. Generate portfolio allocation  



## Key Features
- Risk-based portfolio allocation  
- Simple and interpretable approach  
- Uses historical data for decision-making  
- No overfitting or unnecessary ML complexity  



## Use Case
- Beginner investors looking for structured allocation  
- Educational understanding of portfolio optimization  
- Base model for advanced quantitative finance projects  



## Limitations
- Does not consider:
  - Market events or macroeconomic factors  
  - Correlation between stocks  
  - Dynamic rebalancing  

- Based only on historical data  


## Future Improvements (Optional)
- Add correlation-based portfolio optimization (Markowitz)  
- Include real-time market data  
- Incorporate user-defined risk appetite  
- Add clustering for stock grouping  



## Tech Stack
- Python  
- Pandas  
- NumPy  
- Matplotlib / Seaborn (optional for visualization)
