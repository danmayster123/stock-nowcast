# XLK Next-Day Nowcast (Learning Project)
Predict next-day returns for the Technology ETF **XLK** using simple features:
market(SPY), volatility (VIX), momentum, and rolling volatility

## Tech Stack
Python, pandas, NumPy, scikit-learn, statsmodels, yfinance, Jupyter, VS Code

## How to Run
```bash
python -m venv .venv
# Windows Powershell
.\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_xlk_nowcast.ipynb