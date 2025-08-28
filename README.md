# 📈 XLK Next-Day Nowcast (Learning Project)

Predict next-day returns for the Technology ETF **XLK** using simple features:
- Market benchmark (**SPY**)  
- Volatility index (**VIX**)  
- Momentum indicators  
- Rolling volatility  


## 🛠️ Tech Stack
- **Core:** Python, pandas, NumPy  
- **ML:** scikit-learn (Ridge, Lasso), statsmodels  
- **Data:** yfinance (financial data APIs)  
- **Viz:** matplotlib, seaborn, Streamlit  
- **Dev Env:** Jupyter Notebook, VS Code  

---

## 📓 Notebooks
- `notebooks/01_xlk_nowcast.ipynb` – full workflow with data prep, feature engineering, model training, and evaluation.

---

##  Interactive EDA Dashboard
The project also includes a **Streamlit dashboard** (`dashboard/EDA.py`) to explore:
- XLK vs SPY overlay with correlations
- Rolling volatility & momentum indicators
- Model performance (Ridge, Lasso vs baselines)

### Run it locally
```bash
# 1. Clone the repo
git clone https://github.com/danmayster123/stock-nowcast.git
cd stock-nowcast

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4a. Run the Jupyter notebook
jupyter notebook notebooks/01_xlk_nowcast.ipynb

# 4b. Launch the Streamlit dashboard
python -m streamlit run dashboard/EDA.py
