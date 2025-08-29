# -*- coding: utf-8 -*-
# dashboard/EDA.py

import streamlit as st
import pandas as pd
import numpy as np
import os
os.environ["YF_USE_CURL"] = "False"   # force yfinance to use 'requests' backend
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import scipy.stats
import pathlib
import datetime as _dt

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def next_bday(d):
    """
    Return the next business day after date d.
    Uses pandas BDay (business day offset).
    """
    return (pd.to_datetime(d) + pd.tseries.offsets.BDay(1)).date()

def _model_contribs_linear(model, x_row):
    """
    For linear models (Ridge/Lasso/LinearRegression):
    compute feature contributions = coef * feature_value.
    Returns a Series sorted by contribution size.
    """
    try:
        coefs = model.coef_.ravel()
        contribs = pd.Series(coefs * x_row.values, index=x_row.index)
        return contribs
    except Exception:
        return None

def pct(x):
    """Format as percentage."""
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "n/a"

def basis_points(x):
    """Format as basis points."""
    try:
        return f"{x*10000:.0f} bp"
    except Exception:
        return "n/a"

def safe_float(x, default=np.nan):
    """Safely convert to float with fallback."""
    try:
        return float(x)
    except Exception:
        return default

def normalize_100(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize series to start at 100."""
    return df / df.iloc[0] * 100.0

def corr_phrase(r):
    """Convert correlation to descriptive phrase."""
    if pd.isna(r): return "no clear relationship"
    if r > 0.8: return "moved almost in lockstep"
    if r > 0.6: return "were closely aligned"
    if r > 0.3: return "had a moderate positive relationship"
    if r > -0.3: return "were weakly related"
    if r > -0.6: return "tended to move in opposite directions"
    return "moved strongly in opposite directions"

def max_drawdown(cum_curve: pd.Series) -> float:
    """Calculate maximum drawdown from cumulative curve."""
    roll_max = cum_curve.cummax()
    dd = (cum_curve / roll_max) - 1.0
    return dd.min()  # negative number

def evaluate(y_true, y_pred):
    """Evaluate predictions with MAE, RMSE, and directional accuracy."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    da = float((np.sign(y_true) == np.sign(y_pred)).mean())
    return mae, rmse, da

def test_directional_accuracy_significance(y_true, y_pred, alpha=0.05):
    """
    Test if directional accuracy is significantly better than random (50%) using binomial test.
    
    Returns:
    - directional_accuracy: proportion of correct predictions
    - p_value: binomial test p-value
    - is_significant: whether result is significant at alpha level
    - confidence_interval: 95% confidence interval for directional accuracy
    """
    correct_predictions = (np.sign(y_true) == np.sign(y_pred)).astype(int)
    n_correct = correct_predictions.sum()
    n_total = len(correct_predictions)
    
    directional_accuracy = n_correct / n_total
    
    # Binomial test: H0: p = 0.5 (random), H1: p > 0.5
    p_value = scipy.stats.binomtest(n_correct, n_total, 0.5, alternative='greater').pvalue
    
    # 95% confidence interval using Clopper-Pearson (exact) method
    ci_lower, ci_upper = scipy.stats.beta.interval(0.95, n_correct + 1, n_total - n_correct + 1)
    
    is_significant = p_value < alpha
    
    return {
        'directional_accuracy': directional_accuracy,
        'n_correct': n_correct,
        'n_total': n_total,
        'p_value': p_value,
        'is_significant': is_significant,
        'confidence_interval': (ci_lower, ci_upper),
        'alpha': alpha
    }

def make_features(px: pd.DataFrame):
    """Create features from price data, matching notebook structure."""
    ret = px.pct_change().dropna()
    feat = pd.DataFrame(index=ret.index)
    feat["ret_xlk"] = ret["XLK"]
    feat["ret_spy"] = ret["SPY"]
    feat["ret_vix"] = ret["VIX"]
    feat["mom_5"]  = ret["XLK"].rolling(5).mean()
    feat["mom_10"] = ret["XLK"].rolling(10).mean()
    feat["vol_10"] = ret["XLK"].rolling(10).std()
    feat["vol_20"] = ret["XLK"].rolling(20).std()
    feat = feat.dropna()

    # Shift features by 1 day: use info available at t-1 to predict t
    X = feat[["ret_spy","ret_vix","mom_5","mom_10","vol_10","vol_20"]].shift(1).dropna()
    y = feat.loc[X.index, "ret_xlk"]
    return X, y, feat

@st.cache_data(ttl=900, show_spinner=False)
def load_prices_robust(start="2015-01-01", end=None, days_back=None) -> tuple[pd.DataFrame, list]:
    """
    Unified robust data loading function.
    Can work with either start/end dates OR days_back from today.
    Returns: (df, missing_tickers)
    """
    tickers = ["XLK", "SPY", "^VIX"]
    
    # Determine date range
    if days_back is not None:
        end_date = _dt.datetime.today()
        start_date = end_date - _dt.timedelta(days=int(days_back * 1.5))
    else:
        start_date = pd.to_datetime(start) if isinstance(start, str) else start
        end_date = pd.to_datetime(end) if end else _dt.datetime.today()
    
    def normalize_columns(df):
        """Normalize multi-index columns to single level and rename VIX."""
        if isinstance(df.columns, pd.MultiIndex):
            if "Close" in df.columns.get_level_values(0):
                df = df["Close"].copy()
            else:
                df = df.copy()
        return df.rename(columns={"^VIX": "VIX"})
    
    # Try different download strategies
    strategies = [
        # Strategy 1: Multi-ticker with period
        lambda: yf.download(tickers, period="10y", auto_adjust=True, 
                           progress=False, threads=False, interval="1d"),
        # Strategy 2: Multi-ticker with date range  
        lambda: yf.download(tickers, start=start_date, end=end_date, auto_adjust=True,
                           progress=False, threads=False, interval="1d")
    ]
    
    df = pd.DataFrame()
    for strategy in strategies:
        try:
            result = strategy()
            if not result.empty:
                df = normalize_columns(result)
                break
        except Exception:
            continue
    
    # Strategy 3: Individual ticker downloads if multi-ticker failed
    missing = []
    if df.empty:
        pieces = []
        for ticker in tickers:
            try:
                data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True,
                                 progress=False, threads=False, interval="1d")
                if not data.empty and "Close" in data.columns:
                    name = "VIX" if ticker == "^VIX" else ticker
                    series = data["Close"].rename(name)
                    series.index = pd.to_datetime(series.index)
                    pieces.append(series)
                else:
                    missing.append("VIX" if ticker == "^VIX" else ticker)
            except Exception:
                missing.append("VIX" if ticker == "^VIX" else ticker)
        
        if pieces:
            df = pd.concat(pieces, axis=1, join="outer").sort_index()
            df = df.dropna(how="all")
        else:
            missing = ["XLK", "SPY", "VIX"]
    
    # Final fallback: cached CSV
    if df.empty:
        cache_path = pathlib.Path("data/prices_xlk_spy_vix.csv")
        if cache_path.exists():
            try:
                cached = pd.read_csv(cache_path, parse_dates=["Date"]).set_index("Date")
                available_cols = [c for c in ["XLK", "SPY", "VIX"] if c in cached.columns]
                if available_cols:
                    df = cached[available_cols].copy()
                    missing = [c for c in ["XLK", "SPY", "VIX"] if c not in df.columns]
            except Exception:
                pass
    
    # Apply date filtering if days_back specified
    if days_back is not None and not df.empty:
        cutoff = pd.to_datetime(end_date) - pd.Timedelta(days=days_back)
        df = df.loc[df.index >= cutoff]
    
    return df, missing

# ---------------- Page setup ----------------
st.set_page_config(page_title="XLK Nowcast – Interactive EDA & Model", layout="wide")
st.title("📊 XLK Nowcast – Interactive EDA & Model")
st.caption(f"File: {pathlib.Path(__file__).resolve()}  |  Launched: {_dt.datetime.now():%Y-%m-%d %H:%M:%S}")

# Helper functions moved to utility section at top of file

# Duplicate functions removed - see utility section at top

# Legacy function maintained for compatibility
@st.cache_data(ttl=3600, show_spinner=False) 
def load_prices(start="2015-01-01", end=None) -> pd.DataFrame:
    """Simple load function - calls the robust version and returns just the dataframe."""
    df, missing = load_prices_robust(start=start, end=end)
    if df.empty:
        raise Exception(f"Could not load price data. Missing tickers: {missing}")
    return df.dropna()

# Functions moved to utility section above

# ---------------- Load data (with fallback) ----------------
with st.spinner("Downloading data from Yahoo Finance…"):
    try:
        px = load_prices()
    except Exception as e:
        st.error(f"Could not load Yahoo data: {e}\nShowing synthetic demo data so the app remains usable.")
        idx = pd.date_range("2023-01-01", periods=350, freq="B")
        rng = np.random.default_rng(7)
        px = pd.DataFrame({
            "XLK": 100*(1+rng.normal(0,0.010,len(idx))).cumprod(),
            "SPY": 100*(1+rng.normal(0,0.008,len(idx))).cumprod(),
            "VIX": 20 + rng.normal(0,0.5,len(idx)).cumsum()
        }, index=idx)

# ============================================================================
# PAGE SETUP AND SIDEBAR CONTROLS  
# ============================================================================
st.sidebar.header("Controls")
start_date = st.sidebar.date_input("Start date", px.index.min().date())
end_date   = st.sidebar.date_input("End date",   px.index.max().date())
overlay_spy = st.sidebar.checkbox("Overlay SPY (normalized to 100)", True)
show_vix_axis = st.sidebar.checkbox("Show VIX on right axis", True)
show_feature_preview = st.sidebar.checkbox("Show feature preview table", False)

subset = px.loc[str(start_date):str(end_date)].copy()
if subset.empty:
    st.warning("Selected date range has no data. Adjust the dates in the sidebar.")
    st.stop()
show_debug = st.sidebar.checkbox("Show debug info", value=False)

# ---------------- Section: Overlay (XLK vs SPY; VIX optional) ----------------
st.subheader("XLK vs SPY (Normalized to 100)")

left, right = st.columns([2, 1], gap="large")

with left:
    fig, ax = plt.subplots(figsize=(9,4))
    norm = normalize_100(subset[["XLK"]])
    ax.plot(norm.index, norm["XLK"], label="XLK (norm=100)")
    if overlay_spy:
        norm_spy = normalize_100(subset[["SPY"]])
        ax.plot(norm_spy.index, norm_spy["SPY"], label="SPY (norm=100)", alpha=0.85)
    ax.set_ylabel("Index (base=100)")
    ax.set_title("Normalized Prices")
    ax.legend(loc="best")
    if show_vix_axis:
        ax2 = ax.twinx()
        ax2.plot(subset.index, subset["VIX"], alpha=0.35, label="VIX", color="tab:gray")
        ax2.set_ylabel("VIX")
    st.pyplot(fig, clear_figure=True)

with right:
    stats = pd.DataFrame({
        "Mean Return (daily)": subset.pct_change().mean(),
        "Volatility (daily std)": subset.pct_change().std(),
        "Min Price": subset.min(),
        "Max Price": subset.max()
    })
    st.markdown("**Quick Stats (selected window)**")
    st.dataframe(stats.style.format("{:.6f}"))

# Live interpretation for overlay
# Calculate key metrics for interpretation
xlk_cum_return = safe_float(subset["XLK"].iloc[-1]/subset["XLK"].iloc[0] - 1)
spy_cum_return = safe_float(subset["SPY"].iloc[-1]/subset["SPY"].iloc[0] - 1)

xlk_returns = subset["XLK"].pct_change().dropna()
spy_returns = subset["SPY"].pct_change().dropna()
vix_returns = subset["VIX"].pct_change().dropna()

corr_xlk_spy = safe_float(xlk_returns.corr(spy_returns))
corr_xlk_vix = safe_float(xlk_returns.corr(vix_returns))

vol_xlk = safe_float(xlk_returns.std())
vol_spy = safe_float(spy_returns.std())
vol_ratio = vol_xlk / vol_spy if vol_spy not in [0, np.nan] else np.nan

st.markdown(
    f"""
**Findings:** Over this window, **XLK {('outperformed' if xlk_cum_return > spy_cum_return else 'underperformed')} SPY**: XLK {pct(xlk_cum_return)} vs SPY {pct(spy_cum_return)}.  
XLK and SPY {corr_phrase(corr_xlk_spy)} (corr = {corr_xlk_spy:.2f}). The relationship with VIX was {corr_phrase(corr_xlk_vix)} (corr = {corr_xlk_vix:.2f}), 
consistent with **volatility spikes** lining up with **tech weakness**.  
Daily volatility: XLK {basis_points(vol_xlk)} vs SPY {basis_points(vol_spy)} ({'~' + f'{vol_ratio:.2f}×' if pd.notna(vol_ratio) else 'n/a'} the market's volatility).
"""
)

# ============================================================================
# SECTION: DISTRIBUTION OF RETURNS
# ============================================================================
st.subheader("Distribution of Daily Returns (XLK)")
rets = xlk_returns.copy()
fig, ax = plt.subplots()
sns.histplot(rets, bins=50, kde=True, ax=ax)
ax.set_xlabel("Daily Return")
ax.set_ylabel("Frequency")
st.pyplot(fig, clear_figure=True)

# Calculate distribution statistics
p05_return = safe_float(rets.quantile(0.05))
p95_return = safe_float(rets.quantile(0.95))
mean_return = safe_float(rets.mean())
median_return = safe_float(rets.median())
skewness = safe_float(rets.skew())
kurtosis = safe_float(rets.kurt())
tail_down_days = (rets < p05_return).sum()
tail_up_days = (rets > p95_return).sum()

st.markdown(
    f"""
**Findings:** Typical daily moves fell between **{pct(p05_return)}** and **{pct(p95_return)}**.  
Average daily return was **{pct(mean_return)}** (median {pct(median_return)}), with **skew = {skewness:.2f}** and **kurtosis = {kurtosis:.2f}**.  
We observed **{tail_down_days}** downside tail days (< p5) and **{tail_up_days}** upside tail days (> p95), underscoring the **fat-tailed** nature of returns.
"""
)

# ---------------- Section: Rolling stats ----------------
st.subheader("Rolling Volatility & Momentum (XLK)")
roll_left, roll_right = st.columns(2)

with roll_left:
    fig, ax = plt.subplots(figsize=(7,3.5))
    vol20 = rets.rolling(20).std()
    vol20.plot(ax=ax)
    ax.set_title("20-Day Rolling Volatility")
    ax.set_ylabel("Std Dev (daily)")
    st.pyplot(fig, clear_figure=True)

with roll_right:
    fig, ax = plt.subplots(figsize=(7,3.5))
    mom10 = rets.rolling(10).mean()
    mom10.plot(ax=ax)
    ax.set_title("10-Day Rolling Momentum")
    ax.set_ylabel("Mean Return (daily)")
    st.pyplot(fig, clear_figure=True)

# Live interpretation for rolling stats
last_vol20 = safe_float(vol20.iloc[-1])
last_mom10 = safe_float(mom10.iloc[-1])
vol_median = safe_float(vol20.median())
regime = "elevated risk" if last_vol20 > vol_median*1.25 else ("subdued risk" if last_vol20 < vol_median*0.75 else "typical risk")

st.markdown(
    f"""
**Findings:** Recent 20-day volatility is **{basis_points(last_vol20)}** per day, which implies **{regime}** vs this window’s median.  
10-day momentum is **{basis_points(last_mom10)}** per day, indicating **{'short-term strength' if last_mom10>0 else 'short-term softness'}** lately.
"""
)

# ---------------- Section: Correlation heatmap ----------------
st.subheader("Correlation Heatmap (Daily Returns)")
corr = subset.pct_change().dropna().corr()
fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
st.pyplot(fig, clear_figure=True)

# Live interpretation for correlations
c_xlk_spy = safe_float(corr.loc["XLK","SPY"]) if "XLK" in corr.index and "SPY" in corr.columns else np.nan
c_xlk_vix = safe_float(corr.loc["XLK","VIX"]) if "XLK" in corr.index and "VIX" in corr.columns else np.nan
st.markdown(
    f"""
**Findings:** In this window, **XLK–SPY corr = {c_xlk_spy:.2f}** (tech {corr_phrase(c_xlk_spy)} with the market).  
**XLK–VIX corr = {c_xlk_vix:.2f}**, reinforcing the pattern that **higher volatility** coincides with **weaker tech returns**.
"""
)

# ---------------- Section: Modeling ----------------
st.header("Modeling: Ridge/Lasso on Shifted Features")

X, y, feat_all = make_features(px)
Xw = X.loc[str(start_date):str(end_date)]
yw = y.loc[Xw.index]

# If the selected window is too short, fall back to full horizon
if len(Xw) < 120:
    X_train, X_test = X.iloc[:int(0.8*len(X))], X.iloc[int(0.8*len(X)):]
    y_train, y_test = y.iloc[:int(0.8*len(y))], y.iloc[int(0.8*len(y)):]
else:
    split = int(0.8 * len(Xw))
    X_train, X_test = Xw.iloc[:split], Xw.iloc[split:]
    y_train, y_test = yw.iloc[:split], yw.iloc[split:]

if show_feature_preview:
    st.markdown("**Feature preview (after 1-day shift; no look-ahead):**")
    st.dataframe(X_train.tail(10).style.format("{:.6f}"))

# Model controls
col1, col2 = st.columns(2)
with col1:
    model_type = st.radio("Model", ["Ridge", "Lasso"], horizontal=True)
with col2:
    default_alpha = 1.0 if model_type == "Ridge" else 0.0005
    alpha = st.slider("Alpha (regularization strength)", 1e-5, 2.0, default_alpha, step=0.0005)

# Train model
if model_type == "Ridge":
    model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=alpha))])
else:
    model = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=alpha, max_iter=10000))])

model.fit(X_train, y_train)
pred = model.predict(X_test)

# Baselines & metrics
baseline_zero = np.zeros_like(y_test)
baseline_lag1 = y_test.shift(1).fillna(0.0).values
res = pd.DataFrame({
    "Baseline Zero": evaluate(y_test, baseline_zero),
    "Baseline Lag-1": evaluate(y_test, baseline_lag1),
    f"{model_type}": evaluate(y_test, pred),
}, index=["MAE","RMSE","Directional Accuracy"]).T

st.subheader("Performance (Test Set)")
st.dataframe(res.style.format({"MAE":"{:.6f}","RMSE":"{:.6f}","Directional Accuracy":"{:.3f}"}))

# Statistical significance testing for directional accuracy
sig_test = test_directional_accuracy_significance(y_test, pred)
st.subheader("Statistical Significance Testing")

col_sig1, col_sig2 = st.columns(2)
with col_sig1:
    st.metric(
        "Directional Accuracy", 
        f"{sig_test['directional_accuracy']:.3f}",
        f"{'Significant' if sig_test['is_significant'] else 'Not significant'}"
    )
    st.metric(
        "Binomial Test p-value",
        f"{sig_test['p_value']:.4f}",
        f"{'< 0.05' if sig_test['p_value'] < 0.05 else '≥ 0.05'}"
    )

with col_sig2:
    st.metric(
        "95% Confidence Interval",
        f"[{sig_test['confidence_interval'][0]:.3f}, {sig_test['confidence_interval'][1]:.3f}]",
        f"{sig_test['n_correct']}/{sig_test['n_total']} correct"
    )

# Cross-validation analysis
st.subheader("Cross-Validation Analysis")
show_cv = st.checkbox("Show cross-validation results", value=False)

if show_cv:
    def cross_validate_timeseries(X, y, model_class, alpha, cv_splits=5):
        """Perform time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        cv_results = {'mae_scores': [], 'rmse_scores': [], 'da_scores': []}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model on fold
            if model_type == "Ridge":
                model_fold = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=alpha))])
            else:
                model_fold = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=alpha, max_iter=10000))])
            
            model_fold.fit(X_train_fold, y_train_fold)
            y_pred_fold = model_fold.predict(X_val_fold)
            
            # Evaluate
            mae, rmse, da = evaluate(y_val_fold, y_pred_fold)
            cv_results['mae_scores'].append(mae)
            cv_results['rmse_scores'].append(rmse)
            cv_results['da_scores'].append(da)
        
        return cv_results
    
    with st.spinner("Running cross-validation..."):
        cv_results = cross_validate_timeseries(X, y, model_type, alpha, cv_splits=5)
        
        # Calculate summary statistics
        mae_scores = np.array(cv_results['mae_scores'])
        rmse_scores = np.array(cv_results['rmse_scores'])
        da_scores = np.array(cv_results['da_scores'])
        
        cv_summary = pd.DataFrame({
            'Mean': [mae_scores.mean(), rmse_scores.mean(), da_scores.mean()],
            'Std': [mae_scores.std(), rmse_scores.std(), da_scores.std()],
            'Min': [mae_scores.min(), rmse_scores.min(), da_scores.min()],
            'Max': [mae_scores.max(), rmse_scores.max(), da_scores.max()]
        }, index=['MAE', 'RMSE', 'Directional Accuracy'])
        
        st.dataframe(cv_summary.style.format("{:.6f}"))
        
        # Test if DA is significantly > 0.5 across folds
        t_stat, p_value = scipy.stats.ttest_1samp(da_scores, 0.5)
        
        st.markdown(f"""
        **Cross-Validation Results:** 
        - Average Directional Accuracy: **{da_scores.mean():.3f} ± {da_scores.std():.3f}**
        - DA vs 50% (t-test): **t={t_stat:.3f}, p={p_value:.4f}**
        - {'✅ Consistently better than random' if p_value < 0.05 else '❌ Not consistently better than random'}
        """)

# Live interpretation for metrics
mae_base = safe_float(res.loc["Baseline Zero","MAE"])
mae_model = safe_float(res.loc[model_type,"MAE"])
rmse_base = safe_float(res.loc["Baseline Zero","RMSE"])
rmse_model = safe_float(res.loc[model_type,"RMSE"])
da_base = safe_float(res.loc["Baseline Lag-1","Directional Accuracy"])
da_model = safe_float(res.loc[model_type,"Directional Accuracy"])

imp_mae = (mae_base - mae_model)/mae_base if (pd.notna(mae_base) and mae_base>0) else np.nan
imp_rmse = (rmse_base - rmse_model)/rmse_base if (pd.notna(rmse_base) and rmse_base>0) else np.nan
delta_da = da_model - da_base if (pd.notna(da_model) and pd.notna(da_base)) else np.nan

# Coefficient magnitudes on standardized features (to rank signal strength)
coef_series = None
try:
    # Extract the linear model and its coef on scaled features
    coefs = model.named_steps["model"].coef_
    coef_series = pd.Series(np.abs(coefs), index=X.columns).sort_values(ascending=False)
    top_feats = ", ".join([f"{k}" for k in coef_series.head(3).index])
except Exception:
    top_feats = "N/A"

st.markdown(
    f"""
**Model Performance Summary:** The **{model_type}** reduced error vs the zero baseline by **MAE {pct(imp_mae)}** and **RMSE {pct(imp_rmse)}**.  
Directional Accuracy reached **{pct(da_model)}** ({'statistically significant' if sig_test['is_significant'] else 'not statistically significant'}), 
a **{pct(delta_da)}** change vs a simple lag-1 rule.  
Top signals by model weight (standardized): **{top_feats}** — indicating these carried the strongest predictive influence in this window.
"""
)

# ---------------- Section: Predicted vs Actual ----------------
st.subheader("Predicted vs Actual Returns (Test)")
plot_df = pd.DataFrame({"Actual": y_test, model_type: pred}, index=y_test.index).dropna()
st.line_chart(plot_df)

# Live interpretation for pred vs actual
pred_corr = safe_float(pd.Series(pred, index=y_test.index).corr(y_test))
big_miss_threshold = 2 * safe_float(y_test.std())
big_misses = int(np.sum(np.abs(pred - y_test) > big_miss_threshold)) if pd.notna(big_miss_threshold) else 0

st.markdown(
    f"""
**Findings:** Prediction–actual correlation was **{pred_corr:.2f}**.  
Large errors (>|2×σ|) occurred **{big_misses}** time(s), which is typical for daily horizons where **noise dominates**.
"""
)

# ---------------- Section: Toy backtest (long if pred>0, else flat) ----------------
st.subheader("Toy Strategy: Long if Prediction > 0, Else Flat")
signal = (pd.Series(pred, index=y_test.index) > 0).astype(int)
strat_ret = signal * y_test
cum_strat = (1 + strat_ret).cumprod()
cum_bh = (1 + y_test).cumprod()
bt = pd.DataFrame({"Strategy": cum_strat, "Buy&Hold XLK": cum_bh})
st.line_chart(bt)

# Live interpretation for backtest
n_days = len(y_test)
ann_factor = np.sqrt(252)
ann_ret_strat = (cum_strat.iloc[-1] ** (252/max(n_days,1))) - 1 if n_days > 0 else np.nan
ann_vol_strat = ann_factor * safe_float(strat_ret.std())
sharpe_strat = (ann_ret_strat / ann_vol_strat) if (pd.notna(ann_ret_strat) and pd.notna(ann_vol_strat) and ann_vol_strat>0) else np.nan
dd_strat = max_drawdown(cum_strat)

ann_ret_bh = (cum_bh.iloc[-1] ** (252/max(n_days,1))) - 1 if n_days > 0 else np.nan
ann_vol_bh = ann_factor * safe_float(y_test.std())
sharpe_bh = (ann_ret_bh / ann_vol_bh) if (pd.notna(ann_ret_bh) and pd.notna(ann_vol_bh) and ann_vol_bh>0) else np.nan
dd_bh = max_drawdown(cum_bh)

st.markdown(
    f"""
**Findings:** Over the test window:  
- **Strategy** → CAGR {pct(ann_ret_strat)}, vol {pct(ann_vol_strat)}, Sharpe ~ {safe_float(sharpe_strat):.2f}, max drawdown {pct(dd_strat)}  
- **Buy & Hold** → CAGR {pct(ann_ret_bh)}, vol {pct(ann_vol_bh)}, Sharpe ~ {safe_float(sharpe_bh):.2f}, max drawdown {pct(dd_bh)}  

This simple **long/flat** rule can reduce exposure during weak periods (lower drawdowns) when Directional Accuracy is above **50%**, 
but buy-and-hold may lead during strong uptrends. *(Illustrative only; ignores costs/slippage.)*
"""
)
# ---------------- Section: Live Next-Day Forecast ----------------
st.header("Live Next-Day Forecast for XLK")
st.caption("Pulls recent XLK/SPY/VIX data, rebuilds features, and applies the current model to forecast *tomorrow's* XLK return.")

colA, colB = st.columns([1, 2])
with colA:
    run_live = st.button("Run Live Forecast", type="primary", use_container_width=True)

if run_live:
    with st.spinner("Fetching latest data and computing features…"):
        px_recent, missing = load_prices_robust(days_back=420)
        
        ordered = ["XLK", "SPY", "VIX"]
        px_recent = px_recent[[c for c in ordered if c in px_recent.columns]]
    
        # Must have at least XLK and SPY
        if px_recent.empty or not all(c in px_recent.columns for c in ["XLK", "SPY"]):
            miss = [c for c in ["XLK", "SPY"] if c not in px_recent.columns]
            st.error(
                "Could not load XLK/SPY data, which are required for the model."
                + (f" Missing: {', '.join(miss)}." if miss else "")
            )
            st.stop()

        # Build feature row that matches the model's expected columns:
        # ret_spy (t-1), ret_vix (t-1), mom_5, mom_10, vol_10, vol_20 (all from XLK).
        # If VIX is unavailable, set ret_vix = 0.0 (neutral) and warn.
        ret = px_recent.pct_change().dropna()
        feat = pd.DataFrame(index=ret.index)

        feat["ret_spy"] = ret["SPY"]

        if "VIX" in px_recent.columns:
            feat["ret_vix"] = ret["VIX"]
            vix_available = True
        else:
            feat["ret_vix"] = 0.0  # neutral placeholder so pipeline sees the same columns
            vix_available = False

        feat["mom_5"]  = ret["XLK"].rolling(5).mean()
        feat["mom_10"] = ret["XLK"].rolling(10).mean()
        feat["vol_10"] = ret["XLK"].rolling(10).std()
        feat["vol_20"] = ret["XLK"].rolling(20).std()
        feat = feat.dropna()

        # Shift by 1 day to avoid look-ahead
        X_live = feat[["ret_spy", "ret_vix", "mom_5", "mom_10", "vol_10", "vol_20"]].shift(1).dropna()
        if len(X_live) == 0:
            st.error("Not enough recent data to compute rolling features. Try increasing days_back.")
            st.stop()

        x_last = X_live.iloc[-1]
        last_date = X_live.index[-1]
        target_day = next_bday(last_date)

        try:
            pred_next = float(model.predict(x_last.to_frame().T)[0])
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            st.stop()

        if not vix_available:
            st.warning("VIX data unavailable. Used a neutral placeholder (ret_vix=0.0) for this forecast; interpretation may be less reliable.")

        # Big metric
        direction = "UP" if pred_next > 0 else ("DOWN" if pred_next < 0 else "FLAT")
        delta_txt = "↑" if direction == "UP" else ("↓" if direction == "DOWN" else "→")
        st.metric(
            label=f"Predicted XLK next-day return ({target_day})",
            value=f"{pred_next*100:.2f}%",
            delta=delta_txt
        )

        # Context table (include VIX only if present)
        ctx_rows = {
            "SPY return (t-1)": x_last["ret_spy"],
            "XLK 10-day momentum": x_last["mom_10"],
            "XLK 20-day volatility": x_last["vol_20"],
        }
        if vix_available:
            ctx_rows["VIX return (t-1)"] = x_last["ret_vix"]

        ctx = pd.DataFrame({"Value": list(ctx_rows.values())}, index=list(ctx_rows.keys())).round(6)

        with colB:
            st.subheader("Context")
            st.dataframe(ctx.style.format("{:.6f}"))

        # Feature contributions (linear model)
        contrib = _model_contribs_linear(model, x_last)
        if contrib is not None and len(contrib) > 0:
            st.subheader("Feature Contributions (standardized)")
            fig, ax = plt.subplots(figsize=(7, 3.6))
            contrib.iloc[:6].sort_values().plot(kind="barh", ax=ax)
            ax.set_xlabel("Contribution to predicted return")
            ax.set_ylabel("Feature")
            st.pyplot(fig, clear_figure=True)

        # Narrative interpretation
        spy_s = "positive" if x_last["ret_spy"] > 0 else ("negative" if x_last["ret_spy"] < 0 else "flat")
        mom_s = "supportive" if x_last["mom_10"] > 0 else ("weak" if x_last["mom_10"] < 0 else "neutral")
        if vix_available:
            vix_s = "falling" if x_last["ret_vix"] < 0 else ("rising" if x_last["ret_vix"] > 0 else "flat")
            vix_phrase = f", **VIX** was *{vix_s}*"
        else:
            vix_phrase = ", **VIX** unavailable (neutralized in model)"

        st.markdown(
            f"""
**Interpretation:** The model calls **{direction}** for the next trading day (**{pred_next*100:.2f}%**).  
Signals at the close: **SPY** was *{spy_s}*{vix_phrase}, 10-day **momentum** looks *{mom_s}*, and 20-day **volatility** ≈ {basis_points(float(x_last['vol_20']))} per day.  
Short-horizon forecasts have high noise; treat this as a directional tilt, not certainty.
"""
        )
else:
    st.info("Click **Run Live Forecast** to generate a next-day prediction using the current model.")

# ============================================================================
# COMPREHENSIVE METHODOLOGY EXPLANATION
# ============================================================================
st.header("Behind the Scenes: Methodology & Technical Implementation")
st.markdown("""
This section provides a comprehensive overview of the machine learning pipeline and methodology used for XLK return prediction. 
The techniques demonstrated here are widely applicable across industries beyond finance.
""")

with st.expander("**Data Engineering & Feature Construction**", expanded=False):
    st.markdown("""
    ### Data Sources & Collection
    - **Price Data**: Real-time retrieval from Yahoo Finance API with robust error handling
    - **Assets**: XLK (Technology ETF), SPY (Market Proxy), VIX (Volatility Index)
    - **Timeframe**: Daily frequency from 2015-present (~2,600+ observations)
    
    ### Feature Engineering Pipeline
    The model employs **lagged features** to avoid look-ahead bias—a critical concept in time series modeling:
    
    **Market Context Features:**
    - `ret_spy(t-1)`: Previous day's S&P 500 return (market direction signal)
    - `ret_vix(t-1)`: Previous day's VIX change (volatility regime indicator)
    
    **Technical Momentum Features:**
    - `mom_5(t-1)`: 5-day rolling average return (short-term momentum)
    - `mom_10(t-1)`: 10-day rolling average return (medium-term momentum)
    
    **Risk Management Features:**
    - `vol_10(t-1)`: 10-day rolling volatility (short-term risk)
    - `vol_20(t-1)`: 20-day rolling volatility (longer-term risk)
    
    ### Data Quality & Preprocessing
    - **Missing Data Handling**: Multiple fallback strategies with cached data
    - **Normalization**: StandardScaler ensures features contribute equally regardless of scale
    - **Temporal Integrity**: 1-day lag ensures all features are "knowable" at prediction time
    """)

with st.expander("**Machine Learning Architecture**", expanded=False):
    st.markdown("""
    ### Model Selection Rationale
    **Ridge Regression** was chosen as the primary model for several strategic reasons:
    
    - **Interpretability**: Linear coefficients provide clear feature importance rankings
    - **Regularization**: L2 penalty prevents overfitting in noisy financial data
    - **Stability**: Less sensitive to multicollinearity between market features
    - **Speed**: Fast training/prediction enables real-time dashboard updates
    
    ### Model Pipeline
    ```
    Raw Prices → Feature Engineering → Scaling → Ridge Regression → Predictions
    ```
    
    **Scaling Strategy**: StandardScaler transforms features to zero mean, unit variance:
    - Prevents features with larger scales (e.g., volatility) from dominating
    - Ensures regularization penalty applies fairly across all features
    - Critical for interpretable coefficient comparisons
    
    ### Training & Validation Framework
    - **Time Series Split**: 80/20 train-test split respecting temporal order
    - **No Data Leakage**: Strict chronological separation prevents future information contamination
    - **Multiple Baselines**: Zero prediction and lag-1 benchmarks establish minimum performance thresholds
    """)

with st.expander("**Performance Evaluation Framework**", expanded=False):
    st.markdown("""
    ### Evaluation Metrics Suite
    The model is assessed using multiple complementary metrics:
    
    **Regression Metrics:**
    - **MAE (Mean Absolute Error)**: Average prediction error magnitude
    - **RMSE (Root Mean Squared Error)**: Penalizes larger errors more heavily
    
    **Classification Metrics:**
    - **Directional Accuracy**: Percentage of correct up/down predictions
    - Critical for trading strategies where direction matters more than magnitude
    
    ### Benchmark Comparisons
    - **Zero Baseline**: Always predicting no change (market efficiency hypothesis)
    - **Lag-1 Baseline**: Using previous day's return as prediction (momentum strategy)
    - Performance improvements demonstrate model's predictive value above naive approaches
    
    ### Strategy Backtesting
    **Simple Long/Flat Strategy**: 
    - Long XLK when prediction > 0, flat otherwise
    - Compares against buy-and-hold benchmark
    - Accounts for reduced market exposure during predicted downturns
    - **Note**: Simplified for demonstration; real strategies require transaction costs, slippage, and risk management
    """)

with st.expander("**Feature Importance & Model Interpretability**", expanded=False):
    st.markdown("""
    ### Coefficient Analysis
    Ridge regression coefficients (after scaling) reveal feature importance:
    - **Positive coefficients**: Features associated with positive XLK returns
    - **Negative coefficients**: Features signaling potential downside
    - **Magnitude**: Larger absolute values indicate stronger predictive relationships
    
    ### Economic Intuition Behind Features
    - **SPY Correlation**: Tech sector typically moves with broader market
    - **VIX Relationship**: High volatility often coincides with tech sell-offs
    - **Momentum Effects**: Recent trends may persist in the short term
    - **Volatility Clustering**: High volatility periods tend to cluster together
    
    ### Contribution Analysis
    The dashboard shows individual feature contributions to each prediction:
    - Decomposes final prediction into component parts
    - Helps understand which signals drove specific forecasts
    - Enables model debugging and validation of economic relationships
    """)

with st.expander("**Real-Time Implementation & Deployment**", expanded=False):
    st.markdown("""
    ### Production Architecture
    - **Caching Strategy**: 15-minute data cache balances freshness with API efficiency
    - **Error Handling**: Multiple fallback data sources ensure system reliability
    - **Graceful Degradation**: System continues operating even with missing data (e.g., VIX unavailable)
    
    ### Scalability Considerations
    The architecture demonstrated here scales to industrial applications:
    - **Feature Pipeline**: Easily extensible to additional assets or alternative data
    - **Model Framework**: Supports ensemble methods, neural networks, or more complex algorithms
    - **Real-Time Updates**: Infrastructure supports live model retraining and deployment
    
    ### Risk Management & Limitations
    **Model Limitations:**
    - Short-term predictions inherently noisy in efficient markets
    - Historical relationships may not persist (regime changes)
    - Single-asset focus doesn't capture portfolio-level risks
    
    **Practical Considerations:**
    - Predictions should inform, not replace, human judgment
    - Transaction costs and market impact not modeled
    - Model performance varies across different market regimes
    """)

# ============================================================================
# PROJECT SUMMARY & BROADER APPLICATIONS
# ============================================================================
st.header("Project Summary & Cross-Industry Applications")

col_summary1, col_summary2 = st.columns(2)

with col_summary1:
    st.subheader("**What This Project Demonstrates**")
    st.markdown("""
    **Technical Skills:**
    - **End-to-End ML Pipeline**: From raw data to production deployment
    - **Time Series Modeling**: Proper handling of temporal dependencies and data leakage
    - **Feature Engineering**: Domain-driven variable construction and transformation
    - **Model Evaluation**: Comprehensive performance assessment with multiple metrics
    - **Interactive Visualization**: Professional dashboard development with Streamlit
    
    **Data Science Best Practices:**
    - Robust data collection with error handling and fallback mechanisms
    - Principled train/test splitting respecting temporal structure
    - Clear separation between exploratory analysis and production code
    - Comprehensive documentation and reproducible workflows
    """)

with col_summary2:
    st.subheader("**Cross-Industry Applications**")
    st.markdown("""
    **The methodologies demonstrated generalize across domains:**
    
    **Manufacturing & Operations:**
    - Demand forecasting using lagged sales, economic indicators
    - Equipment failure prediction using sensor data, maintenance history
    - Quality control using process parameters, environmental factors
    
    **Healthcare & Life Sciences:**
    - Patient outcome prediction using historical vitals, treatment responses  
    - Drug efficacy modeling using biomarkers, patient characteristics
    - Epidemiological forecasting using case data, mobility patterns
    
    **Marketing & E-commerce:**
    - Customer lifetime value prediction using transaction history, engagement metrics
    - Campaign effectiveness using historical performance, audience features
    - Inventory optimization using seasonal patterns, demand signals
    """)

st.subheader("**Technical Architecture Highlights**")

arch_col1, arch_col2, arch_col3 = st.columns(3)

with arch_col1:
    st.markdown("""
    **Data Engineering**
    - Real-time API integration
    - Robust error handling
    - Efficient caching strategies
    - Data quality validation
    """)

with arch_col2:
    st.markdown("""
    **Machine Learning**
    - Time series feature engineering
    - Regularized linear modeling  
    - Cross-validation framework
    - Model interpretability
    """)

with arch_col3:
    st.markdown("""
    **Production Deployment**
    - Interactive web dashboard
    - Real-time predictions
    - Comprehensive documentation
    - Scalable architecture
    """)

st.subheader("**Key Learning Outcomes**")
st.markdown("""
This project showcases the complete lifecycle of a data science solution, from problem formulation through production deployment. 
Key takeaways include:

- **Domain Expertise Integration**: Combining financial market knowledge with machine learning techniques
- **Temporal Data Challenges**: Properly handling time series data to avoid common pitfalls
- **Production Considerations**: Building systems that are robust, interpretable, and maintainable  
- **Stakeholder Communication**: Presenting technical work in an accessible, actionable format

The skills and methodologies demonstrated here are **directly transferable** to predictive modeling challenges across industries, 
making this project a strong foundation for data science roles in any domain requiring quantitative analysis and forecasting.
""")

st.markdown("---")
st.markdown("*Built with Python, Streamlit, scikit-learn, and modern data science best practices*")
    