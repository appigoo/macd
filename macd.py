import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# 計算 MACD
def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# 計算 RSI
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 計算 Stochastic
def calculate_stochastic(df, k_period=14, d_period=3):
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()
    return k, d

# 計算 OBV
def calculate_obv(df):
    sign = np.sign(df['Close'].diff())
    obv = (sign * df['Volume']).fillna(0).cumsum()
    return obv

# 計算 MFI
def calculate_mfi(df, period=14):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    positive_flow = raw_money_flow.where(typical_price.diff() > 0, 0).rolling(window=period).sum()
    negative_flow = raw_money_flow.where(typical_price.diff() < 0, 0).rolling(window=period).sum()
    money_ratio = positive_flow / negative_flow
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi

# 檢測多頭分歧
def detect_bullish_divergence(df, histogram):
    if len(df) < 3:
        return False
    recent_lows = pd.to_numeric(df['Low'].iloc[-3:], errors='coerce')
    hist_lows = pd.to_numeric(histogram.iloc[-3:], errors='coerce')
    diff_lows = recent_lows.diff().dropna()
    diff_hists = hist_lows.diff().dropna()
    # 確保數值比較
    lows_decreasing = all(pd.to_numeric(d, errors='coerce') <= 0 and pd.notna(pd.to_numeric(d, errors='coerce')) for d in diff_lows)
    hist_decreasing = all(pd.to_numeric(d, errors='coerce') <= 0 and pd.notna(pd.to_numeric(d, errors='coerce')) for d in diff_hists)
    # 多頭分歧判斷是價格創新低，但指標沒有創新低
    if lows_decreasing and not hist_decreasing:
        return True
    return False

# 獲取數據
def get_data(ticker, period, interval):
    try:
        # 嘗試使用 Ticker.history 以避免 download 的某些錯誤
        data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if data.empty:
            # 後備：嘗試每日數據（適合周末）
            is_weekend = datetime.now().weekday() >= 5
            if is_weekend:
                data = yf.Ticker(ticker).history(period='5d', interval='1d', auto_adjust=False)
        if data.empty:
            return pd.DataFrame()
        
        # 確保 OHLCV 為數值型
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.dropna(subset=['Close'])  # 移除無效行
        
        return data
    except Exception as e:
        st.error(f"獲取數據失敗: {e}")
        # 後備每日數據
        try:
            data = yf.Ticker(ticker).history(period='5d', interval='1d', auto_adjust=False)
            if not data.empty:
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in data.columns:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                return data
        except:
            pass
        return pd.DataFrame()

# Streamlit app 主介面
st.title('股票日內交易助手')
st.write('基於 MACD、Histogram 變化、多頭分歧、RSI、Stochastic、OBV、MFI 指標，自動更新。')

# 側邊欄輸入參數
with st.sidebar:
    st.subheader('自訂參數')
    ticker = st.text_input('股票代碼', value='TSLA')
    period = st.selectbox('數據天數', ['1d', '5d', '10d'], index=1)  # 默認 5d 以避免周末 1d 問題
    interval = st.selectbox('K線間隔', ['1m', '5m', '15m', '1d'], index=1)  # 添加 1d 選項
    refresh_minutes = st.number_input('建議刷新間隔（分鐘）', value=5, min_value=1)

    st.subheader('指標設置')
    macd_fast = st.number_input('MACD Fast Period', value=12, min_value=1)
    macd_slow = st.number_input('MACD Slow Period', value=26, min_value=1)
    macd_signal = st.number_input('MACD Signal Period', value=9, min_value=1)
    rsi_period = st.number_input('RSI Period', value=14, min_value=1)
    stoch_k = st.number_input('Stochastic K Period', value=14, min_value=1)
    stoch_d = st.number_input('Stochastic D Period', value=3, min_value=1)
    mfi_period = st.number_input('MFI Period', value=14, min_value=1)

placeholder = st.empty()

def refresh_data():
    data = get_data(ticker, period, interval)
    if data.empty:
        with placeholder:
            is_weekend = datetime.now().weekday() >= 5
            msg = '無法獲取數據，請檢查股票代碼。' if not is_weekend else '無法獲取數據，市場周末無 intraday 數據。'
            st.error(msg)
        return

    required_cols = ['Close', 'High', 'Low', 'Volume']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        with placeholder:
            st.error(f"數據缺少必要欄位: {missing_cols}，請檢查ticker或interval。")
        return

    data = data.tail(500)  # 限制數據長度

    macd_line, signal_line, histogram = calculate_macd(data, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    data['MACD'] = macd_line
    data['Signal'] = signal_line
    data['Histogram'] = histogram

    data['RSI'] = calculate_rsi(data, period=rsi_period)
    k, d = calculate_stochastic(data, k_period=stoch_k, d_period=stoch_d)
    data['%K'] = k
    data['%D'] = d
    data['OBV'] = calculate_obv(data)
    data['MFI'] = calculate_mfi(data, period=mfi_period)
    data = data.dropna()

    if len(data) < 10:
        with placeholder:
            st.warning('數據不足（<10 根 K 線），無法計算完整指標。請調整 period 或 interval。')
        return

    latest_hist = pd.to_numeric(data['Histogram'].tail(3), errors='coerce')
    diff_hist = latest_hist.diff().dropna()
    # 確保數值比較
    hist_increasing = all(pd.to_numeric(d, errors='coerce') > 0 and pd.notna(pd.to_numeric(d, errors='coerce')) for d in diff_hist) and (latest_hist.iloc[-1] < 0)
    divergence = detect_bullish_divergence(data, data['Histogram'])
    rsi_latest = data['RSI'].iloc[-1]
    rsi_signal = (rsi_latest > 40) and (data['RSI'].iloc[-2] < 30) if len(data) > 1 else False
    stoch_cross = (data['%K'].iloc[-1] > data['%D'].iloc[-1]) and (data['%K'].iloc[-2] < 20) if len(data) > 1 else False
    vol_mean = data['Volume'].rolling(10).mean().iloc[-1]
    volume_spike = (not pd.isna(vol_mean)) and (data['Volume'].iloc[-1] > vol_mean * 1.5) if len(data) > 10 else False
    obv_up = (data['OBV'].diff().iloc[-1] > 0) if len(data) > 1 else False
    mfi_signal = (data['MFI'].iloc[-1] > 20) and (data['MFI'].iloc[-2] < 20) if len(data) > 1 else False

    signals = [hist_increasing, divergence, rsi_signal, stoch_cross, volume_spike, obv_up, mfi_signal]
    score = sum(signals)

    suggestion = '無明顯買入信號。繼續監測。'
    if score >= 3:
        suggestion = '潛在買入機會：MACD Histogram 縮小，預測 MACD 可能即將從負轉正。建議關注。'
    if score >= 5:
        suggestion = '強烈買入信號：多指標確認，預測 MACD 即將交叉轉正。考慮進場，設止損。'

    with placeholder:
        st.subheader('最新數據和指標')
        st.metric("最新收盤價", f"{data['Close'].iloc[-1]:.2f}")
        st.write(f'MACD Histogram: {data["Histogram"].iloc[-1]:.4f} (是否縮小: {"是" if hist_increasing else "否"})')
        st.write(f'多頭分歧: {"檢測到" if divergence else "無"}')
        st.write(f'RSI: {rsi_latest:.2f} (信號: {"是" if rsi_signal else "否"})')
        st.write(f'Stochastic %K/%D: {data["%K"].iloc[-1]:.2f} / {data["%D"].iloc[-1]:.2f} (交叉: {"是" if stoch_cross else "否"})')
        st.write(f'OBV: {data["OBV"].iloc[-1]:,.0f} (上漲: {"是" if obv_up else "否"})')
        st.write(f'MFI: {data["MFI"].iloc[-1]:.2f} (信號: {"是" if mfi_signal else "否"})')
        st.write(f'成交量尖峰: {"是" if volume_spike else "否"}')

        st.subheader('交易建議')
        st.write(suggestion)
        st.write(f'信號強度: {score}/7')

        st.subheader('最近 10 根 K 線數據')
        st.dataframe(data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']])

        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(data['Close'].tail(50))
        with col2:
            st.line_chart(data['Histogram'].tail(50))

# 初始載入數據
refresh_data()

# 手動刷新按鈕（側邊欄參數變化時自動 reruns）
st.sidebar.markdown("---")
if st.sidebar.button('立即刷新數據'):
    st.rerun()

st.sidebar.info(f'建議每 {refresh_minutes} 分鐘手動刷新一次，以獲取最新數據。周末將自動切換至每日數據。')
