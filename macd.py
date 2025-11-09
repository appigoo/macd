import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import time

# 計算 MACD（支持自訂設置）
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

# 計算 OBV（向量化版本，提升效能）
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

# 檢測 bullish divergence（簡單版：最近 3 根價格新低但 histogram 沒新低）
def detect_bullish_divergence(df, histogram):
    recent_lows = df['Low'][-3:]
    hist_lows = histogram[-3:]
    if recent_lows.is_monotonic_decreasing and not hist_lows.is_monotonic_decreasing:
        return True
    return False

# 獲取數據（使用 yfinance）
def get_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            return pd.DataFrame()
        # 移除 rename，保持大寫欄位
        return data
    except Exception as e:
        st.error(f"獲取數據失敗: {e}")
        return pd.DataFrame()

# Streamlit app 主介面
st.title('股票日內交易助手')
st.write('基於 MACD 預測、Histogram 變化、多頭分歧、RSI、Stochastic、OBV、MFI 等指標，自動更新。')

# 自訂設置（側邊欄）
with st.sidebar:
    st.subheader('自訂參數')
    ticker = st.text_input('股票代碼', value='TSLA')
    period = st.selectbox('數據天數', ['1d', '5d', '10d'], index=0)
    interval = st.selectbox('K線間隔', ['1m', '5m', '15m'], index=1)
    refresh_minutes = st.number_input('自動刷新分鐘', value=5, min_value=1)
    
    st.subheader('自訂指標設置')
    macd_fast = st.number_input('MACD Fast Period', value=12)
    macd_slow = st.number_input('MACD Slow Period', value=26)
    macd_signal = st.number_input('MACD Signal Period', value=9)
    rsi_period = st.number_input('RSI Period', value=14)
    stoch_k = st.number_input('Stochastic K Period', value=14)
    stoch_d = st.number_input('Stochastic D Period', value=3)
    mfi_period = st.number_input('MFI Period', value=14)

placeholder = st.empty()

while True:
    try:
        data = get_data(ticker, period, interval)
        
        if not data.empty:
            # 限制數據長度以提升效能（最多 500 根 K 線）
            data = data.tail(500)
            # 移除 NaN（因 rolling/ewm 產生）
            data = data.dropna()
            
            if len(data) < 10:
                st.warning('數據不足，無法計算完整指標。請調整 period 或 interval。')
                time.sleep(refresh_minutes * 60)
                st.rerun()
                continue
            
            # 計算指標
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
            
            # 分析信號
            latest_hist = histogram.iloc[-3:]  # 最近 3 根 histogram
            hist_increasing = latest_hist.diff().dropna().gt(0).all() and latest_hist.iloc[-1] < 0  # 負值但增加
            
            divergence = detect_bullish_divergence(data, histogram)
            
            rsi_latest = data['RSI'].iloc[-1]
            rsi_signal = rsi_latest > 40 and data['RSI'].iloc[-2] < 30 if len(data) > 1 else False  # 從超賣反彈
            
            stoch_cross = data['%K'].iloc[-1] > data['%D'].iloc[-1] and data['%K'].iloc[-2] < 20 if len(data) > 1 else False
            
            # 修正 volume_spike：用 'Volume' 並加 NaN 檢查
            vol_mean = data['Volume'].rolling(10).mean().iloc[-1]
            volume_spike = (not pd.isna(vol_mean)) and data['Volume'].iloc[-1] > vol_mean * 1.5 if len(data) > 10 else False
            
            obv_up = data['OBV'].diff().iloc[-1] > 0 if len(data) > 1 else False
            
            mfi_signal = data['MFI'].iloc[-1] > 20 and data['MFI'].iloc[-2] < 20 if len(data) > 1 else False
            
            # 信號計分（滿分 7）
            signals = [hist_increasing, divergence, rsi_signal, stoch_cross, volume_spike, obv_up, mfi_signal]
            score = sum(signals)
            
            # 交易建議
            suggestion = '無明顯買入信號。繼續監測。'
            if score >= 3:
                suggestion = '潛在買入機會：MACD Histogram 縮小，預測 MACD 可能即將從負轉正。建議關注。'
            if score >= 5:
                suggestion = '強烈買入信號：多指標確認，預測 MACD 即將交叉轉正。考慮進場，設止損。'
            
            # 顯示內容
            with placeholder.container():
                st.subheader('最新數據和指標')
                st.metric("最新收盤價", f"{data['Close'].iloc[-1]:.2f}")  # 用 metric 更美觀
                st.write(f'MACD Histogram: {histogram.iloc[-1]:.4f} (是否縮小: {"是" if hist_increasing else "否"})')
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
                st.dataframe(data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']])  # 只顯示主要欄位
                
                # 可選：加簡單圖表
                col1, col2 = st.columns(2)
                with col1:
                    st.line_chart(data['Close'].tail(50))
                with col2:
                    st.line_chart(data['Histogram'].tail(50))
        
        else:
            st.error('無法獲取數據，請檢查股票代碼或市場是否開盤。')
        
    except Exception as e:
        st.error(f'運行錯誤: {e}')
    
    time.sleep(refresh_minutes * 60)  # 等待指定分鐘
    st.rerun()  # 修正為 st.rerun()
