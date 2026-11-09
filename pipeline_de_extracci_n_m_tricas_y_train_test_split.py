"""
Pipeline completo para TFG:
1. Descarga de datos históricos desde Yahoo Finance (S&P 500 o Cobre).
2. Ingeniería de Características (Feature Engineering): Cálculo de todos los medidores
   estructurales (paper DI2XAI), tendencia, impulso, volatilidad, volumen y velas.
3. Creación de la etiqueta de clasificación (Target) basada en márgenes de rentabilidad.
4. Limpieza, partición temporal (Train/Test Split sin Data Leakage) y escalado para KNN/MLP.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. CONFIGURACIÓN DEL PROYECTO (Parámetros ajustables)
# ==============================================================================
TICKER = "^GSPC"          # "^GSPC" para S&P 500 | "HG=F" para Futuros del Cobre
START_DATE = "2010-01-01"
END_DATE = "2026-01-01"

# Parámetros de la estrategia de inversión (Target / Etiqueta)
HORIZON_DAYS = 20         # Horizonte máximo de espera en sesiones (ej. 20 días ~ 1 mes, 252 ~ 1 año)
PROFIT_TARGET_PCT = 3.0   # Margen de beneficio objetivo en % (ej. +3% en 20 días, o +12% en 252 días)
USE_SMOOTHED_EXIT = False # False: mira si el precio máximo/cierre toca el margen.
                          # True: mira la Media Móvil futura (estilo largo plazo paper DI2XAI).

TEST_SIZE_RATIO = 0.20    # 80% pasado para entrenar (Train), 20% más reciente para validar (Test)


# ==============================================================================
# 2. DESCARGA Y ESTANDARIZACIÓN DEL CSV
# ==============================================================================
def download_and_prepare_raw_data(ticker: str, start: str, end: str, csv_path: str) -> pd.DataFrame:
    print(f"[1/4] Descargando historial de {ticker} ({start} a {end})...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)

    # En versiones recientes de yfinance (1.x), las columnas pueden venir como MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Nos quedamos con las columnas estándar OHLCV y eliminamos filas vacías
    cols_needed = ["Open", "High", "Low", "Close", "Volume"]
    df = df[cols_needed].dropna().copy()

    # Guardamos el CSV bruto por si queréis inspeccionarlo en Excel
    df.to_csv(csv_path)
    print(f"      -> CSV bruto guardado en '{csv_path}' con {len(df)} sesiones bursátiles.")
    return df


# ==============================================================================
# 3. INGENIERÍA DE CARACTERÍSTICAS (Cálculo de todos los medidores)
# ==============================================================================
def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/4] Calculando indicadores estructurales, técnicos y de microestructura...")
    data = df.copy()

    close = data["Close"]
    open_p = data["Open"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    # --------------------------------------------------------------------------
    # A. MEDIDORES ESTRUCTURALES E HISTÓRICOS (Paper DI2XAI)
    # --------------------------------------------------------------------------
    # Máximo histórico conocido hasta el día t (expanding().max() evita mirar al futuro)
    running_max = close.expanding(min_periods=1).max()

    # 1. dif2max: % de caída actual desde el máximo histórico conocido
    data["dif2max"] = ((running_max - close) / running_max) * 100.0

    # 2. dif2maxAvg: Promedio histórico de dif2max hasta la fecha actual (riesgo estructural)
    data["dif2maxAvg"] = data["dif2max"].expanding(min_periods=20).mean()

    # 3. profitMa3years: Rentabilidad en 3 años (756 sesiones) medida sobre la MA20
    ma20_raw = close.rolling(window=20).mean()
    data["profitMa3years"] = ((ma20_raw - ma20_raw.shift(756)) / ma20_raw.shift(756)) * 100.0

    # --------------------------------------------------------------------------
    # B. MEDIDORES DE TENDENCIA RELATIVA (Medias Móviles Estacionarias)
    # --------------------------------------------------------------------------
    for window in [5, 10, 20, 50, 100]:
        ma = close.rolling(window=window).mean()
        # Distancia porcentual del precio respecto a su media móvil
        data[f"dist_MA{window}"] = ((close - ma) / ma) * 100.0

    # Ratios de cruce entre medias rápidas y lentas
    ma10 = close.rolling(window=10).mean()
    ma20 = close.rolling(window=20).mean()
    ma50 = close.rolling(window=50).mean()
    ma100 = close.rolling(window=100).mean()

    data["ratio_MA10_MA50"] = (ma10 / ma50) - 1.0
    data["ratio_MA20_MA100"] = (ma20 / ma100) - 1.0

    # Pendiente (Slope) de la media de 50 sesiones en los últimos 5 días (%)
    data["slope_MA50_5d"] = ma50.pct_change(periods=5) * 100.0

    # --------------------------------------------------------------------------
    # C. MEDIDORES DE IMPULSO Y AGOTAMIENTO (Momentum)
    # --------------------------------------------------------------------------
    # 1. RSI de 14 sesiones (Suavizado exponencial de Wilder)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    data["rsi_14"] = data["rsi_14"].fillna(50.0)

    # 2. MACD Normalizado por el precio (EMA 12, EMA 26, Señal 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    data["macd_norm"] = (macd_line / close) * 100.0
    data["macd_hist_norm"] = (macd_hist / close) * 100.0

    # 3. Retornos porcentuales recientes (ROC: 1 día, 3 días, 5 días, 20 días)
    for days in [1, 3, 5, 20]:
        data[f"ret_{days}d"] = close.pct_change(periods=days) * 100.0

    # 4. Posición Estocástica (%K de 14 sesiones)
    low_14 = low.rolling(window=14).min()
    high_14 = high.rolling(window=14).max()
    denom_stoch = (high_14 - low_14).replace(0, np.nan)
    data["stoch_k_14"] = ((close - low_14) / denom_stoch) * 100.0

    # --------------------------------------------------------------------------
    # D. MEDIDORES DE VOLATILIDAD Y RÉGIMEN DE MERCADO
    # --------------------------------------------------------------------------
    # 1. NATR (Normalized Average True Range - 14 sesiones)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = true_range.rolling(window=14).mean()
    data["natr_14"] = (atr_14 / close) * 100.0

    # 2. Bandas de Bollinger (20 sesiones, 2 desviaciones estándar)
    std_20 = close.rolling(window=20).std()
    bb_upper = ma20 + (2.0 * std_20)
    bb_lower = ma20 - (2.0 * std_20)
    data["bb_width"] = ((bb_upper - bb_lower) / ma20) * 100.0
    data["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    # --------------------------------------------------------------------------
    # E. FLUJO DE DINERO Y MICROESTRUCTURA DE VELAS
    # --------------------------------------------------------------------------
    # 1. MFI (Money Flow Index - 14 sesiones)
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume
    tp_diff = typical_price.diff()

    pos_flow = np.where(tp_diff > 0, raw_money_flow, 0.0)
    neg_flow = np.where(tp_diff < 0, raw_money_flow, 0.0)

    pos_mf_14 = pd.Series(pos_flow, index=data.index).rolling(window=14).sum()
    neg_mf_14 = pd.Series(neg_flow, index=data.index).rolling(window=14).sum()

    money_ratio = pos_mf_14 / neg_mf_14.replace(0, np.nan)
    data["mfi_14"] = 100.0 - (100.0 / (1.0 + money_ratio))
    data["mfi_14"] = data["mfi_14"].fillna(50.0)

    # 2. RVOL (Volumen Relativo respecto a la media de 20 sesiones)
    vol_ma20 = volume.rolling(window=20).mean().replace(0, np.nan)
    data["rvol_20"] = volume / vol_ma20
    data["rvol_20"] = data["rvol_20"].fillna(1.0)

    # 3. Geometría de la vela diaria (Cuerpo y Mechas respecto al rango diario)
    candle_range = (high - low).replace(0, np.nan)
    data["candle_body_ratio"] = (close - open_p) / candle_range
    data["upper_wick_ratio"] = (high - np.maximum(open_p, close)) / candle_range
    data["lower_wick_ratio"] = (np.minimum(open_p, close) - low) / candle_range
    data[["candle_body_ratio", "upper_wick_ratio", "lower_wick_ratio"]] = (
        data[["candle_body_ratio", "upper_wick_ratio", "lower_wick_ratio"]].fillna(0.0)
    )

    # 4. Estacionalidad (Día de la semana: 0=Lunes ... 4=Viernes)
    data["day_of_week"] = data.index.dayofweek

    return data


# ==============================================================================
# 4. ETIQUETADO POR MÁRGENES DE RENTABILIDAD (Creación del Target)
# ==============================================================================
def create_profit_margin_target(
    df: pd.DataFrame, horizon: int, target_pct: float, use_smoothed_exit: bool = False
) -> pd.DataFrame:
    """
    Crea la columna 'Target' (1 = Comprar / Oportunidad alcanza el margen; 0 = No alcanza).
    - Si use_smoothed_exit=False: Comprueba si en los próximos 'horizon' días el precio
      de cierre máximo supera el precio actual en al menos 'target_pct'%.
    - Si use_smoothed_exit=True: Replica exactamente la etiqueta de NpcML del paper DI2XAI
      (compara la Media Móvil dentro de 'horizon' días con el precio actual).
    """
    print(f"[3/4] Generando etiqueta Target (Margen: +{target_pct}% en horizonte de {horizon} sesiones)...")
    data = df.copy()
    close = data["Close"]

    if use_smoothed_exit:
        # Estilo NpcML del paper: mira la MA20 situada 'horizon' días en el futuro
        future_ma = close.rolling(window=20).mean().shift(-horizon)
        future_return_pct = ((future_ma - close) / close) * 100.0
    else:
        # Enfoque de ventana futura: busca el mejor cierre dentro de las próximas 'horizon' sesiones
        # Invertimos la serie para usar rolling().max() mirando hacia adelante y volvemos a invertir
        future_max_close = close.iloc[::-1].rolling(window=horizon, min_periods=1).max().iloc[::-1].shift(-1)
        future_return_pct = ((future_max_close - close) / close) * 100.0

    # 1 si supera el margen objetivo, 0 en caso contrario
    data["Target"] = (future_return_pct >= target_pct).astype(int)

    # IMPORTANTE: Eliminamos las últimas 'horizon' filas porque no tienen futuro completo conocido
    data = data.iloc[:-horizon].copy()

    # Eliminamos también las primeras filas que tengan NaN por el cálculo de medias largas (ej. 3 años)
    data = data.dropna().copy()

    return data


# ==============================================================================
# 5. PARTICIÓN TEMPORAL (TRAIN/TEST SPLIT) Y ESCALADO SIN DATA LEAKAGE
# ==============================================================================
def chronological_train_test_split(df: pd.DataFrame, test_ratio: float = 0.20):
    print("[4/4] Ejecutando Train/Test Split cronológico (shuffle=False) y escalado...")

    # Definimos cuáles son las columnas de entrada (Features) excluyendo precios brutos y Target
    raw_cols = ["Open", "High", "Low", "Close", "Volume", "Target"]
    feature_cols = [c for c in df.columns if c not in raw_cols]

    X = df[feature_cols]
    y = df["Target"]

    # Índice de corte temporal estricto (el pasado entrena, el futuro evalúa)
    split_idx = int(len(df) * (1.0 - test_ratio))

    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

    # Escalado estándar (imprescindible para que KNN y MLP funcionen correctamente)
    # REGLA DE ORO: .fit() SOLO sobre X_train para no filtrar estadísticas del futuro
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), index=X_train.index, columns=feature_cols
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), index=X_test.index, columns=feature_cols
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols


# ==============================================================================
# BLOQUE PRINCIPAL DE EJECUCIÓN
# ==============================================================================
if __name__ == "__main__":
    raw_csv_name = f"dataset_bruto_{TICKER.replace('^', '').replace('=', '_')}.csv"
    processed_csv_name = f"dataset_indicadores_{TICKER.replace('^', '').replace('=', '_')}.csv"

    # Paso 1: Descargar CSV
    df_raw = download_and_prepare_raw_data(TICKER, START_DATE, END_DATE, raw_csv_name)

    # Paso 2: Calcular todos los indicadores técnicos y estructurales
    df_features = compute_all_indicators(df_raw)

    # Paso 3: Etiquetar por margen de rentabilidad y limpiar NaNs
    df_final = create_profit_margin_target(
        df_features,
        horizon=HORIZON_DAYS,
        target_pct=PROFIT_TARGET_PCT,
        use_smoothed_exit=USE_SMOOTHED_EXIT,
    )

    # Guardamos el dataset procesado completo
    df_final.to_csv(processed_csv_name)
    print(f"      -> Dataset con indicadores guardado en '{processed_csv_name}'.")

    # Paso 4: Separar en Train y Test cronológico y escalar
    X_train, X_test, y_train, y_test, features = chronological_train_test_split(
        df_final, test_ratio=TEST_SIZE_RATIO
    )

    # Resumen por consola para verificar que todo está listo
    print("\n" + "=" * 75)
    print("RESUMEN DEL DATASET LISTO PARA KNN Y PERCEPTRÓN (MLP)")
    print("=" * 75)
    print(f"Activo analizado          : {TICKER}")
    print(f"Total de sesiones válidas : {len(df_final)} días")
    print(f"Número de indicadores (X) : {len(features)} features")
    print(f"Lista de features         : {features}")
    print("-" * 75)
    print(f"Conjunto de ENTRENAMIENTO : {len(X_train)} sesiones ({X_train.index.min().date()} a {X_train.index.max().date()})")
    print(f"  -> Distribución Target  : {y_train.mean()*100:.1f}% oportunidades positivas (Clase 1)")
    print(f"Conjunto de PRUEBA (TEST) : {len(X_test)} sesiones ({X_test.index.min().date()} a {X_test.index.max().date()})")
    print(f"  -> Distribución Target  : {y_test.mean()*100:.1f}% oportunidades positivas (Clase 1)")
    print("=" * 75)