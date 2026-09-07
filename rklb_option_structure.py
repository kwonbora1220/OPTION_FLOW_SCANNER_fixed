import os
import sys
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import requests

# Matplotlib 한글 및 깨짐 방지 기본 설정
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
SYMBOL = os.getenv("SYMBOL", "RKLB")
MIN_STRIKE = float(os.getenv("MIN_STRIKE", "50"))
MAX_STRIKE = float(os.getenv("MAX_STRIKE", "100"))
MAX_DTE = int(os.getenv("MAX_DTE", "365"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ---------------------------------------------------------
# Fetch & Process Option Data
# ---------------------------------------------------------
def fetch_option_data(symbol):
    ticker_obj = yf.Ticker(symbol)
    
    # Spot Price
    try:
        spot_price = ticker_obj.fast_info['lastPrice']
    except Exception:
        hist = ticker_obj.history(period="1d")
        spot_price = hist['Close'].iloc[-1] if not hist.empty else 0.0

    # yfinance 버전 이슈 대응 (expirations -> options)
    expirations = getattr(ticker_obj, 'options', None)
    if expirations is None or len(expirations) == 0:
        expirations = getattr(ticker_obj, 'expirations', ())

    if not expirations:
        print("옵션 만기일 데이터를 불러올 수 없습니다.")
        return spot_price, pd.DataFrame()

    today = datetime.datetime.now().date()
    all_contracts = []

    for exp_str in expirations:
        exp_date = datetime.datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte <= 0 or dte > MAX_DTE:
            continue

        try:
            opt_chain = ticker_obj.option_chain(exp_str)
        except Exception:
            continue

        for opt_type, df in [('CALL', opt_chain.calls), ('PUT', opt_chain.puts)]:
            if df.empty:
                continue
            
            df = df.copy()
            df = df[(df['strike'] >= MIN_STRIKE) & (df['strike'] <= MAX_STRIKE)]
            if df.empty:
                continue

            df['type'] = opt_type
            df['expiration'] = exp_str
            df['dte'] = dte
            
            # Mid Price
            df['midPrice'] = (df['bid'].fillna(0) + df['ask'].fillna(0)) / 2.0
            df['midPrice'] = np.where(df['midPrice'] == 0, df['lastPrice'], df['midPrice'])
            
            # Premium
            df['premium'] = df['volume'].fillna(0) * df['midPrice'] * 100
            
            # GEX Proxy (gamma가 없으면 delta 기반 간이 추정)
            gamma = df['gamma'] if 'gamma' in df.columns else 0.01
            df['gex'] = gamma * df['openInterest'].fillna(0) * 100 * (spot_price ** 2) * 0.01
            if opt_type == 'PUT':
                df['gex'] = -df['gex']

            all_contracts.append(df)

    if not all_contracts:
        return spot_price, pd.DataFrame()

    full_df = pd.concat(all_contracts, ignore_index=True)
    return spot_price, full_df

# ---------------------------------------------------------
# Create Clean Dashboard Image (깨짐 방지 레이아웃 적용)
# ---------------------------------------------------------
def create_dashboard(spot_price, full_df, output_img="option_dashboard.png"):
    if full_df.empty:
        print("시각화할 데이터가 없습니다.")
        return

    # 다크 테마 설정
    plt.style.use('dark_background')
    
    # 넉넉한 피규어 크기 설정 (가로 18, 세로 12인치)
    fig = plt.figure(figsize=(18, 12), dpi=300)
    
    # 2x2 그리드 레이아웃 생성
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    # Strike 기준 집계
    strike_grp = full_df.groupby(['strike', 'type'])[['volume', 'openInterest', 'premium', 'gex']].sum().unstack(fill_value=0)
    strikes = strike_grp.index

    call_vol = strike_grp['volume']['CALL'] if 'CALL' in strike_grp['volume'] else pd.Series(0, index=strikes)
    put_vol = strike_grp['volume']['PUT'] if 'PUT' in strike_grp['volume'] else pd.Series(0, index=strikes)
    
    call_oi = strike_grp['openInterest']['CALL'] if 'CALL' in strike_grp['openInterest'] else pd.Series(0, index=strikes)
    put_oi = strike_grp['openInterest']['PUT'] if 'PUT' in strike_grp['openInterest'] else pd.Series(0, index=strikes)

    # 1. Volume Profile (Top-Left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(strikes, call_vol, height=0.4, color='#2ecc71', alpha=0.8, label='Call Volume')
    ax1.barh(strikes, -put_vol, height=0.4, color='#e74c3c', alpha=0.8, label='Put Volume')
    ax1.axhline(spot_price, color='#f1c40f', linestyle='--', linewidth=2, label=f'Spot Price (${spot_price:.2f})')
    ax1.set_title(f"{SYMBOL} Volume Profile by Strike", fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel("Volume", fontsize=11)
    ax1.set_ylabel("Strike Price ($)", fontsize=11)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{abs(x):,.0f}'))
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.4)

    # 2. Open Interest Profile (Top-Right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(strikes, call_oi, height=0.4, color='#2ecc71', alpha=0.8, label='Call OI')
    ax2.barh(strikes, -put_oi, height=0.4, color='#e74c3c', alpha=0.8, label='Put OI')
    ax2.axhline(spot_price, color='#f1c40f', linestyle='--', linewidth=2, label=f'Spot Price (${spot_price:.2f})')
    ax2.set_title(f"{SYMBOL} Open Interest Profile by Strike", fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel("Open Interest", fontsize=11)
    ax2.set_ylabel("Strike Price ($)", fontsize=11)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{abs(x):,.0f}'))
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.4)

    # 3. Expiration Breakdown (Bottom-Left)
    exp_grp = full_df.groupby(['expiration', 'type'])['openInterest'].sum().unstack(fill_value=0)
    ax3 = fig.add_subplot(gs[1, 0])
    exp_grp.plot(kind='bar', stacked=True, ax=ax3, color=['#2ecc71', '#e74c3c'], alpha=0.8)
    ax3.set_title(f"{SYMBOL} Open Interest by Expiration", fontsize=14, fontweight='bold', pad=10)
    ax3.set_xlabel("Expiration Date", fontsize=11)
    ax3.set_ylabel("Total Open Interest", fontsize=11)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, linestyle=':', alpha=0.4)

    # 4. Summary Table / Key Metrics (Bottom-Right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    total_call_vol = call_vol.sum()
    total_put_vol = put_vol.sum()
    total_call_oi = call_oi.sum()
    total_put_oi = put_oi.sum()
    pc_ratio_vol = total_put_vol / total_call_vol if total_call_vol > 0 else 0
    pc_ratio_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0

    summary_text = (
        f"  === {SYMBOL} OPTIONS SUMMARY DASHBOARD ===\n\n"
        f"  • Current Spot Price: ${spot_price:.2f}\n"
        f"  • Total Call Volume : {total_call_vol:,.0f}\n"
        f"  • Total Put Volume  : {total_put_vol:,.0f}\n"
        f"  • Volume P/C Ratio  : {pc_ratio_vol:.2f}\n\n"
        f"  • Total Call OI     : {total_call_oi:,.0f}\n"
        f"  • Total Put OI      : {total_put_oi:,.0f}\n"
        f"  • OI P/C Ratio      : {pc_ratio_oi:.2f}\n\n"
        f"  • Max Call Strike   : ${call_oi.idxmax() if not call_oi.empty else 0}\n"
        f"  • Max Put Strike    : ${put_oi.idxmax() if not put_oi.empty else 0}\n"
    )

    ax4.text(0.1, 0.5, summary_text, fontsize=13, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round,pad=1', facecolor='#1e272e', edgecolor='#34495e'))

    # 여백 및 겹침 자동 교정
    plt.tight_layout()
    
    # 텍스트 잘림 방지 및 고해상도 저장
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"대시보드가 성공적으로 생성되었습니다: {output_img}")

# ---------------------------------------------------------
# Telegram Bot Dispatcher
# ---------------------------------------------------------
def send_telegram_notification(msg, img_path=None):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    # Text
    text_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(text_url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})

    # Image
    if img_path and os.path.exists(img_path):
        photo_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        with open(img_path, 'rb') as f:
            requests.post(photo_url, data={"chat_id": TELEGRAM_CHAT_ID}, files={"photo": f})

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    print(f"[{SYMBOL}] 옵션 데이터 수집 및 분석 시작...")
    spot_price, full_df = fetch_option_data(SYMBOL)

    if full_df.empty:
        print("데이터를 가져오는 데 실패했거나 조건에 맞는 옵션 데이터가 없습니다.")
        return

    # CSV 데이터 저장
    full_df.to_csv("contracts.csv", index=False)
    print("계약 데이터 저장 완료: contracts.csv")

    # 대시보드 이미지 생성
    create_dashboard(spot_price, full_df, "option_dashboard.png")

    # 텔레그램 전송
    report_msg = f"📊 [{SYMBOL}] 옵션 스캐너 분석 완료\n- Spot Price: ${spot_price:.2f}\n- Total Contracts: {len(full_df):,}개"
    send_telegram_notification(report_msg, "option_dashboard.png")

if __name__ == "__main__":
    main()
