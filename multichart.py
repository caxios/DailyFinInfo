import yfinance as yf
from datetime import datetime, timedelta
from typing import List
from pydantic import BaseModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os
import json

def create_multi_comparison_charts(ticker_groups: List[dict], period="1y"):
    """
    여러 개의 비교 차트를 하나의 페이지에 표시합니다.
    
    ticker_groups: 차트별 티커 그룹 리스트
        예: [
            {"title": "Tech Giants", "tickers": ["AAPL", "MSFT", "GOOGL"]},
            {"title": "EV Companies", "tickers": ["TSLA", "RIVN", "LCID"]},
            {"title": "Crypto", "tickers": ["BTC-USD", "ETH-USD"]}
        ]
    period: 기간 (예: '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max', 'ytd')
    """
    
    num_charts = len(ticker_groups)
    
    # 서브플롯 생성 (세로로 배치)
    fig = make_subplots(
        rows=num_charts, 
        cols=1,
        subplot_titles=[group.get("title", f"Chart {i+1}") for i, group in enumerate(ticker_groups)],
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    # 각 그룹에 대해 차트 추가
    for idx, group in enumerate(ticker_groups, start=1):
        tickers = group["tickers"]
        title = group.get("title", f"Chart {idx}")
        
        print(f"[{title}] {tickers} 데이터를 다운로드 중...")
        
        df = yf.download(tickers, period=period, progress=False)['Close']
        
        if df.empty:
            print(f"[{title}] 데이터를 가져오지 못했습니다.")
            continue
        
        # 단일 티커인 경우 Series를 DataFrame으로 변환
        if isinstance(df, type(df)) and not hasattr(df, 'columns'):
            df = df.to_frame(name=tickers[0])
        elif len(tickers) == 1:
            df.columns = [tickers[0]]
            
        # 누적 수익률 계산
        cumulative_returns = (df / df.iloc[0] - 1) * 100
        
        # 각 티커에 대해 라인 추가
        for ticker in cumulative_returns.columns:
            fig.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns[ticker],
                    name=f"{ticker}",
                    legendgroup=f"group{idx}",
                    legendgrouptitle_text=title,
                    mode='lines',
                    hovertemplate=f"{ticker}: %{{y:.2f}}%<extra></extra>"
                ),
                row=idx,
                col=1
            )
        
        # Y축 레이블 설정
        fig.update_yaxes(ticksuffix="%", title_text="수익률 (%)", row=idx, col=1)
    
    # 전체 레이아웃 설정
    fig.update_layout(
        height=400 * num_charts,  # 차트당 400px
        title_text=f"주식 누적 수익률 비교 ({period})",
        title_x=0.5,
        hovermode="x unified",
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # X축 레이블 (마지막 차트에만)
    fig.update_xaxes(title_text="날짜", row=num_charts, col=1)
    
    # HTML 파일로 저장 및 브라우저에서 열기
    output_file = os.path.join(os.path.dirname(__file__), "multi_stock_charts.html")
    fig.write_html(output_file)
    webbrowser.open('file://' + os.path.realpath(output_file))
    print(f"\n차트가 저장되었습니다: {output_file}")
    
    return fig


if __name__ == "__main__":
    # watchlists.json에서 그룹 정보 로드
    watchlist_path = os.path.join(os.path.dirname(__file__), "watchlists.json")
    with open(watchlist_path, "r", encoding="utf-8") as f:
        watchlist_data = json.load(f)
    
    # 카테고리 정보를 my_ticker_groups 형식으로 변환
    my_ticker_groups = [
        {"title": category["name"], "tickers": category["tickers"]}
        for category in watchlist_data.get("categories", [])
    ]
    
    # 차트 생성 (기간: 1년)
    create_multi_comparison_charts(my_ticker_groups, period="3mo")