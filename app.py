from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from datetime import datetime, timedelta
from typing import List
from pydantic import BaseModel

# Import watchlist
from _watchlist import watchlist

app = FastAPI(title="Intraday Stock Tracker API")

# Add CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StockData(BaseModel):
    ticker: str
    company_name: str
    current_price: float
    change_percent: float
    timestamps: List[str]
    returns: List[float]


@app.get("/")
async def root():
    """API health check"""
    return {"status": "ok", "message": "Intraday Stock Tracker API"}


@app.get("/api/watchlist")
async def get_watchlist():
    """Return the list of tracked stocks"""
    return {"stocks": watchlist}


@app.get("/api/stock/{ticker}")
async def get_stock_data(ticker: str, date_offset: int = 0):
    """
    Get intraday data for a stock.
    date_offset: 0 = today, 1 = yesterday, 2 = 2 days ago, etc.
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get company info
        info = stock.info
        company_name = info.get("shortName", ticker)
        
        # Calculate target date
        target_date = datetime.now() - timedelta(days=date_offset)
        
        # For historical dates, we need to fetch daily data first
        # to get the previous day's close
        if date_offset == 0:
            # Today: use 1d period with 5m interval
            intraday_data = stock.history(period="1d", interval="5m")
            daily_data = stock.history(period="2d", interval="1d")
        else:
            # Historical: need to calculate date ranges
            end_date = target_date + timedelta(days=1)
            start_date = target_date - timedelta(days=1)
            
            # Get intraday data for the target date
            intraday_data = stock.history(
                start=target_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="5m"
            )
            
            # Get daily data to find previous close
            daily_data = stock.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1d"
            )
        
        if intraday_data.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {ticker} on this date")
        
        if daily_data.empty or len(daily_data) < 1:
            raise HTTPException(status_code=404, detail=f"No daily data available for {ticker}")
        
        # Get previous day's close price
        prev_close = daily_data.iloc[0]["Close"]
        
        # Calculate returns vs previous close
        returns = []
        for price in intraday_data["Close"]:
            ret = round(((price - prev_close) / prev_close) * 100, 2)
            returns.append(ret)
        
        # Format timestamps
        timestamps = [ts.strftime("%H:%M") for ts in intraday_data.index]
        
        # Current price and change
        current_price = round(intraday_data["Close"].iloc[-1], 2)
        change_percent = returns[-1] if returns else 0.0
        
        return StockData(
            ticker=ticker,
            company_name=company_name,
            current_price=current_price,
            change_percent=change_percent,
            timestamps=timestamps,
            returns=returns
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/all")
async def get_all_stocks(date_offset: int = 0):
    """Get data for all stocks in the watchlist"""
    results = []
    for ticker in watchlist:
        try:
            data = await get_stock_data(ticker, date_offset)
            results.append(data.model_dump())
        except HTTPException:
            # Skip failed stocks
            results.append({
                "ticker": ticker,
                "company_name": ticker,
                "current_price": 0,
                "change_percent": 0,
                "timestamps": [],
                "returns": [],
                "error": True
            })
    return {"stocks": results, "date_offset": date_offset}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
