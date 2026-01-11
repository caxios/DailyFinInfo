from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional
from pydantic import BaseModel
import json
import os

# Watchlists file path
WATCHLISTS_FILE = "watchlists.json"
MEMOS_FILE = "memos.json"
RETURNS_CACHE_FILE = "returns.json"

def load_watchlists():
    """Load watchlists from JSON file"""
    if os.path.exists(WATCHLISTS_FILE):
        with open(WATCHLISTS_FILE, "r") as f:
            return json.load(f)
    return {"categories": []}

def save_watchlists(data):
    """Save watchlists to JSON file"""
    with open(WATCHLISTS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_memos():
    """Load memos from JSON file"""
    if os.path.exists(MEMOS_FILE):
        with open(MEMOS_FILE, "r") as f:
            return json.load(f)
    return {"memos": []}

def save_memos(data):
    """Save memos to JSON file"""
    with open(MEMOS_FILE, "w") as f:
        json.dump(data, f, indent=2)

# Returns cache functions
def load_returns_cache():
    """Load returns cache from JSON file"""
    if os.path.exists(RETURNS_CACHE_FILE):
        with open(RETURNS_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_returns_cache(data):
    """Save returns cache to JSON file"""
    with open(RETURNS_CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_cached_stock(ticker: str, date: str):
    """Get cached stock data for ticker+date, returns None if not cached"""
    cache = load_returns_cache()
    cache_key = f"{ticker.upper()}_{date}"
    return cache.get(cache_key)

def cache_stock_data(ticker: str, date: str, data: dict):
    """Save stock data to cache"""
    cache = load_returns_cache()
    cache_key = f"{ticker.upper()}_{date}"
    data["cached_at"] = datetime.now().isoformat()
    cache[cache_key] = data
    save_returns_cache(cache)

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
async def get_stock_data(ticker: str, target_date: str = None):
    """
    Get intraday data for a stock.
    target_date: Date string in YYYY-MM-DD format (defaults to today)
    """
    try:
        # Normalize ticker and date
        ticker_upper = ticker.upper()
        
        # Parse target date from string, default to today
        if target_date:
            date_str = target_date
        else:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Check cache first
        cached = get_cached_stock(ticker_upper, date_str)
        if cached:
            print(f"CACHE HIT: {ticker_upper}_{date_str}")
            return StockData(
                ticker=cached["ticker"],
                company_name=cached["company_name"],
                current_price=cached["current_price"],
                change_percent=cached["change_percent"],
                timestamps=cached["timestamps"],
                returns=cached["returns"]
            )
        
        print(f"CACHE MISS: {ticker_upper}_{date_str} - fetching from yfinance")
        
        # Not in cache - fetch from yfinance
        stock = yf.Ticker(ticker_upper)
        
        # Get company info
        info = stock.info
        company_name = info.get("shortName", ticker_upper)
        
        # Parse date
        target_dt = datetime.strptime(date_str, "%Y-%m-%d")
        next_day = target_dt + timedelta(days=1)
        
        # Get previous close: end=target_dt (exclusive), so iloc[-1] is the day before target
        daily_data = stock.history(
            start=target_dt - timedelta(days=7),
            end=target_dt,
            interval="1d"
        )
        
        if daily_data.empty:
            raise HTTPException(status_code=404, detail=f"No daily data for {ticker}")
        
        prev_close = daily_data['Close'].iloc[-1]  # Previous day's close
        
        # Get intraday 5-minute data for target date
        intraday_data = stock.history(
            start=target_dt,
            end=next_day,
            interval="5m"
        )
        
        if intraday_data.empty:
            raise HTTPException(status_code=404, detail=f"No intraday data for {ticker} on this date")
        
        # Calculate returns vs previous close (for the intraday graph)
        returns = []
        for price in intraday_data["Close"]:
            ret = round(((price - prev_close) / prev_close) * 100, 2)
            returns.append(ret)
        
        # Format timestamps
        timestamps = [ts.strftime("%H:%M") for ts in intraday_data.index]
        
        # Current price = last intraday close
        current_price = round(intraday_data["Close"].iloc[-1], 2)
        
        # Daily change = last intraday return
        change_percent = returns[-1] if returns else 0.0
        
        # Save to cache
        cache_data = {
            "ticker": ticker_upper,
            "date": date_str,
            "company_name": company_name,
            "current_price": current_price,
            "change_percent": change_percent,
            "timestamps": timestamps,
            "returns": returns
        }
        cache_stock_data(ticker_upper, date_str, cache_data)
        
        return StockData(
            ticker=ticker_upper,
            company_name=company_name,
            current_price=current_price,
            change_percent=change_percent,
            timestamps=timestamps,
            returns=returns
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/all")
async def get_all_stocks(target_date: str = None):
    """Get data for all stocks in the watchlist"""
    results = []
    for ticker in watchlist:
        try:
            data = await get_stock_data(ticker, target_date)
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
    return {"stocks": results, "target_date": target_date}


# ==================== CATEGORY ENDPOINTS ====================

class CategoryCreate(BaseModel):
    name: str

class TickerAdd(BaseModel):
    ticker: str


@app.get("/api/categories")
async def get_categories():
    """Get all categories"""
    data = load_watchlists()
    return {"categories": data.get("categories", [])}


@app.post("/api/categories")
async def create_category(category: CategoryCreate):
    """Create a new category"""
    data = load_watchlists()
    
    # Generate ID from name
    category_id = category.name.lower().replace(" ", "_")
    
    # Check if already exists
    for cat in data["categories"]:
        if cat["id"] == category_id:
            raise HTTPException(status_code=400, detail="Category already exists")
    
    new_category = {
        "id": category_id,
        "name": category.name,
        "tickers": []
    }
    data["categories"].append(new_category)
    save_watchlists(data)
    
    return new_category


@app.delete("/api/categories/{category_id}")
async def delete_category(category_id: str):
    """Delete a category"""
    data = load_watchlists()
    data["categories"] = [c for c in data["categories"] if c["id"] != category_id]
    save_watchlists(data)
    return {"status": "deleted"}


@app.post("/api/categories/{category_id}/tickers")
async def add_ticker_to_category(category_id: str, ticker_data: TickerAdd):
    """Add a ticker to a category"""
    data = load_watchlists()
    
    for cat in data["categories"]:
        if cat["id"] == category_id:
            ticker = ticker_data.ticker.upper()
            if ticker not in cat["tickers"]:
                cat["tickers"].append(ticker)
                save_watchlists(data)
            return cat
    
    raise HTTPException(status_code=404, detail="Category not found")


@app.delete("/api/categories/{category_id}/tickers/{ticker}")
async def remove_ticker_from_category(category_id: str, ticker: str):
    """Remove a ticker from a category"""
    data = load_watchlists()
    
    for cat in data["categories"]:
        if cat["id"] == category_id:
            ticker_upper = ticker.upper()
            if ticker_upper in cat["tickers"]:
                cat["tickers"].remove(ticker_upper)
                save_watchlists(data)
            return cat
    
    raise HTTPException(status_code=404, detail="Category not found")


@app.get("/api/categories/{category_id}/stocks")
async def get_category_stocks(category_id: str, target_date: str = None):
    """Get stock data for all tickers in a category"""
    data = load_watchlists()
    
    for cat in data["categories"]:
        if cat["id"] == category_id:
            results = []
            for ticker in cat["tickers"]:
                try:
                    stock_data = await get_stock_data(ticker, target_date)
                    results.append(stock_data.model_dump())
                except HTTPException:
                    results.append({
                        "ticker": ticker,
                        "company_name": ticker,
                        "current_price": 0,
                        "change_percent": 0,
                        "timestamps": [],
                        "returns": [],
                        "error": True
                    })
            return {"category": cat, "stocks": results, "target_date": target_date}
    
    raise HTTPException(status_code=404, detail="Category not found")


# ==================== MEMO ENDPOINTS ====================

class MemoCreate(BaseModel):
    ticker: str
    date: str
    content: str


@app.get("/api/memos")
async def get_all_memos():
    """Get all memos (for notebook hub)"""
    data = load_memos()
    return {"memos": data.get("memos", [])}


@app.get("/api/memos/{ticker}/{date}")
async def get_memo(ticker: str, date: str):
    """Get memo for a specific stock and date"""
    data = load_memos()
    ticker_upper = ticker.upper()
    
    for memo in data["memos"]:
        if memo["ticker"] == ticker_upper and memo["date"] == date:
            return memo
    
    return {"ticker": ticker_upper, "date": date, "content": ""}


@app.post("/api/memos")
async def create_or_update_memo(memo: MemoCreate):
    """Create or update a memo"""
    data = load_memos()
    ticker_upper = memo.ticker.upper()
    now = datetime.now().isoformat()
    
    # Check if memo exists
    for existing in data["memos"]:
        if existing["ticker"] == ticker_upper and existing["date"] == memo.date:
            existing["content"] = memo.content
            existing["updated_at"] = now
            save_memos(data)
            return existing
    
    # Create new memo
    new_memo = {
        "id": f"{ticker_upper}_{memo.date}",
        "ticker": ticker_upper,
        "date": memo.date,
        "content": memo.content,
        "created_at": now,
        "updated_at": now
    }
    data["memos"].append(new_memo)
    save_memos(data)
    
    return new_memo


@app.delete("/api/memos/{ticker}/{date}")
async def delete_memo(ticker: str, date: str):
    """Delete a memo"""
    data = load_memos()
    ticker_upper = ticker.upper()
    
    data["memos"] = [m for m in data["memos"] 
                     if not (m["ticker"] == ticker_upper and m["date"] == date)]
    save_memos(data)
    
    return {"status": "deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
