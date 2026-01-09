export interface Stock {
    ticker: string;
    company_name: string;
    current_price: number;
    change_percent: number;
    timestamps: string[];
    returns: number[];
    error?: boolean;
}

export interface StocksResponse {
    stocks: Stock[];
    target_date: string | null;
}

export interface Category {
    id: string;
    name: string;
    tickers: string[];
}

export interface CategoriesResponse {
    categories: Category[];
}

export interface CategoryStocksResponse {
    category: Category;
    stocks: Stock[];
    target_date: string | null;
}

export interface Memo {
    id?: string;
    ticker: string;
    date: string;
    content: string;
    created_at?: string;
    updated_at?: string;
}

export interface MemosResponse {
    memos: Memo[];
}
