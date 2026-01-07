// Stock data types
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
    date_offset: number;
}
