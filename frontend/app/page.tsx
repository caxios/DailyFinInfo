'use client';

import { useState, useEffect } from 'react';
import { Stock, StocksResponse } from '@/types';
import StockCard from '@/components/StockCard';
import DateNav from '@/components/DateNav';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

export default function Home() {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [dateOffset, setDateOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchStocks();
  }, [dateOffset]);

  async function fetchStocks() {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/api/stocks/all?date_offset=${dateOffset}`);
      if (!response.ok) throw new Error('Failed to fetch stocks');

      const data: StocksResponse = await response.json();
      setStocks(data.stocks);
    } catch (err) {
      setError('Error loading stocks. Is the backend running?');
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 text-white p-4">
      <div className="max-w-xl mx-auto">
        {/* Header */}
        <header className="flex justify-between items-center py-4 mb-4 border-b border-slate-800">
          <h1 className="text-xl font-semibold">📈 Intraday Tracker</h1>
        </header>

        {/* Date Navigation */}
        <DateNav
          dateOffset={dateOffset}
          onPrev={() => setDateOffset(prev => prev + 1)}
          onNext={() => setDateOffset(prev => Math.max(0, prev - 1))}
        />

        {/* Stock List */}
        <div className="flex flex-col gap-3">
          {loading ? (
            <div className="text-center py-12 text-slate-400">
              <div className="w-10 h-10 border-3 border-slate-600 border-t-blue-500 rounded-full animate-spin mx-auto mb-4" />
              <p>Loading stocks...</p>
            </div>
          ) : error ? (
            <div className="text-center py-12 text-slate-400">
              <p>{error}</p>
            </div>
          ) : (
            stocks.map((stock) => (
              <StockCard key={stock.ticker} stock={stock} />
            ))
          )}
        </div>
      </div>
    </main>
  );
}
