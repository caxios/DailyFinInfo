'use client';

import { useState, useEffect } from 'react';
import { Category, Stock } from '@/types';
import StockCard from './StockCard';

interface CategoryItemProps {
    category: Category;
    selectedDate: string;
    onAddTicker: (categoryId: string) => void;
    onDeleteCategory: (categoryId: string) => void;
    onRemoveTicker: (categoryId: string, ticker: string) => void;
    onMemoClick: (ticker: string, companyName: string) => void;
    memoTickers: string[];  // List of tickers that have memos on selected date
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

export default function CategoryItem({
    category,
    selectedDate,
    onAddTicker,
    onDeleteCategory,
    onRemoveTicker,
    onMemoClick,
    memoTickers
}: CategoryItemProps) {
    const [isExpanded, setIsExpanded] = useState(false);
    const [stocks, setStocks] = useState<Stock[]>([]);
    const [loading, setLoading] = useState(false);

    // Fetch stocks function
    const fetchStocks = async () => {
        setLoading(true);
        try {
            const response = await fetch(
                `${API_URL}/api/categories/${category.id}/stocks?target_date=${selectedDate}`
            );
            const data = await response.json();
            setStocks(data.stocks || []);
        } catch (err) {
            console.error('Failed to load stocks:', err);
        } finally {
            setLoading(false);
        }
    };

    // Re-fetch when date changes while expanded
    useEffect(() => {
        if (isExpanded) {
            fetchStocks();
        }
    }, [selectedDate]);

    const toggleExpand = async () => {
        if (!isExpanded) {
            // Fetch stocks when expanding
            await fetchStocks();
        }
        setIsExpanded(!isExpanded);
    };

    return (
        <div className="bg-slate-800/50 rounded-xl overflow-hidden mb-2">
            {/* Category Header */}
            <div
                className="flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-slate-700/50 transition-all"
                onClick={toggleExpand}
            >
                <div className="flex items-center gap-3">
                    <span className="text-white font-medium">{category.name}</span>
                    <span className="text-slate-400 text-sm">({category.tickers.length})</span>
                </div>

                <div className="flex items-center gap-2">
                    {/* Edit button */}
                    <button
                        onClick={(e) => {
                            e.stopPropagation();
                            onAddTicker(category.id);
                        }}
                        className="p-1.5 rounded-lg hover:bg-slate-600 transition-all text-slate-400 hover:text-white"
                        title="Add ticker"
                    >
                        ➕
                    </button>

                    {/* Delete button */}
                    <button
                        onClick={(e) => {
                            e.stopPropagation();
                            onDeleteCategory(category.id);
                        }}
                        className="p-1.5 rounded-lg hover:bg-red-600 transition-all text-slate-400 hover:text-white"
                        title="Delete category"
                    >
                        🗑️
                    </button>

                    {/* Expand/collapse arrow */}
                    <span className={`text-slate-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}>
                        ▼
                    </span>
                </div>
            </div>

            {/* Expanded Content */}
            {isExpanded && (
                <div className="px-4 pb-4 border-t border-slate-700">
                    {loading ? (
                        <div className="py-4 text-center text-slate-400">Loading...</div>
                    ) : stocks.length === 0 ? (
                        <div className="py-4 text-center text-slate-400">
                            No tickers in this category
                            <button
                                onClick={() => onAddTicker(category.id)}
                                className="ml-2 text-blue-400 hover:underline"
                            >
                                Add one
                            </button>
                        </div>
                    ) : (
                        <div className="flex flex-col gap-2 mt-3">
                            {stocks.map((stock) => (
                                <div key={stock.ticker} className="relative group">
                                    <StockCard
                                        stock={stock}
                                        selectedDate={selectedDate}
                                        onMemoClick={onMemoClick}
                                        hasMemo={memoTickers.includes(stock.ticker)}
                                    />
                                    <button
                                        onClick={() => onRemoveTicker(category.id, stock.ticker)}
                                        className="absolute top-2 right-2 p-1 bg-red-500/80 rounded opacity-0 group-hover:opacity-100 transition-opacity text-xs"
                                        title="Remove ticker"
                                    >
                                        ✕
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
