'use client';

import { Stock } from '@/types';
import Sparkline from './Sparkline';

interface StockCardProps {
    stock: Stock;
    selectedDate?: string;
    onMemoClick?: (ticker: string, companyName: string) => void;
    hasMemo?: boolean;
}

export default function StockCard({
    stock,
    selectedDate,
    onMemoClick,
    hasMemo = false
}: StockCardProps) {
    const isPositive = stock.change_percent >= 0;

    return (
        <div className={`
      bg-slate-800/70 border border-slate-700 rounded-xl p-4
      grid grid-cols-[1fr_120px_auto] items-center gap-4
      transition-all duration-200 hover:border-blue-500 hover:-translate-y-0.5
      ${stock.error ? 'opacity-50' : ''}
    `}>
            {/* Stock Info */}
            <div className="flex flex-col gap-1">
                <div className="font-semibold text-white">{stock.ticker}</div>
                <div className="text-xs text-slate-400 truncate max-w-[150px]">
                    {stock.company_name}
                </div>
            </div>

            {/* Sparkline */}
            <div className="h-10">
                <Sparkline data={stock.returns} isPositive={isPositive} />
            </div>

            {/* Price, Change & Memo */}
            <div className="flex items-center gap-3">
                <div className="text-right flex flex-col gap-1">
                    <div className="font-semibold text-white">
                        ${stock.current_price.toFixed(2)}
                    </div>
                    <span className={`
          inline-block px-2 py-0.5 rounded text-xs font-semibold
          ${isPositive
                            ? 'bg-green-500/15 text-green-400'
                            : 'bg-red-500/15 text-red-400'
                        }
        `}>
                        {isPositive ? '+' : ''}{stock.change_percent.toFixed(2)}%
                    </span>
                </div>

                {/* Memo Button */}
                {onMemoClick && (
                    <button
                        onClick={(e) => {
                            e.stopPropagation();
                            onMemoClick(stock.ticker, stock.company_name);
                        }}
                        className={`
                            p-1.5 rounded-lg transition-all
                            ${hasMemo
                                ? 'text-yellow-400 hover:bg-yellow-500/20'
                                : 'text-slate-500 hover:text-slate-300 hover:bg-slate-600/50'
                            }
                        `}
                        title={hasMemo ? 'Edit memo' : 'Add memo'}
                    >
                        📝
                    </button>
                )}
            </div>
        </div>
    );
}
