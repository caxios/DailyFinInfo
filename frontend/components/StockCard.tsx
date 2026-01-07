'use client';

import { Stock } from '@/types';
import Sparkline from './Sparkline';

interface StockCardProps {
    stock: Stock;
}

export default function StockCard({ stock }: StockCardProps) {
    const isPositive = stock.change_percent >= 0;

    return (
        <div className={`
      bg-slate-800/70 border border-slate-700 rounded-xl p-4
      grid grid-cols-[1fr_120px_100px] items-center gap-4
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

            {/* Price & Change */}
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
        </div>
    );
}
