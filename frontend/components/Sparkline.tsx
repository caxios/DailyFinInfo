'use client';

import { Stock } from '@/types';

interface SparklineProps {
    data: number[];
    isPositive: boolean;
    width?: number;
    height?: number;
}

export default function Sparkline({
    data,
    isPositive,
    width = 120,
    height = 40
}: SparklineProps) {
    if (!data || data.length === 0) {
        return <svg className="w-full h-full" />;
    }

    const padding = 4;
    const min = Math.min(...data, 0);
    const max = Math.max(...data, 0);
    const range = max - min || 1;

    const points = data.map((val, i) => {
        const x = padding + (i / (data.length - 1)) * (width - 2 * padding);
        const y = height - padding - ((val - min) / range) * (height - 2 * padding);
        return `${x},${y}`;
    }).join(' ');

    const zeroY = height - padding - ((0 - min) / range) * (height - 2 * padding);
    const color = isPositive ? '#22c55e' : '#ef4444';

    return (
        <svg
            viewBox={`0 0 ${width} ${height}`}
            className="w-full h-full"
        >
            {/* Zero line */}
            <line
                x1={padding}
                y1={zeroY}
                x2={width - padding}
                y2={zeroY}
                stroke="#4b5563"
                strokeWidth="1"
                strokeDasharray="2,2"
            />
            {/* Data line */}
            <polyline
                points={points}
                fill="none"
                stroke={color}
                strokeWidth="1.5"
            />
        </svg>
    );
}
