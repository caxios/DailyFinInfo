'use client';

interface DateNavProps {
    dateOffset: number;
    onPrev: () => void;
    onNext: () => void;
    maxOffset?: number;
}

export default function DateNav({
    dateOffset,
    onPrev,
    onNext,
    maxOffset = 30
}: DateNavProps) {
    const date = new Date();
    date.setDate(date.getDate() - dateOffset);

    const options: Intl.DateTimeFormatOptions = {
        weekday: 'short',
        month: 'short',
        day: 'numeric'
    };
    const dateString = date.toLocaleDateString('en-US', options);

    let label = 'Today';
    if (dateOffset === 1) label = 'Yesterday';
    else if (dateOffset > 1) label = `${dateOffset} days ago`;

    return (
        <nav className="flex items-center gap-4 bg-slate-800/50 px-4 py-2 rounded-xl mb-6">
            <button
                onClick={onPrev}
                disabled={dateOffset >= maxOffset}
                className="
          bg-slate-700 border border-slate-600 text-white
          w-9 h-9 rounded-full flex items-center justify-center
          transition-all hover:bg-blue-500 hover:border-blue-500
          disabled:opacity-30 disabled:cursor-not-allowed
        "
            >
                ←
            </button>

            <div className="flex-1 text-center">
                <div className="text-xs text-slate-400">{label}</div>
                <div className="font-medium text-white">{dateString}</div>
            </div>

            <button
                onClick={onNext}
                disabled={dateOffset === 0}
                className="
          bg-slate-700 border border-slate-600 text-white
          w-9 h-9 rounded-full flex items-center justify-center
          transition-all hover:bg-blue-500 hover:border-blue-500
          disabled:opacity-30 disabled:cursor-not-allowed
        "
            >
                →
            </button>
        </nav>
    );
}
