'use client';

interface DateNavProps {
    selectedDate: string;  // YYYY-MM-DD format
    onDateChange: (date: string) => void;
}

export default function DateNav({
    selectedDate,
    onDateChange,
}: DateNavProps) {
    const today = new Date();
    const maxDate = today.toISOString().split('T')[0];

    // Min date: 60 days ago
    const minDate = new Date();
    minDate.setDate(today.getDate() - 60);
    const minDateStr = minDate.toISOString().split('T')[0];

    // Format selected date for display
    const displayDate = new Date(selectedDate + 'T00:00:00');
    const options: Intl.DateTimeFormatOptions = {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    };
    const dateString = displayDate.toLocaleDateString('en-US', options);

    return (
        <nav className="bg-slate-800/50 px-4 py-3 rounded-xl">
            <div className="flex items-center justify-center gap-4">
                <span className="text-slate-400">📅</span>
                <input
                    type="date"
                    value={selectedDate}
                    max={maxDate}
                    min={minDateStr}
                    onChange={(e) => onDateChange(e.target.value)}
                    className="
            bg-slate-700 border border-slate-600 text-white
            px-4 py-2 rounded-lg
            focus:outline-none focus:border-blue-500
            cursor-pointer
          "
                />
            </div>
            <div className="text-center mt-2 text-slate-300 font-medium">
                {dateString}
            </div>
        </nav>
    );
}
