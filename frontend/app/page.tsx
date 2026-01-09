'use client';

import { useState } from 'react';
import DateNav from '@/components/DateNav';
import CategoryList from '@/components/CategoryList';
import NotebookModal from '@/components/NotebookModal';

// Get today's date in YYYY-MM-DD format
function getTodayString() {
  return new Date().toISOString().split('T')[0];
}

export default function Home() {
  const [selectedDate, setSelectedDate] = useState(getTodayString());
  const [showNotebook, setShowNotebook] = useState(false);

  return (
    <main className="min-h-screen bg-slate-950 text-white p-4">
      <div className="max-w-xl mx-auto">
        {/* Header */}
        <header className="flex justify-between items-center py-4 mb-4 border-b border-slate-800">
          <h1 className="text-xl font-semibold">📈 Intraday Tracker</h1>
        </header>

        {/* Date Picker with Notebook Icon */}
        <div className="flex items-center gap-3 mb-6">
          <div className="flex-1">
            <DateNav
              selectedDate={selectedDate}
              onDateChange={setSelectedDate}
            />
          </div>
          <button
            onClick={() => setShowNotebook(true)}
            className="p-3 bg-slate-800/50 rounded-xl hover:bg-slate-700/50 transition-all text-2xl"
            title="Open Notebook"
          >
            📓
          </button>
        </div>

        {/* Watchlist Section Header */}
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm text-slate-400 uppercase tracking-wide">
            Portfolios & Watchlists
          </h2>
        </div>

        {/* Category List */}
        <CategoryList selectedDate={selectedDate} />
      </div>

      {/* Notebook Modal */}
      {showNotebook && (
        <NotebookModal onClose={() => setShowNotebook(false)} />
      )}
    </main>
  );
}
