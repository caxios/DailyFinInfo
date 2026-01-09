'use client';

import { useState, useEffect } from 'react';
import { Memo } from '@/types';

interface NotebookModalProps {
    onClose: () => void;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

export default function NotebookModal({ onClose }: NotebookModalProps) {
    const [memos, setMemos] = useState<Memo[]>([]);
    const [loading, setLoading] = useState(true);
    const [filter, setFilter] = useState('');

    useEffect(() => {
        fetchMemos();
    }, []);

    async function fetchMemos() {
        setLoading(true);
        try {
            const response = await fetch(`${API_URL}/api/memos`);
            const data = await response.json();
            // Sort by date (newest first), then by ticker
            const sorted = (data.memos || []).sort((a: Memo, b: Memo) => {
                if (a.date !== b.date) return b.date.localeCompare(a.date);
                return a.ticker.localeCompare(b.ticker);
            });
            setMemos(sorted);
        } catch (err) {
            console.error('Failed to load memos:', err);
        } finally {
            setLoading(false);
        }
    }

    // Group memos by date
    const groupedMemos = memos
        .filter(m =>
            m.ticker.toLowerCase().includes(filter.toLowerCase()) ||
            m.content.toLowerCase().includes(filter.toLowerCase())
        )
        .reduce((groups, memo) => {
            const date = memo.date;
            if (!groups[date]) groups[date] = [];
            groups[date].push(memo);
            return groups;
        }, {} as Record<string, Memo[]>);

    return (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
            <div className="bg-slate-800 rounded-xl w-full max-w-2xl max-h-[85vh] flex flex-col">
                {/* Header */}
                <div className="px-4 py-3 border-b border-slate-700 flex items-center justify-between">
                    <h2 className="text-white font-semibold text-lg">📓 My Notebook</h2>
                    <button
                        onClick={onClose}
                        className="text-slate-400 hover:text-white text-xl"
                    >
                        ✕
                    </button>
                </div>

                {/* Search */}
                <div className="px-4 py-2 border-b border-slate-700">
                    <input
                        type="text"
                        placeholder="Search memos..."
                        value={filter}
                        onChange={(e) => setFilter(e.target.value)}
                        className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-blue-500"
                    />
                </div>

                {/* Content */}
                <div className="flex-1 overflow-auto p-4">
                    {loading ? (
                        <div className="text-center py-8 text-slate-400">Loading memos...</div>
                    ) : memos.length === 0 ? (
                        <div className="text-center py-8 text-slate-400">
                            <p className="text-4xl mb-2">📝</p>
                            <p>No memos yet</p>
                            <p className="text-sm mt-1">Click the memo icon on any stock to add notes</p>
                        </div>
                    ) : Object.keys(groupedMemos).length === 0 ? (
                        <div className="text-center py-8 text-slate-400">
                            No memos match your search
                        </div>
                    ) : (
                        <div className="space-y-6">
                            {Object.entries(groupedMemos).map(([date, dateMemos]) => (
                                <div key={date}>
                                    <h3 className="text-sm text-slate-400 mb-2 sticky top-0 bg-slate-800 py-1">
                                        📅 {new Date(date + 'T00:00:00').toLocaleDateString('en-US', {
                                            weekday: 'long',
                                            year: 'numeric',
                                            month: 'long',
                                            day: 'numeric'
                                        })}
                                    </h3>
                                    <div className="space-y-2">
                                        {dateMemos.map((memo) => (
                                            <div
                                                key={memo.id}
                                                className="bg-slate-700/50 rounded-lg p-3"
                                            >
                                                <div className="flex items-center justify-between mb-2">
                                                    <span className="text-blue-400 font-medium">{memo.ticker}</span>
                                                    <span className="text-xs text-slate-500">
                                                        {memo.updated_at && new Date(memo.updated_at).toLocaleTimeString()}
                                                    </span>
                                                </div>
                                                <p className="text-slate-300 text-sm whitespace-pre-wrap">
                                                    {memo.content}
                                                </p>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
