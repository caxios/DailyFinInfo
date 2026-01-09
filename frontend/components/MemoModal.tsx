'use client';

import { useState, useEffect } from 'react';
import { Memo } from '@/types';

interface MemoModalProps {
    ticker: string;
    companyName: string;
    date: string;
    onClose: () => void;
    onSave: () => void;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

export default function MemoModal({
    ticker,
    companyName,
    date,
    onClose,
    onSave
}: MemoModalProps) {
    const [content, setContent] = useState('');
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        fetchMemo();
    }, [ticker, date]);

    async function fetchMemo() {
        setLoading(true);
        try {
            const response = await fetch(`${API_URL}/api/memos/${ticker}/${date}`);
            const data: Memo = await response.json();
            setContent(data.content || '');
        } catch (err) {
            console.error('Failed to load memo:', err);
        } finally {
            setLoading(false);
        }
    }

    async function handleSave() {
        setSaving(true);
        try {
            await fetch(`${API_URL}/api/memos`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ticker, date, content })
            });
            onSave();
            onClose();
        } catch (err) {
            console.error('Failed to save memo:', err);
        } finally {
            setSaving(false);
        }
    }

    return (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
            <div className="bg-slate-800 rounded-xl w-full max-w-lg max-h-[80vh] flex flex-col">
                {/* Header */}
                <div className="px-4 py-3 border-b border-slate-700 flex items-center justify-between">
                    <div>
                        <h2 className="text-white font-semibold">{ticker}</h2>
                        <p className="text-sm text-slate-400">{companyName} • {date}</p>
                    </div>
                    <button
                        onClick={onClose}
                        className="text-slate-400 hover:text-white text-xl"
                    >
                        ✕
                    </button>
                </div>

                {/* Content */}
                <div className="flex-1 p-4 overflow-auto">
                    {loading ? (
                        <div className="text-center py-8 text-slate-400">Loading...</div>
                    ) : (
                        <textarea
                            value={content}
                            onChange={(e) => setContent(e.target.value)}
                            placeholder="Write your notes about this stock..."
                            className="w-full h-48 bg-slate-700 border border-slate-600 rounded-lg p-3 text-white resize-none focus:outline-none focus:border-blue-500"
                            autoFocus
                        />
                    )}
                </div>

                {/* Footer */}
                <div className="px-4 py-3 border-t border-slate-700 flex justify-end gap-2">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-500"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleSave}
                        disabled={saving}
                        className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
                    >
                        {saving ? 'Saving...' : 'Save'}
                    </button>
                </div>
            </div>
        </div>
    );
}
