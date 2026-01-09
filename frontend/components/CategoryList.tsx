'use client';

import { useState, useEffect } from 'react';
import { Category, Memo } from '@/types';
import CategoryItem from './CategoryItem';
import MemoModal from './MemoModal';

interface CategoryListProps {
    selectedDate: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

export default function CategoryList({ selectedDate }: CategoryListProps) {
    const [categories, setCategories] = useState<Category[]>([]);
    const [loading, setLoading] = useState(true);
    const [showAddCategory, setShowAddCategory] = useState(false);
    const [newCategoryName, setNewCategoryName] = useState('');
    const [showAddTicker, setShowAddTicker] = useState<string | null>(null);
    const [newTicker, setNewTicker] = useState('');

    // Memo state
    const [memos, setMemos] = useState<Memo[]>([]);
    const [memoModal, setMemoModal] = useState<{ ticker: string, companyName: string } | null>(null);

    useEffect(() => {
        fetchCategories();
        fetchMemos();
    }, []);

    useEffect(() => {
        fetchMemos();
    }, [selectedDate]);

    async function fetchCategories() {
        setLoading(true);
        try {
            const response = await fetch(`${API_URL}/api/categories`);
            const data = await response.json();
            setCategories(data.categories || []);
        } catch (err) {
            console.error('Failed to load categories:', err);
        } finally {
            setLoading(false);
        }
    }

    async function fetchMemos() {
        try {
            const response = await fetch(`${API_URL}/api/memos`);
            const data = await response.json();
            setMemos(data.memos || []);
        } catch (err) {
            console.error('Failed to load memos:', err);
        }
    }

    async function createCategory() {
        if (!newCategoryName.trim()) return;

        try {
            const response = await fetch(`${API_URL}/api/categories`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: newCategoryName.trim() })
            });

            if (response.ok) {
                setNewCategoryName('');
                setShowAddCategory(false);
                fetchCategories();
            }
        } catch (err) {
            console.error('Failed to create category:', err);
        }
    }

    async function deleteCategory(categoryId: string) {
        console.log('Deleting category:', categoryId);
        try {
            const response = await fetch(`${API_URL}/api/categories/${categoryId}`, {
                method: 'DELETE'
            });
            console.log('Delete response:', response.status);
            if (response.ok) {
                fetchCategories();
            }
        } catch (err) {
            console.error('Failed to delete category:', err);
        }
    }

    async function addTicker(categoryId: string) {
        if (!newTicker.trim()) return;

        try {
            const response = await fetch(`${API_URL}/api/categories/${categoryId}/tickers`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ticker: newTicker.trim() })
            });

            if (response.ok) {
                setNewTicker('');
                setShowAddTicker(null);
                fetchCategories();
            }
        } catch (err) {
            console.error('Failed to add ticker:', err);
        }
    }

    async function removeTicker(categoryId: string, ticker: string) {
        try {
            await fetch(`${API_URL}/api/categories/${categoryId}/tickers/${ticker}`, {
                method: 'DELETE'
            });
            fetchCategories();
        } catch (err) {
            console.error('Failed to remove ticker:', err);
        }
    }

    if (loading) {
        return (
            <div className="text-center py-12 text-slate-400">
                <div className="w-10 h-10 border-3 border-slate-600 border-t-blue-500 rounded-full animate-spin mx-auto mb-4" />
                <p>Loading categories...</p>
            </div>
        );
    }

    return (
        <div>
            {/* Category List */}
            {categories.map((category) => (
                <div key={category.id}>
                    <CategoryItem
                        category={category}
                        selectedDate={selectedDate}
                        onAddTicker={(id) => setShowAddTicker(id)}
                        onDeleteCategory={deleteCategory}
                        onRemoveTicker={removeTicker}
                        onMemoClick={(ticker, companyName) => setMemoModal({ ticker, companyName })}
                        memoTickers={memos.filter(m => m.date === selectedDate).map(m => m.ticker)}
                    />

                    {/* Add Ticker Modal */}
                    {showAddTicker === category.id && (
                        <div className="bg-slate-700/50 rounded-lg p-3 mb-2 flex gap-2">
                            <input
                                type="text"
                                placeholder="Enter ticker (e.g., AAPL)"
                                value={newTicker}
                                onChange={(e) => setNewTicker(e.target.value.toUpperCase())}
                                onKeyDown={(e) => e.key === 'Enter' && addTicker(category.id)}
                                className="flex-1 bg-slate-800 border border-slate-600 rounded px-3 py-2 text-white"
                                autoFocus
                            />
                            <button
                                onClick={() => addTicker(category.id)}
                                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                            >
                                Add
                            </button>
                            <button
                                onClick={() => setShowAddTicker(null)}
                                className="px-4 py-2 bg-slate-600 text-white rounded hover:bg-slate-500"
                            >
                                Cancel
                            </button>
                        </div>
                    )}
                </div>
            ))}

            {/* Add Category Section */}
            {showAddCategory ? (
                <div className="bg-slate-700/50 rounded-xl p-3 flex gap-2">
                    <input
                        type="text"
                        placeholder="Category name"
                        value={newCategoryName}
                        onChange={(e) => setNewCategoryName(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && createCategory()}
                        className="flex-1 bg-slate-800 border border-slate-600 rounded px-3 py-2 text-white"
                        autoFocus
                    />
                    <button
                        onClick={createCategory}
                        className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
                    >
                        Create
                    </button>
                    <button
                        onClick={() => setShowAddCategory(false)}
                        className="px-4 py-2 bg-slate-600 text-white rounded hover:bg-slate-500"
                    >
                        Cancel
                    </button>
                </div>
            ) : (
                <button
                    onClick={() => setShowAddCategory(true)}
                    className="w-full py-3 border-2 border-dashed border-slate-600 rounded-xl text-slate-400 hover:border-blue-500 hover:text-blue-400 transition-all"
                >
                    + Add Category
                </button>
            )}

            {/* Memo Modal */}
            {memoModal && (
                <MemoModal
                    ticker={memoModal.ticker}
                    companyName={memoModal.companyName}
                    date={selectedDate}
                    onClose={() => setMemoModal(null)}
                    onSave={() => fetchMemos()}
                />
            )}
        </div>
    );
}
