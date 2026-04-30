import React, { useState } from 'react';
import NoteInput from './NoteInput';
import HighlightedText from './HighlightedText';
import EntityTable from './EntityTable';
import MedicationSummary from './MedicationSummary';
import { useAuth } from '../context/AuthContext';
import { Stethoscope, CheckCircle, XCircle, LogOut, User } from 'lucide-react';

const Dashboard = () => {
    const { user, logout, authFetch } = useAuth();
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [activeTab, setActiveTab] = useState('visualized');
    const [extractionId, setExtractionId] = useState(null); // Track extraction ID for verification
    const [toast, setToast] = useState(null); // Toast notification state

    const handleProcess = async (text) => {
        setLoading(true);
        setError(null);
        try {
            const response = await authFetch('/api/v1/extract', {
                method: 'POST',
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            setData(result);
            // Use actual extraction_id from backend response
            setExtractionId(result.extraction_id || null);
        } catch (error) {
            console.error("Error fetching NER:", error);
            setError(error.message || "Failed to connect to MedTex Backend");
        } finally {
            setLoading(false);
        }
    };

    const handleVerify = async (verifiedMedications) => {
        if (!extractionId) {
            console.warn("No extraction ID available for verification");
            setToast({ type: 'error', message: 'No extraction ID available' });
            return;
        }

        try {
            const response = await authFetch(`/api/v1/extractions/${extractionId}/verify`, {
                method: 'PUT',
                body: JSON.stringify({
                    verified_json: {
                        medications: verifiedMedications,
                        standardized_text: data?.standardized_text,
                        original_text: data?.original_text
                    },
                    status: 'verified'
                })
            });

            if (!response.ok) {
                throw new Error(`Verification failed: ${response.status}`);
            }

            const result = await response.json();
            console.log("Verification successful:", result);
            setToast({ type: 'success', message: 'Verification saved successfully!' });
        } catch (error) {
            console.error("Verification error:", error);
            setToast({ type: 'error', message: 'Verification failed. Please try again.' });
        }
    };

    const handleUpload = async (file) => {
        setLoading(true);
        setError(null);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await authFetch('/api/v1/upload-prescription', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || `Upload failed: ${response.status}`);
            }

            const result = await response.json();
            setData(result);
        } catch (error) {
            console.error("Upload failed:", error);
            setError(error.message || "Failed to upload file");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 p-8">
            <div className="max-w-6xl mx-auto">
                <header className="mb-8 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="bg-teal-600 p-2 rounded-lg">
                            <Stethoscope className="w-6 h-6 text-white" />
                        </div>
                        <div>
                            <h1 className="text-3xl font-bold text-sky-900">MedTex <span className="text-sky-500 font-light">Clinical NER</span></h1>
                            <p className="text-slate-500">Advanced clinical entity extraction powered by multi-model NLP ensemble</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-3">
                        <div className="flex items-center gap-2 px-3 py-1 bg-slate-100 rounded-lg">
                            <User className="w-4 h-4 text-slate-600" />
                            <span className="text-sm text-slate-700">{user?.username} ({user?.role})</span>
                        </div>
                        <button
                            onClick={logout}
                            className="flex items-center gap-2 px-3 py-1 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                        >
                            <LogOut className="w-4 h-4" />
                            <span className="text-sm">Logout</span>
                        </button>
                    </div>
                </header>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <NoteInput
                        onProcess={handleProcess}
                        onUpload={handleUpload}
                        loading={loading}
                    />

                    <div className="space-y-8">
                        {error && (
                            <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-xl">
                                <p className="font-medium">Error: {error}</p>
                            </div>
                        )}

                        {data && (
                            <>
                                {/* Tab Navigation */}
                                <div className="flex gap-2 mb-6">
                                    <button
                                        onClick={() => setActiveTab('visualized')}
                                        className={`px-4 py-2 rounded-lg font-medium transition-colors ${activeTab === 'visualized'
                                            ? 'bg-teal-600 text-white'
                                            : 'bg-white text-slate-600 hover:bg-slate-100 border border-slate-200'
                                            }`}
                                    >
                                        Visualized Text
                                    </button>
                                    {data.medications && data.medications.length > 0 && (
                                        <button
                                            onClick={() => setActiveTab('summary')}
                                            className={`px-4 py-2 rounded-lg font-medium transition-colors ${activeTab === 'summary'
                                                ? 'bg-teal-600 text-white'
                                                : 'bg-white text-slate-600 hover:bg-slate-100 border border-slate-200'
                                                }`}
                                        >
                                            Clinical Summary
                                        </button>
                                    )}
                                </div>

                                {/* Tab Content */}
                                {activeTab === 'visualized' && (
                                    <>
                                        <HighlightedText text={data.original_text} entities={data.entities} />
                                        <EntityTable entities={data.entities} />
                                    </>
                                )}

                                {activeTab === 'summary' && data.medications && data.medications.length > 0 && (
                                    <MedicationSummary
                                        medications={data.medications}
                                        standardizedText={data.standardized_text}
                                        originalText={data.original_text}
                                        onVerify={handleVerify}
                                    />
                                )}
                            </>
                        )}
                    </div>
                </div>
            </div>

            {/* Toast Notification */}
            {toast && (
                <div className={`fixed bottom-6 right-6 flex items-center gap-3 px-4 py-3 rounded-lg shadow-lg transition-all duration-300 ${toast.type === 'success' ? 'bg-emerald-500 text-white' : 'bg-red-500 text-white'
                    }`}>
                    {toast.type === 'success' ? <CheckCircle className="w-5 h-5" /> : <XCircle className="w-5 h-5" />}
                    <span className="font-medium">{toast.message}</span>
                    <button
                        onClick={() => setToast(null)}
                        className="ml-2 hover:opacity-80"
                    >
                        <XCircle className="w-4 h-4" />
                    </button>
                </div>
            )}
        </div>
    );
};

export default Dashboard;