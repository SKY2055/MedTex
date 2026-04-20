import React, { useState } from 'react';
import NoteInput from './NoteInput';
import HighlightedText from './HighlightedText';
import EntityTable from './EntityTable';

const Dashboard = () => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleProcess = async (text) => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch('http://localhost:8080/api/v1/extract', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            setData(result);
        } catch (error) {
            console.error("Error fetching NER:", error);
            setError(error.message || "Failed to connect to MedTex Backend");
        } finally {
            setLoading(false);
        }
    };

    const handleUpload = async (file) => {
        setLoading(true);
        setError(null);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:8080/api/v1/upload-prescription', {
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
                <header className="mb-8">
                    <h1 className="text-3xl font-bold text-sky-900">MedTex <span className="text-sky-500 font-light">Clinical NER</span></h1>
                    <p className="text-slate-500">Extract medication, anatomy & chemical entities from prescriptions, PDFs, or images</p>
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
                                <HighlightedText text={data.original_text} entities={data.entities} />
                                <EntityTable entities={data.entities} />
                            </>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;