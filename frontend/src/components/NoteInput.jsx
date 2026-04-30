import React, { useState } from 'react';
import { Upload, Loader2, FileText, Stethoscope, Image, ScanLine } from 'lucide-react';

const NoteInput = ({ onProcess, onUpload, loading, uploadProgress = 0, processingStatus = "" }) => {
    const [text, setText] = useState("");
    const [isDragging, setIsDragging] = useState(false);

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file) onUpload(file);
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => {
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        if (file) onUpload(file);
    };

    return (
        <div className="space-y-4">
            {/* Text Input Card */}
            <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm hover-lift">
                <div className="px-5 py-3.5 border-b border-slate-100 bg-slate-50/80 flex items-center gap-2">
                    <FileText className="w-4 h-4 text-slate-500" />
                    <h2 className="text-sm font-bold text-slate-700 uppercase tracking-wider">Text Input</h2>
                </div>
                <div className="p-5">
                    <textarea
                        className="w-full h-32 p-4 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-700 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-400 transition-all resize-none font-medium"
                        placeholder="Paste clinical notes or prescription text here..."
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        disabled={loading}
                    />
                    <button
                        onClick={() => onProcess(text)}
                        className="w-full mt-3 bg-teal-600 hover:bg-teal-700 text-white py-2.5 rounded-lg flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-semibold transition-colors shadow-sm"
                        disabled={loading}
                    >
                        {loading && uploadProgress === 0 ? (
                            <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                Processing...
                            </>
                        ) : (
                            <>
                                <Stethoscope className="w-4 h-4" />
                                Analyze Text
                            </>
                        )}
                    </button>
                </div>
            </div>

            {/* Upload Card */}
            <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm hover-lift">
                <div className="px-5 py-3.5 border-b border-slate-100 bg-slate-50/80 flex items-center gap-2">
                    <ScanLine className="w-4 h-4 text-slate-500" />
                    <h2 className="text-sm font-bold text-slate-700 uppercase tracking-wider">Document Upload</h2>
                </div>
                <div className="p-5">
                    <label
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        className={`flex flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed p-6 cursor-pointer transition-all ${loading && uploadProgress > 0
                            ? 'border-slate-200 bg-slate-50 cursor-not-allowed opacity-50'
                            : isDragging
                                ? 'border-teal-400 bg-teal-50/50'
                                : 'border-slate-200 hover:border-teal-400 hover:bg-slate-50'
                            }`}
                    >
                        {loading && uploadProgress > 0 ? (
                            <div className="flex flex-col items-center gap-3 w-full">
                                <Loader2 className="w-6 h-6 text-teal-600 animate-spin" />
                                <div className="flex flex-col items-center gap-1.5">
                                    <span className="text-sm font-semibold text-slate-600">
                                        {uploadProgress < 100 ? `Uploading... ${uploadProgress}%` : processingStatus}
                                    </span>
                                    <div className="w-48 bg-slate-200 rounded-full h-1.5 overflow-hidden">
                                        <div
                                            className="bg-teal-600 h-1.5 rounded-full transition-all duration-300"
                                            style={{ width: `${uploadProgress}%` }}
                                        />
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <>
                                <div className="w-12 h-12 bg-teal-50 rounded-full flex items-center justify-center">
                                    {isDragging ? (
                                        <Image className="w-5 h-5 text-teal-600" />
                                    ) : (
                                        <Upload className="w-5 h-5 text-teal-600" />
                                    )}
                                </div>
                                <div className="text-center">
                                    <p className="text-sm font-semibold text-slate-700">
                                        {isDragging ? 'Drop to upload' : 'Click or drag to upload'}
                                    </p>
                                    <p className="text-xs text-slate-400 mt-1">PDF, PNG, JPG, JPEG, TIFF, BMP</p>
                                </div>
                            </>
                        )}
                        <input
                            type="file"
                            className="hidden"
                            onChange={handleFileChange}
                            accept=".pdf,.txt,.png,.jpg,.jpeg,.gif,.bmp,.tiff"
                            disabled={loading}
                        />
                    </label>
                </div>
            </div>
        </div>
    );
};

export default NoteInput;