import React, { useState } from 'react';

const NoteInput = ({ onProcess, onUpload, loading }) => {
    const [text, setText] = useState("");

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file) onUpload(file);
    };

    return (
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <h2 className="text-lg font-semibold text-slate-800 mb-4">Input Prescription</h2>

            <textarea
                className="w-full h-32 p-4 border border-slate-300 rounded-lg mb-4"
                placeholder="Paste notes here..."
                value={text}
                onChange={(e) => setText(e.target.value)}
            />

            <div className="flex gap-4">
                <button
                    onClick={() => onProcess(text)}
                    className="flex-1 bg-sky-600 text-white py-2 rounded-lg"
                >
                    Analyze Text
                </button>

                <label className="flex-1 border-2 border-dashed border-sky-200 hover:border-sky-400 flex items-center justify-center rounded-lg cursor-pointer transition-colors">
                    <span className="text-sky-600 text-sm font-medium">Upload PDF or Image</span>
                    <input type="file" className="hidden" onChange={handleFileChange} accept=".pdf,.txt,.png,.jpg,.jpeg,.gif,.bmp,.tiff" />
                </label>
            </div>
        </div>
    );
};

export default NoteInput;