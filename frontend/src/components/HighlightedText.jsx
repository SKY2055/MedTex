import React from 'react';
import { entityColors } from '../utils/colors';

const HighlightedText = ({ text, entities }) => {
    if (!entities.length) return <p className="text-slate-600">{text}</p>;

    let lastIndex = 0;
    const parts = [];

    entities.forEach((ent, i) => {
        // Add text before the entity
        parts.push(text.substring(lastIndex, ent.start));

        // Add the highlighted entity
        parts.push(
            <span
                key={i}
                className={`${entityColors[ent.label] || 'bg-slate-100'} px-1 rounded font-medium mx-0.5 group relative cursor-help`}
            >
                {ent.text}
                <span className="absolute -top-8 left-0 bg-slate-800 text-white text-[10px] px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                    {ent.label}
                </span>
            </span>
        );
        lastIndex = ent.end;
    });

    // Add remaining text
    parts.push(text.substring(lastIndex));

    return (
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 leading-relaxed">
            <h2 className="text-lg font-semibold text-slate-800 mb-4">Visualized Entities</h2>
            <div className="text-slate-700 whitespace-pre-wrap">{parts}</div>
        </div>
    );
};

export default HighlightedText;