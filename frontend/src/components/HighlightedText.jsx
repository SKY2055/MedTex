import React from 'react';
import { entityColors, entityBorderColors } from '../utils/colors';
import { Highlighter } from 'lucide-react';

const HighlightedText = ({ text, entities }) => {
    if (!entities.length) return (
        <div className="bg-white p-8 rounded-xl border border-slate-200">
            <div className="w-12 h-12 bg-slate-50 rounded-full flex items-center justify-center mb-3">
                <Highlighter className="w-5 h-5 text-slate-400" />
            </div>
            <p className="text-sm font-medium text-slate-500">No entities to visualize.</p>
        </div>
    );

    let lastIndex = 0;
    const parts = [];

    entities.forEach((ent, i) => {
        parts.push(text.substring(lastIndex, ent.start));

        parts.push(
            <span
                key={i}
                className={`${entityColors[ent.label] || 'bg-slate-100 text-slate-700'} ${entityBorderColors[ent.label] || 'border border-slate-200'} px-1.5 py-0.5 rounded-md text-sm font-semibold mx-0.5 group relative cursor-default inline-block transition-all duration-150 hover:shadow-sm hover:-translate-y-px`}
            >
                {ent.text}
                <span className="absolute -top-8 left-1/2 -translate-x-1/2 bg-slate-800 text-white text-[10px] font-semibold px-2.5 py-1 rounded-md opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-20 shadow-lg pointer-events-none">
                    {ent.label}
                    <span className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-slate-800"></span>
                </span>
            </span>
        );
        lastIndex = ent.end;
    });

    parts.push(text.substring(lastIndex));

    return (
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm hover-lift">
            <div className="px-5 py-3.5 border-b border-slate-100 bg-slate-50/80 flex items-center gap-2">
                <Highlighter className="w-4 h-4 text-slate-500" />
                <h2 className="text-sm font-bold text-slate-700 uppercase tracking-wider">Visualized Text</h2>
            </div>
            <div className="p-5 leading-relaxed text-slate-600 text-sm whitespace-pre-wrap font-medium">
                {parts}
            </div>
            <div className="px-5 py-2.5 bg-slate-50 border-t border-slate-100">
                <div className="flex flex-wrap gap-1.5">
                    {entities.map((ent, i) => (
                        <span
                            key={i}
                            className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${entityColors[ent.label] || 'bg-slate-100 text-slate-600'} ${entityBorderColors[ent.label] || 'border border-slate-200'}`}
                        >
                            {ent.label}
                        </span>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default HighlightedText;