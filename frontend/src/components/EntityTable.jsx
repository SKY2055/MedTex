import React from 'react';
import { entityColors, entityBorderColors } from '../utils/colors';
import { Tag, Hash, FileSearch } from 'lucide-react';

// Map entity labels to user-friendly display names
const labelDisplayMap = {
    'FREQUENCY_LATIN': 'Frequency',
    'FORM': 'Dosage Form',
    'STRENGTH': 'Strength',
    'DURATION': 'Duration',
    'ROUTE': 'Route',
    'DRUG': 'Drug',
    'DOSAGE': 'Dosage',
    'DISEASE': 'Disease',
    'SYMPTOM': 'Symptom',
    'ANATOMY': 'Anatomy',
    'PROCEDURE': 'Procedure',
    'LAB': 'Lab',
    'LATIN_INSTRUCTION': 'Instruction',
    'UNCERTAIN': 'Uncertain'
};

const EntityTable = ({ entities }) => {
    if (!entities || entities.length === 0) {
        return (
            <div className="bg-white p-8 rounded-xl border border-slate-200 text-center">
                <div className="w-12 h-12 bg-slate-50 rounded-full flex items-center justify-center mx-auto mb-3">
                    <FileSearch className="w-5 h-5 text-slate-400" />
                </div>
                <p className="text-sm font-medium text-slate-500">No clinical entities detected.</p>
            </div>
        );
    }

    return (
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm hover-lift">
            <div className="px-5 py-3.5 border-b border-slate-100 bg-slate-50/80 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Tag className="w-4 h-4 text-slate-500" />
                    <h2 className="text-sm font-bold text-slate-700 uppercase tracking-wider">Extracted Entities</h2>
                </div>
                <span className="text-xs font-semibold text-slate-400 bg-slate-100 px-2 py-0.5 rounded-full">
                    {entities.length}
                </span>
            </div>
            <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-slate-100">
                    <thead className="bg-slate-50/50">
                        <tr>
                            <th className="px-5 py-2.5 text-left text-[11px] font-bold text-slate-500 uppercase tracking-widest">
                                Entity Text
                            </th>
                            <th className="px-5 py-2.5 text-left text-[11px] font-bold text-slate-500 uppercase tracking-widest">
                                Category
                            </th>
                            <th className="px-5 py-2.5 text-left text-[11px] font-bold text-slate-500 uppercase tracking-widest">
                                Offset
                            </th>
                        </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-slate-75">
                        {entities.map((ent, index) => (
                            <tr
                                key={index}
                                className="hover:bg-teal-50/40 transition-colors duration-150 group cursor-default"
                            >
                                <td className="px-5 py-3 whitespace-nowrap">
                                    <div className="flex items-center gap-2">
                                        <span className="w-1.5 h-1.5 rounded-full bg-slate-300 group-hover:bg-teal-400 transition-colors"></span>
                                        <span className="text-sm font-semibold text-slate-800 group-hover:text-slate-900">
                                            {ent.text}
                                        </span>
                                    </div>
                                </td>
                                <td className="px-5 py-3 whitespace-nowrap">
                                    <span className={`entity-badge ${entityColors[ent.label] || 'bg-slate-100 text-slate-700'} ${entityBorderColors[ent.label] || 'border-slate-200'}`}>
                                        {labelDisplayMap[ent.label] || ent.label}
                                    </span>
                                </td>
                                <td className="px-5 py-3 whitespace-nowrap">
                                    <span className="text-xs font-mono text-slate-400 font-medium flex items-center gap-1">
                                        <Hash className="w-3 h-3" />
                                        [{ent.start}, {ent.end}]
                                    </span>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            <div className="px-5 py-2.5 bg-slate-50 border-t border-slate-100 flex items-center justify-between">
                <span className="text-[11px] text-slate-400 font-medium">
                    {entities.length} entity{entities.length !== 1 ? 'ies' : 'y'} detected
                </span>
                <span className="text-[11px] text-slate-400 font-medium">
                    Med7 + SciSpaCy + BC5CDR ensemble
                </span>
            </div>
        </div>
    );
};

export default EntityTable;