import React, { useState } from 'react';
import { Pill, Clock, Calendar, Droplet, FileText, CheckCircle, AlertCircle, AlertTriangle, ShieldAlert, Package, Sparkles, Gauge, Stethoscope, Edit2, X, Save } from 'lucide-react';

// Confidence color coding: emerald (high), amber (medium), rose (low)
const getConfidenceStyle = (confidence) => {
  if (!confidence) return { bg: 'bg-slate-100', text: 'text-slate-600', label: 'N/A', bar: 'bg-slate-400' };
  if (confidence >= 0.9) return { bg: 'bg-emerald-50', text: 'text-emerald-700', label: 'High', bar: 'bg-emerald-500' };
  if (confidence >= 0.7) return { bg: 'bg-amber-50', text: 'text-amber-700', label: 'Medium', bar: 'bg-amber-500' };
  return { bg: 'bg-rose-50', text: 'text-rose-700', label: 'Low', bar: 'bg-rose-500' };
};

const MedicationCard = ({ medication, index, isVerified, onVerify, onUpdate, originalText }) => {
  // Defensive check for nested data
  const card_summary = medication?.card_summary || {};
  const display = medication?.display || medication?.name || 'Unknown Medication';

  const header = card_summary?.header || medication?.name || 'Unknown Drug';
  const subheader = card_summary?.subheader || medication?.dosage || '';
  const instructions = card_summary?.instructions || medication?.frequency || 'No instructions';
  const details = card_summary?.details || {
    frequency: medication?.frequency || 'N/A',
    duration: medication?.duration || 'N/A',
    form: medication?.form || 'N/A',
    route: medication?.route || 'N/A'
  };

  const isAllergy = medication?.is_allergy;
  const isDiscontinued = medication?.is_discontinued;
  const isDrugClass = medication?.is_drug_class;
  const suggestedCorrection = medication?.suggested_correction;
  const confidence = medication?.confidence;
  const intent = medication?.intent;
  const confStyle = getConfidenceStyle(confidence);

  // Edit mode state
  const [isEditing, setIsEditing] = useState(false);
  const [editedData, setEditedData] = useState({
    drug: header,
    strength: subheader,
    instructions: instructions,
    frequency: details.frequency,
    duration: details.duration,
    form: details.form,
    route: details.route
  });

  const handleSave = () => {
    onUpdate(index, editedData);
    setIsEditing(false);
  };

  const handleCancel = () => {
    setEditedData({
      drug: header,
      strength: subheader,
      instructions: instructions,
      frequency: details.frequency,
      duration: details.duration,
      form: details.form,
      route: details.route
    });
    setIsEditing(false);
  };

  // Determine card style based on status
  const getHeaderStyle = () => {
    if (isDiscontinued) return 'bg-gradient-to-r from-slate-500 to-slate-600';
    if (isAllergy) return 'bg-gradient-to-r from-rose-500 to-red-600';
    if (isVerified) return 'bg-gradient-to-r from-emerald-500 to-teal-600';
    return 'bg-gradient-to-r from-teal-600 to-cyan-700';
  };

  const getBorderStyle = () => {
    if (isDiscontinued) return 'border-slate-300 ring-2 ring-slate-100';
    if (isAllergy) return 'border-rose-300 ring-2 ring-rose-100';
    if (isVerified) return 'border-emerald-300 ring-2 ring-emerald-100';
    return 'border-slate-100';
  };

  return (
    <div className={`bg-white rounded-xl shadow-lg border overflow-hidden hover:shadow-xl transition-all duration-300 ${getBorderStyle()}`}>
      {/* Card Header */}
      <div className={`px-5 py-4 transition-colors duration-300 ${getHeaderStyle()}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-white/20 p-2 rounded-lg">
              <Pill className="w-5 h-5 text-white" />
            </div>
            <div className="flex-1">
              {isEditing ? (
                <div className="space-y-2">
                  <input
                    type="text"
                    value={editedData.drug}
                    onChange={(e) => setEditedData({ ...editedData, drug: e.target.value })}
                    className="w-full bg-white/20 text-white placeholder-white/50 border border-white/30 rounded px-2 py-1 text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-white/50"
                    placeholder="Drug name"
                  />
                  <input
                    type="text"
                    value={editedData.strength}
                    onChange={(e) => setEditedData({ ...editedData, strength: e.target.value })}
                    className="w-full bg-white/20 text-white placeholder-white/50 border border-white/30 rounded px-2 py-1 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-white/50"
                    placeholder="Strength"
                  />
                  {originalText && medication.drug_start !== undefined && medication.drug_end !== undefined && (
                    <div className="mt-2 p-2 bg-white/10 rounded border border-white/20">
                      <p className="text-xs text-white/70 mb-1">Original text snippet:</p>
                      <div className="text-xs text-white/90 font-mono">
                        {originalText.substring(Math.max(0, medication.drug_start - 20), medication.drug_start)}
                        <span className="bg-yellow-400/30 px-1 rounded font-semibold">
                          {originalText.substring(medication.drug_start, medication.drug_end)}
                        </span>
                        {originalText.substring(medication.drug_end, Math.min(originalText.length, medication.drug_end + 20))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <>
                  <div className="flex items-center gap-2">
                    <h3 className="text-white font-bold text-lg">{header}</h3>
                    {suggestedCorrection && (
                      <span className="bg-yellow-400/30 text-yellow-100 text-xs font-semibold px-2 py-0.5 rounded-full flex items-center gap-1">
                        <Sparkles className="w-3 h-3" />
                        → {suggestedCorrection} ({Math.round(confidence * 100)}%)
                      </span>
                    )}
                  </div>
                  {subheader && (
                    <p className="text-blue-100 text-sm font-medium">{subheader}</p>
                  )}
                </>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            {isAllergy && (
              <span className="bg-white/30 text-white text-xs font-semibold px-3 py-1 rounded-full flex items-center gap-1">
                <ShieldAlert className="w-3 h-3" />
                Allergy
              </span>
            )}
            {isDiscontinued && (
              <span className="bg-white/30 text-white text-xs font-semibold px-3 py-1 rounded-full flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" />
                Discontinued
              </span>
            )}
            {isDrugClass && (
              <span className="bg-white/30 text-white text-xs font-semibold px-3 py-1 rounded-full flex items-center gap-1">
                <Package className="w-3 h-3" />
                Drug Class
              </span>
            )}
            {confidence && (
              <span className={`text-xs font-semibold px-3 py-1 rounded-full flex items-center gap-1 ${confStyle.bg} ${confStyle.text}`}>
                <Gauge className="w-3 h-3" />
                {Math.round(confidence * 100)}% ({confStyle.label})
              </span>
            )}
            {!isVerified && confidence < 0.8 && (
              <span className="bg-orange-100 text-orange-700 text-xs font-semibold px-3 py-1 rounded-full flex items-center gap-1 animate-pulse">
                <AlertCircle className="w-3 h-3" />
                Requires Review
              </span>
            )}
            {isVerified && (
              <span className="bg-white/30 text-white text-xs font-semibold px-3 py-1 rounded-full flex items-center gap-1">
                <CheckCircle className="w-3 h-3" />
                Verified
              </span>
            )}
            <span className="bg-white/20 text-white text-xs font-semibold px-3 py-1 rounded-full">
              #{index + 1}
            </span>
          </div>
        </div>
      </div>

      {/* Card Body */}
      <div className="p-5 space-y-4">
        {/* Confidence Bar */}
        {confidence && (
          <div className="flex items-center gap-3">
            <span className="text-xs font-semibold text-gray-500 w-20">Confidence</span>
            <div className="flex-1 bg-gray-200 rounded-full h-2.5 overflow-hidden">
              <div
                className={`h-2.5 rounded-full transition-all duration-500 ${confStyle.bar}`}
                style={{ width: `${Math.round(confidence * 100)}%` }}
              />
            </div>
            <span className={`text-xs font-bold ${confStyle.text}`}>{Math.round(confidence * 100)}%</span>
            {!isVerified && confidence < 0.9 && (
              <span className="text-xs text-red-500 font-medium animate-pulse">Needs Review</span>
            )}
          </div>
        )}

        {/* Intent Badge */}
        {intent && !isDiscontinued && (
          <div className="flex items-center gap-2">
            <span className={`text-xs font-semibold px-3 py-1 rounded-full ${intent === 'start' ? 'bg-green-100 text-green-700' :
              intent === 'consider' ? 'bg-yellow-100 text-yellow-700' :
                intent === 'as needed' ? 'bg-blue-100 text-blue-700' :
                  intent === 'continue' ? 'bg-teal-100 text-teal-700' :
                    intent === 'stop' ? 'bg-red-100 text-red-700' :
                      intent === 'hold' ? 'bg-orange-100 text-orange-700' :
                        'bg-gray-100 text-gray-700'
              }`}>
              Intent: {intent}
            </span>
          </div>
        )}

        {/* Instructions Row */}
        <div className="flex items-start gap-3">
          <div className={`p-2 rounded-lg ${isDiscontinued ? 'bg-gray-50' : isAllergy ? 'bg-red-50' : isVerified ? 'bg-green-50' : 'bg-blue-50'}`}>
            <FileText className={`w-4 h-4 ${isDiscontinued ? 'text-gray-600' : isAllergy ? 'text-red-600' : isVerified ? 'text-green-600' : 'text-blue-600'}`} />
          </div>
          <div className="flex-1">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Instructions</p>
            {isEditing ? (
              <input
                type="text"
                value={editedData.instructions}
                onChange={(e) => setEditedData({ ...editedData, instructions: e.target.value })}
                className="w-full mt-1 px-2 py-1 border border-slate-200 rounded text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-400"
                placeholder="Dosage instructions"
              />
            ) : (
              <p className="text-gray-800 font-medium">{instructions}</p>
            )}
          </div>
        </div>

        {/* Details Grid */}
        <div className="grid grid-cols-2 gap-3">
          <div className={`flex items-center gap-2 p-3 rounded-lg ${isVerified ? 'bg-green-50/50' : 'bg-gray-50'}`}>
            <Droplet className="w-4 h-4 text-blue-500" />
            <div className="flex-1">
              <p className="text-xs text-gray-500">Route</p>
              {isEditing ? (
                <input
                  type="text"
                  value={editedData.route}
                  onChange={(e) => setEditedData({ ...editedData, route: e.target.value })}
                  className="w-full mt-1 px-2 py-1 border border-slate-200 rounded text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-400"
                  placeholder="Route"
                />
              ) : (
                <p className="text-sm font-medium text-gray-800">{details.route}</p>
              )}
            </div>
          </div>
          <div className={`flex items-center gap-2 p-3 rounded-lg ${isVerified ? 'bg-green-50/50' : 'bg-gray-50'}`}>
            <Clock className="w-4 h-4 text-orange-500" />
            <div className="flex-1">
              <p className="text-xs text-gray-500">Frequency</p>
              {isEditing ? (
                <input
                  type="text"
                  value={editedData.frequency}
                  onChange={(e) => setEditedData({ ...editedData, frequency: e.target.value })}
                  className="w-full mt-1 px-2 py-1 border border-slate-200 rounded text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-400"
                  placeholder="Frequency"
                />
              ) : (
                <p className="text-sm font-medium text-gray-800">{details.frequency}</p>
              )}
            </div>
          </div>
          <div className={`flex items-center gap-2 p-3 rounded-lg ${isVerified ? 'bg-green-50/50' : 'bg-gray-50'}`}>
            <Calendar className="w-4 h-4 text-purple-500" />
            <div className="flex-1">
              <p className="text-xs text-gray-500">Duration</p>
              {isEditing ? (
                <input
                  type="text"
                  value={editedData.duration}
                  onChange={(e) => setEditedData({ ...editedData, duration: e.target.value })}
                  className="w-full mt-1 px-2 py-1 border border-slate-200 rounded text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-400"
                  placeholder="Duration"
                />
              ) : (
                <p className="text-sm font-medium text-gray-800">{details.duration}</p>
              )}
            </div>
          </div>
          <div className={`flex items-center gap-2 p-3 rounded-lg ${isVerified ? 'bg-green-50/50' : 'bg-gray-50'}`}>
            <Pill className="w-4 h-4 text-pink-500" />
            <div className="flex-1">
              <p className="text-xs text-gray-500">Form</p>
              {isEditing ? (
                <input
                  type="text"
                  value={editedData.form}
                  onChange={(e) => setEditedData({ ...editedData, form: e.target.value })}
                  className="w-full mt-1 px-2 py-1 border border-slate-200 rounded text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-400"
                  placeholder="Form"
                />
              ) : (
                <p className="text-sm font-medium text-gray-800">{details.form}</p>
              )}
            </div>
          </div>
        </div>

        {/* Standardized String */}
        <div className="mt-4 pt-4 border-t border-gray-100">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
            Standardized
          </p>
          <p className={`text-sm p-3 rounded-lg font-medium ${isVerified ? 'bg-green-50 text-green-800' : 'bg-blue-50 text-gray-700'
            }`}>
            {display}
          </p>
        </div>
      </div>

      {/* Card Footer */}
      <div className="px-5 py-3 bg-gray-50 border-t border-gray-100 flex items-center justify-between">
        <span className="text-xs text-gray-500">
          {medication.related_entities?.length || 0} related entities
        </span>
        <div className="flex items-center gap-2">
          {isEditing ? (
            <>
              <button
                onClick={handleCancel}
                className="flex items-center gap-2 text-sm font-medium px-3 py-2 rounded-lg bg-slate-100 text-slate-600 hover:bg-slate-200 transition-colors"
              >
                <X className="w-4 h-4" />
                Cancel
              </button>
              <button
                onClick={handleSave}
                className="flex items-center gap-2 text-sm font-medium px-3 py-2 rounded-lg bg-teal-600 text-white hover:bg-teal-700 transition-colors"
              >
                <Save className="w-4 h-4" />
                Save
              </button>
            </>
          ) : (
            <>
              <button
                onClick={() => setIsEditing(true)}
                className="flex items-center gap-2 text-sm font-medium px-3 py-2 rounded-lg bg-slate-100 text-slate-600 hover:bg-slate-200 transition-colors"
              >
                <Edit2 className="w-4 h-4" />
                Edit
              </button>
              <button
                onClick={onVerify}
                disabled={isVerified}
                className={`flex items-center gap-2 text-sm font-medium px-3 py-2 rounded-lg transition-all ${isVerified
                  ? 'bg-green-100 text-green-700 cursor-default'
                  : 'bg-green-500 hover:bg-green-600 text-white active:scale-95'
                  }`}
              >
                {isVerified ? (
                  <>
                    <CheckCircle className="w-4 h-4" />
                    Verified
                  </>
                ) : (
                  <>
                    <CheckCircle className="w-4 h-4" />
                    Verify
                  </>
                )}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

const MedicationSummary = ({ medications, standardizedText, originalText, onVerify }) => {
  // Track verified medications by index
  const [verifiedMeds, setVerifiedMeds] = useState(new Set());
  // Track edited medications
  const [editedMeds, setEditedMeds] = useState(new Map());

  // Handle undefined/null medications
  const meds = medications || [];

  // Reset verified state when medications change
  React.useEffect(() => {
    setVerifiedMeds(new Set());
    setEditedMeds(new Map());
  }, [JSON.stringify(medications)]);

  const handleVerify = (index) => {
    setVerifiedMeds(prev => {
      const newSet = new Set([...prev, index]);
      console.log(`Medication #${index + 1} verified:`, meds[index]?.drug);

      // If all medications are verified, call parent onVerify
      if (newSet.size === meds.length && onVerify) {
        const verifiedMedications = meds.map((med, idx) => {
          if (editedMeds.has(idx)) {
            const edited = editedMeds.get(idx);
            return {
              ...med,
              drug: edited.drug,
              card_summary: {
                ...med.card_summary,
                header: edited.drug,
                subheader: edited.strength,
                instructions: edited.instructions,
                details: {
                  ...med.card_summary.details,
                  frequency: edited.frequency,
                  duration: edited.duration,
                  form: edited.form,
                  route: edited.route
                }
              },
              display: `${edited.drug}${edited.strength ? ` (${edited.strength})` : ''}${edited.frequency ? ` - ${edited.frequency}` : ''}${edited.duration ? ` for ${edited.duration}` : ''}`
            };
          }
          return med;
        });
        onVerify(verifiedMedications);
      }

      return newSet;
    });
  };

  const handleUpdate = (index, newData) => {
    setEditedMeds(prev => new Map(prev.set(index, newData)));
    console.log(`Medication #${index + 1} updated:`, newData);
  };

  // Get the medication data (original or edited)
  const getMedicationData = (index) => {
    if (editedMeds.has(index)) {
      const edited = editedMeds.get(index);
      return {
        ...meds[index],
        drug: edited.drug,
        card_summary: {
          ...meds[index].card_summary,
          header: edited.drug,
          subheader: edited.strength,
          instructions: edited.instructions,
          details: {
            ...meds[index].card_summary.details,
            frequency: edited.frequency,
            duration: edited.duration,
            form: edited.form,
            route: edited.route
          }
        },
        display: `${edited.drug}${edited.strength ? ` (${edited.strength})` : ''}${edited.frequency ? ` - ${edited.frequency}` : ''}${edited.duration ? ` for ${edited.duration}` : ''}`
      };
    }
    return meds[index];
  };

  const verifiedCount = verifiedMeds.size;
  const allVerified = meds.length > 0 && verifiedCount === meds.length;

  if (meds.length === 0) {
    return (
      <div className="bg-slate-50 rounded-xl p-8 text-center border border-slate-200">
        <div className="bg-white w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 shadow-sm border border-slate-100">
          <Pill className="w-8 h-8 text-slate-400" />
        </div>
        <h3 className="text-slate-800 font-semibold text-lg mb-2">No Medications Detected</h3>
        <p className="text-slate-500 text-sm max-w-sm mx-auto leading-relaxed">
          Upload a prescription or enter text with drug names, dosages, and frequencies to begin extraction.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Summary Header with Verification Status */}
      <div className={`rounded-xl p-5 border transition-colors duration-300 ${allVerified
        ? 'bg-gradient-to-r from-emerald-50 to-teal-50 border-emerald-200'
        : 'bg-gradient-to-r from-teal-50 to-cyan-50 border-teal-100'
        }`}>
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-xl font-bold text-slate-800">Clinical Summary</h2>
              {allVerified && (
                <span className="bg-emerald-500 text-white text-xs font-semibold px-3 py-1 rounded-full flex items-center gap-1">
                  <CheckCircle className="w-3 h-3" />
                  All Verified
                </span>
              )}
            </div>
            <p className="text-slate-600 text-sm mt-1">
              {meds.length} medication{meds.length !== 1 ? 's' : ''} detected
              {verifiedCount > 0 && (
                <span className="text-emerald-600 font-medium ml-2">
                  ({verifiedCount} verified)
                </span>
              )}
            </p>
          </div>
          <div className={`p-3 rounded-xl shadow-sm ${allVerified ? 'bg-emerald-100' : 'bg-white'}`}>
            <Stethoscope className={`w-8 h-8 ${allVerified ? 'text-emerald-600' : 'text-teal-600'}`} />
          </div>
        </div>
      </div>

      {/* Standardized Text */}
      {standardizedText && (
        <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm">
          <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
            <FileText className="w-4 h-4" />
            Standardized Prescription
          </h3>
          <p className="text-slate-800 leading-relaxed text-sm font-medium bg-slate-50 p-3 rounded-lg border border-slate-100">{standardizedText}</p>
        </div>
      )}

      {/* Medication Cards Grid */}
      <div className="grid gap-5">
        {meds.map((med, index) => (
          <MedicationCard
            key={`${med.drug}-${index}`}
            medication={getMedicationData(index)}
            index={index}
            isVerified={verifiedMeds.has(index)}
            onVerify={() => handleVerify(index)}
            onUpdate={handleUpdate}
            originalText={originalText}
          />
        ))}
      </div>
    </div>
  );
};

export default MedicationSummary;
