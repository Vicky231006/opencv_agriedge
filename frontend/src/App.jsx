import React, { useState } from 'react';

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [selectedCrop, setSelectedCrop] = useState("tomato"); // New state to hold crop context
  const [loading, setLoading] = useState(false);
  const [analysisData, setAnalysisData] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setAnalysisData(null);
    }
  };

  const triggerPipelineAnalysis = async () => {
    if (!selectedFile) return;
    
    setLoading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("crop_type", selectedCrop); // Safely shipping crop type to the backend

    try {
      const response = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        body: formData,
      });
      
      const data = await response.json();
      if (data.status === "success") {
        setAnalysisData(data);
      } else {
        alert("Pipeline error: Missing validation context from server.");
      }
    } catch (error) {
      console.error("Connection failed:", error);
      alert("Failed to connect to PyTorch backend instance.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0B0F19] flex flex-col font-sans">
      <header className="border-b border-slate-800 bg-[#111827]/80 backdrop-blur-md sticky top-0 z-50 px-6 py-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="h-3 w-3 rounded-full bg-[#B4FF00] animate-pulse" />
            <h1 className="text-xl font-bold tracking-wider text-white">
              HERMES <span className="text-[#B4FF00] font-medium text-sm border border-[#B4FF00]/30 px-1.5 py-0.5 rounded ml-1">AGRI-EDGE v1.0</span>
            </h1>
          </div>
          <span className="text-xs text-slate-400 font-mono tracking-widest uppercase">
            Dense Spatial Segmentation Core
          </span>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* LEFT COLUMN: Data Control & Ingestion Panel */}
        <section className="lg:col-span-4 flex flex-col gap-6">
          <div className="bg-[#111827] border border-slate-800 rounded-xl p-6 lime-glow transition-all">
            <h2 className="text-sm font-semibold text-slate-300 tracking-wider uppercase mb-4">
              Canopy Ingestion Source
            </h2>
            
            {/* New Crop Selection Dropdown Component */}
            <div className="mb-4">
              <label className="block text-xs font-mono text-slate-400 uppercase tracking-wider mb-2">
                Target Crop Variety
              </label>
              <select 
                value={selectedCrop}
                onChange={(e) => setSelectedCrop(e.target.value)}
                className="w-full bg-[#0F1622] border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-[#B4FF00] transition-colors"
              >
                <option value="tomato">Tomato (Solanum lycopersicum)</option>
                <option value="apple">Apple (Malus domestica)</option>
                <option value="grape">Grape (Vitis vinifera)</option>
                <option value="potato">Potato (Solanum tuberosum)</option>
              </select>
            </div>
            
            {/* Interactive Upload Dropzone */}
            <label className="group flex flex-col items-center justify-center w-full h-44 border-2 border-dashed border-slate-700 hover:border-[#B4FF00]/50 rounded-lg cursor-pointer bg-[#0F1622] hover:bg-[#151D2A] transition-all p-4 text-center">
              <div className="flex flex-col items-center justify-center">
                <svg className="w-8 h-8 text-slate-400 group-hover:text-[#B4FF00] mb-2 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p className="text-sm text-slate-300 font-medium mb-1">Upload Leaf Profile</p>
                <p className="text-xs text-slate-500 font-mono">PNG, JPG up to 224x224</p>
              </div>
              <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
            </label>

            {previewUrl && (
              <button
                onClick={triggerPipelineAnalysis}
                disabled={loading}
                className="w-full mt-5 bg-[#B4FF00] hover:bg-[#97D600] text-black font-semibold uppercase tracking-wider text-xs py-3 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="h-4 w-4 border-2 border-black border-t-transparent rounded-full animate-spin" />
                    Executing Gradients...
                  </>
                ) : "Execute Diagnostic Loop"}
              </button>
            )}
          </div>
        </section>

        {/* RIGHT COLUMN: Pipeline Outputs & Analytics Matrix */}
        <section className="lg:col-span-8 flex flex-col gap-6">
          <div className="bg-[#111827] border border-slate-800 rounded-xl p-6 grid grid-cols-1 md:grid-cols-2 gap-6 min-h-[300px]">
            <div className="flex flex-col gap-2">
              <h3 className="text-xs font-mono text-slate-400 uppercase tracking-wider">Input Canopy Channel</h3>
              <div className="flex-1 min-h-[224px] bg-[#0F1622] rounded-lg border border-slate-800 flex items-center justify-center overflow-hidden">
                {previewUrl ? (
                  <img src={previewUrl} alt="Input source" className="w-full h-full object-cover" />
                ) : (
                  <span className="text-xs font-mono text-slate-600">Awaiting file input stream...</span>
                )}
              </div>
            </div>

            <div className="flex flex-col gap-2">
              <h3 className="text-xs font-mono text-slate-400 uppercase tracking-wider">Predicted Lesion Boundary (Class 1)</h3>
              <div className="flex-1 min-h-[224px] bg-[#0F1622] rounded-lg border border-slate-800 flex items-center justify-center overflow-hidden relative">
                {analysisData ? (
                  <img 
                    src={`data:image/png;base64,${analysisData.mask_image}`} 
                    alt="AI Prediction Mask" 
                    className="w-full h-full object-cover" 
                  />
                ) : (
                  <span className="text-xs font-mono text-slate-600">
                    {loading ? "Computing dense spatial matrices..." : "Inference pipeline offline."}
                  </span>
                )}
              </div>
            </div>
          </div>

          {analysisData && (
            <div className="bg-[#111827] border border-slate-800 rounded-xl p-6 grid grid-cols-1 md:grid-cols-12 gap-6">
              <div className="md:col-span-4 flex flex-col justify-center items-center bg-[#0F1622] p-4 rounded-lg border border-slate-800 text-center">
                <span className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-2">Calculated Threat</span>
                <div className="text-4xl font-extrabold text-[#B4FF00] font-mono tracking-tighter">
                  {analysisData.analytics.calculated_yield_risk_percentage.toFixed(2)}%
                </div>
                <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest mt-1">Yield Loss At Risk</span>
              </div>

              <div className="md:col-span-8 flex flex-col justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-[10px] font-mono text-slate-400 uppercase bg-slate-800 px-2 py-0.5 rounded font-semibold tracking-wider">
                      {analysisData.management_prescription.alarm_level}
                    </span>
                  </div>
                  <p className="text-sm text-slate-300 leading-relaxed font-sans">
                    <span className="text-slate-400 font-medium">Directive:</span> {analysisData.management_prescription.intervention_directive}
                  </p>
                </div>
                
                <div className="border-t border-slate-800/60 pt-3 flex justify-between items-center font-mono">
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-wider">Target Dosage</div>
                    <div className="text-sm font-semibold text-white">{analysisData.management_prescription.pesticide_dosage_ml_per_m2} ml/m²</div>
                  </div>
                  <div className="text-right">
                    <div className="text-[10px] text-slate-500 uppercase tracking-wider">Chemical Spec</div>
                    <div className="text-sm font-semibold text-[#B4FF00]">{analysisData.management_prescription.chemical_composition}</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}