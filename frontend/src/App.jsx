import React, { useState, useRef } from 'react';
import axios from 'axios';

// --- Icon Components ---
const UploadIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-10 h-10 mb-4 text-gray-400">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5V9.75m0 0l-3.75 3.75M12 9.75l3.75 3.75M3 17.25V6.75A2.25 2.25 0 015.25 4.5h13.5A2.25 2.25 0 0121 6.75v10.5A2.25 2.25 0 0118.75 19.5H5.25A2.25 2.25 0 013 17.25z" />
    </svg>
);

const CheckCircleIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 text-green-500">
      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z" clipRule="evenodd" />
    </svg>
);

// --- Helper Components ---
const Spinner = () => (
    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
);

// Renders a key-value pair. Now handles nested objects recursively.
const DetailItem = ({ label, value }) => {
    if (!value || value === 'N/A' || (Array.isArray(value) && value.length === 0)) {
        return null;
    }

    // ** THE FIX IS HERE **
    // If the value is a plain object (but not an array), render its keys and values recursively.
    if (typeof value === 'object' && !Array.isArray(value) && value !== null) {
        return (
             <div className="mb-4">
                <h4 className="font-semibold text-gray-800 capitalize text-sm">{label.replace(/_/g, ' ')}</h4>
                <div className="pl-4 mt-1 border-l-2 border-gray-200">
                    {Object.entries(value).map(([key, val]) => (
                        <DetailItem key={key} label={key} value={val} />
                    ))}
                </div>
            </div>
        )
    }

    // Original logic for arrays and simple text values
    return (
        <div className="mb-4">
            <h4 className="font-semibold text-gray-800 capitalize text-sm">{label.replace(/_/g, ' ')}</h4>
            {Array.isArray(value) ? (
                <ul className="list-disc list-inside text-gray-600 text-sm pl-4 mt-1 space-y-1">
                    {value.map((item, index) => <li key={index}>{item}</li>)}
                </ul>
            ) : (
                <p className="text-gray-600 text-sm mt-1">{String(value)}</p>
            )}
        </div>
    );
};

// --- Main App Components ---
const JobMatchCard = ({ match }) => {
    const [isExpanded, setIsExpanded] = useState(false);
    let jobData = {};
    try {
        jobData = JSON.parse(match.job_description);
    } catch (e) {
        console.error("Failed to parse job description JSON:", e);
        jobData = { "details": match.job_description };
    }
    
    const scorePercentage = Math.max(0, 100 - (match.score / 2) * 100).toFixed(1);

    return (
        <div className="bg-white p-5 rounded-xl shadow-md border border-gray-100 transition-all duration-300 hover:shadow-lg hover:border-indigo-200">
            <div className="flex justify-between items-start">
                <div>
                    <p className="text-xs font-semibold text-indigo-600 bg-indigo-100 px-2 py-1 rounded-full inline-block">Rank #{match.rank}</p>
                    <h3 className="text-xl font-bold text-gray-900 mt-2">{match.company}</h3>
                </div>
                <div className="text-right flex-shrink-0 ml-4">
                    <p className="text-2xl font-bold text-indigo-600">{scorePercentage}%</p>
                    <p className="text-xs text-gray-500">Match Score</p>
                </div>
            </div>
            <div className={`mt-4 pt-4 border-t border-dashed transition-all duration-500 ease-in-out overflow-hidden ${isExpanded ? 'max-h-[1000px]' : 'max-h-0'}`}>
                {Object.entries(jobData).map(([key, value]) => (
                    <DetailItem key={key} label={key} value={value} />
                ))}
            </div>
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="text-indigo-600 hover:text-indigo-800 text-sm font-semibold mt-4 w-full text-left"
            >
                {isExpanded ? 'Show Less Details' : 'Show More Details'}
            </button>
        </div>
    );
};
  
const StructuredResumeDisplay = ({ resume }) => (
    <div className="bg-white p-5 rounded-xl shadow-md border border-gray-100">
        <h3 className="text-xl font-bold text-gray-900 mb-4 pb-3 border-b border-dashed">Extracted Resume Details</h3>
        {Object.entries(resume).map(([key, value]) => (
            <DetailItem key={key} label={key} value={value} />
        ))}
    </div>
);

const App = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [apiResponse, setApiResponse] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const fileInputRef = useRef(null);

    const resetState = () => {
        setSelectedFile(null);
        setApiResponse(null);
        setIsLoading(false);
        setError('');
        if(fileInputRef.current) fileInputRef.current.value = '';
    };

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file && file.type === 'application/pdf') {
            setSelectedFile(file);
            setError('');
        } else {
            setSelectedFile(null);
            setError('Please select a valid PDF file.');
        }
    };

    const handleDragOver = (e) => e.preventDefault();
    const handleDrop = (e) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file && file.type === 'application/pdf') {
            setSelectedFile(file);
            setError('');
        } else {
            setError('Please drop a valid PDF file.');
        }
    };

    const handleSubmit = async () => {
        if (!selectedFile) {
            setError('Please select a file first.');
            return;
        }
        setIsLoading(true);
        setApiResponse(null);
        setError('');
        const formData = new FormData();
        formData.append('file', selectedFile);
        try {
            const response = await axios.post('http://localhost:8000/upload_and_match/?top_n=5', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setApiResponse(response.data);
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'An unexpected error occurred.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="bg-gray-50 min-h-screen font-sans text-gray-800">
            <header className="bg-white border-b border-gray-200">
                <div className="max-w-7xl mx-auto py-5 px-4 sm:px-6 lg:px-8 flex justify-between items-center">
                    <div>
                      <h1 className="text-3xl font-bold text-gray-900">Resume Rover</h1>
                      <p className="text-sm text-gray-500 mt-1">Navigate Your Career Path. Find the Perfect Job Match.</p>
                    </div>
                </div>
            </header>
      
            <main className="max-w-7xl mx-auto py-10 px-4 sm:px-6 lg:px-8">
                {!apiResponse ? (
                    <div className="max-w-2xl mx-auto">
                        <div className="bg-white p-6 sm:p-8 rounded-xl shadow-md border border-gray-100">
                            <h2 className="text-2xl font-bold text-gray-900 mb-1">Upload Your Resume</h2>
                            <p className="text-sm text-gray-500 mb-6">Let's find your next opportunity. Drop your PDF below.</p>
                            <div onDragOver={handleDragOver} onDrop={handleDrop}>
                                <label htmlFor="dropzone-file" className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-indigo-50 transition">
                                    <div className="flex flex-col items-center justify-center">
                                        <UploadIcon />
                                        <p className="mb-2 text-sm text-gray-500"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                                        <p className="text-xs text-gray-500">PDF (MAX. 5MB)</p>
                                    </div>
                                    <input ref={fileInputRef} id="dropzone-file" type="file" className="hidden" accept=".pdf" onChange={handleFileChange} />
                                </label>
                            </div>
                            {selectedFile && (
                                <div className="mt-5 flex items-center justify-between bg-green-50 p-3 rounded-lg border border-green-200">
                                    <div className="flex items-center gap-3">
                                        <CheckCircleIcon />
                                        <span className="text-sm font-medium text-green-800">{selectedFile.name}</span>
                                    </div>
                                    <button onClick={() => {setSelectedFile(null); if (fileInputRef.current) fileInputRef.current.value = ''}} className="text-gray-500 hover:text-gray-700 text-sm font-semibold">Clear</button>
                                </div>
                            )}
                            <div className="mt-6">
                                <button
                                    onClick={handleSubmit}
                                    disabled={isLoading || !selectedFile}
                                    className="w-full bg-indigo-600 text-white font-bold py-3 px-4 rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-indigo-400 disabled:cursor-not-allowed transition-all flex items-center justify-center h-12"
                                >
                                    {isLoading ? <Spinner /> : 'Find My Matches'}
                                </button>
                            </div>
                            {error && <p className="mt-4 text-sm text-red-600 text-center">{error}</p>}
                        </div>
                    </div>
                ) : (
                    <div>
                        <div className="text-center mb-10">
                            <h2 className="text-3xl font-bold text-gray-900">Matching Results</h2>
                            <p className="text-md text-gray-500 mt-1">Here are the top job opportunities based on your resume.</p>
                        </div>
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
                            <div className="lg:sticky top-8">
                                <StructuredResumeDisplay resume={apiResponse.structured_resume} />
                            </div>
                            <div className="space-y-6">
                                {apiResponse.matches.map((match) => (
                                    <JobMatchCard key={match.job_id} match={match} />
                                ))}
                            </div>
                        </div>
                        <div className="text-center mt-12">
                             <button
                                onClick={resetState}
                                className="bg-gray-700 text-white font-bold py-3 px-6 rounded-lg hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500 transition-all"
                            >
                                Start New Search
                            </button>
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
};

export default App;

