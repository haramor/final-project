import React, { useState, useEffect } from 'react';
import './SearchForm.css';
import Select from 'react-select';
import { getDropdowns, callRagApi } from '../services/api';
import ReactMarkdown from 'react-markdown';

function SearchForm() {
  const [options, setOptions] = useState({
    birth_control_methods: [],
    side_effects: [],
    age_groups: [],
    primary_reason: [],
    additional_filters: [],
    mesh_terms: []
  });
  
  const [selectedFilters, setSelectedFilters] = useState({
    birth_control: [],
    side_effects: [],
    age_group: [],
    primary_reason: [],
    additional: [],
    mesh_terms: []
  });
  
  const [naturalLanguageQuery, setNaturalLanguageQuery] = useState('');
  const [ragAnswer, setRagAnswer] = useState("");
  const [ragSources, setRagSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [isFilterSidebarOpen, setIsFilterSidebarOpen] = useState(true);
  
  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const response = await getDropdowns();
        console.log("Response from /dropdown-options API:", response);
        
        setOptions(prevOptions => ({
          ...prevOptions, 
          ...response,    
          birth_control_methods: response.birth_control_methods || [],
          side_effects: response.side_effects || [],
          age_groups: response.age_groups || [],
          primary_reason: response.primary_reason || [] 
        }));
      } catch (err) {
        setError('Failed to load dropdown options');
        console.error("Error in fetchOptions trying to get dropdowns:", err);
      }
    };
    
    fetchOptions();
  }, []);
  
  const handleSearch = async () => {
    if (!naturalLanguageQuery.trim()) {
      setError("Please enter a question.");
      return;
    }
    setLoading(true);
    setError(null);
    setRagAnswer("");
    setRagSources([]);
    setHasSearched(true);
    
    // Prepare filters for the API call
    const apiFilters = {
      birth_control: selectedFilters.birth_control.length > 0 ? selectedFilters.birth_control.join(', ') : 'Not specified',
      side_effects: selectedFilters.side_effects.length > 0 ? selectedFilters.side_effects.join(', ') : 'Not specified',
      age_group: selectedFilters.age_group.length > 0 ? selectedFilters.age_group.join(', ') : 'Not specified',
      primary_reason: selectedFilters.primary_reason.length > 0 ? selectedFilters.primary_reason.join(', ') : 'Not specified'
    };

    try {
      // Pass the natural language query and the prepared filters
      const response = await callRagApi(naturalLanguageQuery, apiFilters);
      console.log("RAG API response:", response);
      
      setRagAnswer(response.answer || "No answer received.");
      setRagSources(response.sources || []);

    } catch (err) {
      setError(err.message || 'Failed to get response from assistant');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  
  const handleFilterChange = (selectedOptions, actionMeta) => {
    const { name } = actionMeta;
    setSelectedFilters({
      ...selectedFilters,
      [name]: selectedOptions ? selectedOptions.map(option => option.value) : []
    });
  };
  
  return (
    <>
      {/* Filter Sidebar */}
      <div className={`filter-sidebar ${isFilterSidebarOpen ? 'open' : 'closed'}`}>
        <button 
          className="filter-toggle"
          onClick={() => setIsFilterSidebarOpen(!isFilterSidebarOpen)}
        >
          {isFilterSidebarOpen ? '←' : '→'}
        </button>
        <div className="filters-section">
          <h3>Tell us about yourself</h3>
          <p>What birth control are you taking?</p>
          <Select
            isMulti
            name="birth_control"
            options={(options.birth_control_methods || []).map(method => ({ label: method, value: method }))}
            onChange={handleFilterChange}
            placeholder="Birth Control Methods"
            className="select-filter"
            classNamePrefix="select"
          />
          <p>Do you experience any side effects?</p>
          <Select
            isMulti
            name="side_effects"
            options={(options.side_effects || []).map(effect => ({ label: effect, value: effect }))}
            onChange={handleFilterChange}
            placeholder="Side Effects"
            className="select-filter"
            classNamePrefix="select"
          />
          <p>What age group are you in?</p>
          <Select
            isMulti
            name="age_group"
            options={(options.age_groups || []).map(age => ({ label: age, value: age }))}
            onChange={handleFilterChange}
            placeholder="Age Groups"
            className="select-filter"
            classNamePrefix="select"
          />
          <p>What is your primary reason for contraception?</p>
          <Select
            isMulti
            name="primary_reason"
            options={(options.primary_reason || []).map(reason => ({ label: reason, value: reason }))}
            onChange={handleFilterChange}
            placeholder="Primary Reason"
            className="select-filter"
            classNamePrefix="select"
          />
        </div>
      </div>

      {/* Main Content */}
      <div className={`chat-container ${!isFilterSidebarOpen ? 'sidebar-closed' : ''}`}>
        {/* Messages/Results Area */}
        <div className="messages-area">
          {loading ? (
            <div className="loading-container">
              <div className="loading-spinner"></div>
              <p>Thinking...</p>
            </div>
          ) : (
            <>
              {ragAnswer && (
                <div className="results-scroll">
                  <div className="message-content markdown-content">
                    <ReactMarkdown>{ragAnswer}</ReactMarkdown>
                    {ragSources.length > 0 && (
                      <div className="sources-section">
                        <h4>Sources:</h4>
                        <ul>
                          {ragSources.map((source, index) => (
                            <li key={index}>{source}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              )}
              
              {!ragAnswer && !hasSearched && (
                <div className="empty-state">
                  <h2>Contraceptives Research Assistant</h2>
                </div>
              )}
            </>
          )}
        </div>

        {/* Input Area */}
        <div className={`input-area ${!hasSearched ? 'centered' : ''}`}>
          <div className="input-container">
            <textarea
              value={naturalLanguageQuery}
              onChange={(e) => setNaturalLanguageQuery(e.target.value)}
              placeholder="Ask your question about contraceptives..."
              className="query-input"
              rows={1}
              onInput={(e) => {
                e.target.style.height = 'auto';
                e.target.style.height = e.target.scrollHeight + 'px';
              }}
            />
            <button 
              className="submit-button" 
              onClick={handleSearch}
              disabled={loading || !naturalLanguageQuery.trim()}
            >
              {loading ? '...' : '→'}
            </button>
          </div>
          {error && <div className="error-message">{error}</div>}
        </div>
      </div>
    </>
  );
};

export default SearchForm;