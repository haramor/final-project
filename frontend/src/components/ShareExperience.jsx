import React, { useState } from 'react';
import './ShareExperience.css';

function ShareExperience() {
  const [experience, setExperience] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    // TODO: Handle submission logic
    console.log('Experience submitted:', experience);
  };

  return (
    <div className="experience-container">
      <div className="experience-content">
        <h2>Share Your Experience</h2>
        <form onSubmit={handleSubmit}>
          <textarea
            className="experience-input"
            value={experience}
            onChange={(e) => setExperience(e.target.value)}
            placeholder="Tell us about your experience with contraceptives..."
            rows={10}
          />
          <button type="submit" className="submit-button">
            Share
          </button>
        </form>
      </div>
    </div>
  );
}

export default ShareExperience; 