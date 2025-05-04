import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SearchForm from './components/SearchForm';
import ShareExperience from './components/ShareExperience';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app-container">
        <header className="app-header">
          <div className="header-titles">
            {/* <h1>Contraceptive Counselor</h1> */}
            <h2>XX</h2>
          </div>
        </header>
        
        <div className="main-content">
          {/* <button 
            className="sidebar-toggle"
            onClick={toggleSidebar}
            aria-label="Toggle Sidebar"
          >
            {isSidebarOpen ? <FaTimes size={20} /> : <FaBars size={20} />}
          </button>

          <nav className={`sidebar ${isSidebarOpen ? 'open' : 'closed'}`}>
            <NavLink to="/" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              Ask a Question
            </NavLink>
            <NavLink to="/share" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              Tell us Your Experience
            </NavLink>
          </nav> */}
          
          <div className="content-area">
            <Routes>
              <Route path="/" element={<SearchForm />} />
              <Route path="/share" element={<ShareExperience />} />
            </Routes>
          </div>
        </div>
      </div>
    </Router>
  );
}

export default App;