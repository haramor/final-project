const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001';

export const searchArticles = async (query) => {
  try {
    console.log("query in search articles", query);
    const response = await fetch(`${API_URL}/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(query)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error searching articles:', error);
    throw error;
  }
}; 

export const getDropdowns = async () => {
  try {
    const response = await fetch(`${API_URL}/dropdown-options`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error getting dropdown:', error);
    throw error;
  }
}; 

export const callRagApi = async (queryText, filters = {}) => {
  try {
    console.log("Calling RAG API with query:", queryText, "and filters:", filters);
    const payload = {
      query: queryText,
      current_birth_control: filters.birth_control || 'Not specified',
      current_side_effects: filters.side_effects || 'Not specified',
      current_age_group: filters.age_group || 'Not specified',
      primary_reason: filters.primary_reason || 'Not specified'
    };

    const response = await fetch(`${API_URL}/api/rag_query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload)
    });
    
    if (!response.ok) {
      let errorBody = "Unknown error";
      try {
          errorBody = await response.json();
          errorBody = errorBody.error || JSON.stringify(errorBody); 
      } catch (parseError) {
          errorBody = response.statusText; 
      }
      throw new Error(`HTTP error! status: ${response.status}, message: ${errorBody}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error calling RAG API:', error);
    throw error;
  }
};