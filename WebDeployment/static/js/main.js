document.addEventListener('DOMContentLoaded', function() {
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultsContainer = document.getElementById('results-container');
    const loadingSpinner = document.getElementById('loading-spinner');
    const resultsContent = document.getElementById('results-content');
    const sentimentBadge = document.getElementById('sentiment-badge');
    const sentimentText = document.getElementById('sentiment-text');
    const resultTicker = document.getElementById('result-ticker');
    const sentimentReasoning = document.getElementById('sentiment-reasoning');

    analyzeBtn.addEventListener('click', async function() {
        // Get input values
        const ticker = document.getElementById('ticker').value.trim();
        const title = document.getElementById('title').value.trim();
        const description = document.getElementById('description').value.trim();

        // Validate inputs
        if (!ticker || !title || !description) {
            alert('Please fill in all fields');
            return;
        }

        // Show results container and loading spinner
        resultsContainer.style.display = 'block';
        loadingSpinner.style.display = 'block';
        resultsContent.style.display = 'none';
        
        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth' });

        try {
            console.log('Sending request to backend...');
            // Send request to backend
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ ticker, title, description })
            });

            // Check if response is ok
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Server error: ${response.status}`);
            }

            // Parse response
            const data = await response.json();
            console.log('Response from server:', data);
            
            if (!data.success && data.error) {
                throw new Error(data.error);
            }

            // Process the response data
            let result;
            
            // Handle different response formats
            if (data.response && typeof data.response === 'object') {
                // If response is already a JSON object
                result = data.response;
                console.log('Using object from response directly');
            } else if (data.response && typeof data.response === 'string') {
                // If response is a string that might contain JSON
                console.log('Attempting to extract JSON from string response');
                try {
                    // First try direct JSON parsing
                    result = JSON.parse(data.response);
                } catch (e) {
                    // If that fails, try to find JSON within the string
                    console.log('Direct parsing failed, trying regex extraction');
                    const match = data.response.match(/\\{.*?\\}/s);
                    if (match) {
                        result = JSON.parse(match[0]);
                    } else {
                        // If no JSON found, create a basic object with the response
                        result = {
                            ticker: ticker,
                            sentiment: "Unknown",
                            sentiment_reasoning: data.response
                        };
                    }
                }
            } else {
                throw new Error('Invalid response format from server');
            }

            // Update UI with results
            console.log('Processed result:', result);
            
            if (result && result.sentiment) {
                // Display the sentiment
                sentimentText.textContent = result.sentiment;
                resultTicker.textContent = result.ticker || ticker;
                sentimentReasoning.textContent = result.sentiment_reasoning || "No reasoning provided";
                
                // Set appropriate class for styling
                sentimentBadge.className = 'sentiment-badge';
                
                if (result.sentiment.toLowerCase() === 'bullish') {
                    sentimentBadge.classList.add('bullish');
                } else if (result.sentiment.toLowerCase() === 'bearish') {
                    sentimentBadge.classList.add('bearish');
                } else {
                    sentimentBadge.classList.add('neutral');
                }
                
                // Hide spinner and show results
                loadingSpinner.style.display = 'none';
                resultsContent.style.display = 'block';
            } else {
                throw new Error('Could not extract sentiment from response');
            }

        } catch (error) {
            console.error('Error:', error);
            alert('Error: ' + error.message);
            loadingSpinner.style.display = 'none';
        }
    });
});