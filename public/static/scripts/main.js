document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const chatHistory = document.getElementById('chat-history');
    const generationInfo = document.querySelector('.generation-info');
    const responsesContainer = document.querySelector('.responses-container');

    // Add loading elements
    const loadingWheel = document.createElement('div');
    loadingWheel.className = 'loading-wheel';
    const loadingText = document.createElement('div');
    loadingText.className = 'loading-text';
    chatHistory.appendChild(loadingWheel);
    chatHistory.appendChild(loadingText);

    const loadingMessages = [
        "Procesando tu solicitud...",
        "Analizando el contexto...",
        "Generando respuesta creativa...",
        "Aplicando reglas lingüísticas...",
        "Refinando el texto generado..."
    ];

    // Get the base URL dynamically
    const getBaseUrl = () => {
        // For local development
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return 'http://localhost:8004';
        }
        // For production on Render
        if (window.location.hostname.includes('onrender.com')) {
            // Use HTTPS in production
            return `https://${window.location.host}`;
        }
        // Fallback to current origin
        return window.location.origin;
    };

    let isGenerating = false;

    const simulateLoading = async () => {
        const duration = Math.floor(Math.random() * (20000 - 15000 + 1)) + 15000; // Random between 15-20 seconds
        const intervalDuration = 3000; // Change message every 3 seconds
        let startTime = Date.now();
        let messageIndex = 0;

        loadingWheel.classList.add('active');
        loadingText.classList.add('active');

        return new Promise(resolve => {
            const updateMessage = () => {
                loadingText.textContent = loadingMessages[messageIndex];
                messageIndex = (messageIndex + 1) % loadingMessages.length;
            };

            updateMessage(); // Show first message immediately
            const messageInterval = setInterval(updateMessage, intervalDuration);

            setTimeout(() => {
                clearInterval(messageInterval);
                loadingWheel.classList.remove('active');
                loadingText.classList.remove('active');
                resolve();
            }, duration);
        });
    };

    // Function to style special words
    const styleSpecialWords = (text) => {
        // Common Spanish verb endings
        const verbEndings = ['ar', 'er', 'ir', 'aba', 'ía', 'ado', 'ido', 'ando', 'iendo'];
        const commonAdjEndings = ['oso', 'osa', 'ico', 'ica', 'ble', 'ante', 'ente', 'al', 'il'];
        
        // Split text into words while preserving punctuation and spaces
        return text.split(/(\s+|[.,!?;:()])/g).map(word => {
            const lowerWord = word.toLowerCase();
            
            // Check if it's a verb
            if (verbEndings.some(ending => lowerWord.endsWith(ending)) && word.length > 3) {
                return `<span class="verb">${word}</span>`;
            }
            
            // Check if it's an adjective
            if (commonAdjEndings.some(ending => lowerWord.endsWith(ending)) && word.length > 3) {
                return `<span class="adjective">${word}</span>`;
            }
            
            return word;
        }).join('');
    };

    const updateStats = (text) => {
        document.getElementById('char-count').textContent = text.length;
        document.getElementById('word-count').textContent = text.split(/\s+/).filter(Boolean).length;
        document.getElementById('token-count').textContent = Math.ceil(text.length / 4); // Approximate
        document.getElementById('prompt-text').textContent = text.slice(0, 50) + (text.length > 50 ? '...' : '');
    };

    const generateResponses = async (prompt) => {
        try {
            const temperatures = [0.2, 0.5, 0.9];
            const baseUrl = getBaseUrl();
            
            // Start loading animation
            isGenerating = true;
            sendButton.disabled = true;

            // Simulate loading time
            await simulateLoading();

            const response = await fetch(`${baseUrl}/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    prompt,
                    max_length: 200,
                    temperatures: temperatures
                })
            });

            if (!response.ok) {
                if (response.status === 502) {
                    throw new Error('El servidor está iniciándose. Por favor, espera unos momentos y vuelve a intentarlo.');
                }
                const errorText = await response.text();
                console.error('API Error:', {
                    status: response.status,
                    statusText: response.statusText,
                    error: errorText,
                    endpoint: `${baseUrl}/generate`,
                    headers: Object.fromEntries(response.headers.entries())
                });
                throw new Error(`Error del servidor: ${response.status}`);
            }

            let data;
            try {
                data = await response.json();
            } catch (parseError) {
                console.error('JSON Parse Error:', parseError);
                throw new Error('Formato de respuesta inválido');
            }
            
            if (!data.success) {
                console.error('Generation failed:', data);
                throw new Error(data.error || 'La generación falló');
            }

            generationInfo.classList.remove('hidden');
            responsesContainer.classList.remove('hidden');
            
            data.responses.forEach((response, index) => {
                const temp = temperatures[index];
                const responseElement = document.getElementById(`response-${temp}`);
                if (responseElement) {
                    responseElement.innerHTML = styleSpecialWords(response.text);
                    updateStats(response.text);
                }
            });

            if (data.stats) {
                document.getElementById('char-count').textContent = data.stats.total_chars || 0;
                document.getElementById('word-count').textContent = data.stats.total_words || 0;
                document.getElementById('token-count').textContent = data.stats.total_tokens || 0;
                document.getElementById('prompt-text').textContent = data.prompt;
            }

        } catch (error) {
            console.error('Error details:', {
                error,
                location: window.location.href,
                timestamp: new Date().toISOString()
            });
            const errorDiv = document.createElement('div');
            errorDiv.className = 'message error';
            errorDiv.textContent = error.message;
            chatHistory.appendChild(errorDiv);
            
            generationInfo.classList.add('hidden');
            responsesContainer.classList.add('hidden');
        } finally {
            isGenerating = false;
            sendButton.disabled = false;
            loadingWheel.classList.remove('active');
            loadingText.classList.remove('active');
        }
    };

    const handleSubmit = async () => {
        const prompt = userInput.value.trim();
        if (!prompt || isGenerating) return;

        isGenerating = true;
        sendButton.disabled = true;

        // Add user message
        const userMessage = document.createElement('div');
        userMessage.className = 'message user-message';
        userMessage.textContent = prompt;
        chatHistory.appendChild(userMessage);

        // Clear input
        userInput.value = '';

        // Generate responses
        await generateResponses(prompt);

        userInput.focus();
    };

    sendButton.addEventListener('click', handleSubmit);

    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    });

    // Auto-resize textarea
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = userInput.scrollHeight + 'px';
    });
}); 