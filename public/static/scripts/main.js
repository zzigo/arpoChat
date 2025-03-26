document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const chatHistory = document.getElementById('chatHistory');
    const responsesContainer = document.getElementById('responsesContainer');
    const generationInfo = document.getElementById('generationInfo');
    const loadingContainer = document.getElementById('loadingContainer');
    const loadingText = document.getElementById('loadingText');

    // Hide generation info and responses by default
    generationInfo.style.display = 'none';
    responsesContainer.style.display = 'none';
    loadingContainer.style.display = 'none';

    // Stats elements
    const charCount = document.getElementById('char-count');
    const wordCount = document.getElementById('word-count');
    const tokenCount = document.getElementById('token-count');
    const promptText = document.getElementById('prompt-text');

    const loadingMessages = [
        "Procesando tu solicitud...",
        "Analizando el contexto...",
        "Generando motores poéticos...",
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

        loadingContainer.style.display = 'flex';

        return new Promise(resolve => {
            const updateMessage = () => {
                loadingText.textContent = loadingMessages[messageIndex];
                messageIndex = (messageIndex + 1) % loadingMessages.length;
            };

            updateMessage(); // Show first message immediately
            const messageInterval = setInterval(updateMessage, intervalDuration);

            setTimeout(() => {
                clearInterval(messageInterval);
                loadingContainer.style.display = 'none';
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

    function toggleInfo() {
        const content = document.getElementById('infoContent');
        const icon = document.querySelector('.toggle-icon');
        content.classList.toggle('expanded');
        icon.style.transform = content.classList.contains('expanded') ? 'rotate(180deg)' : 'rotate(0deg)';
    }

    function updateStats(responses, prompt, modelInfo) {
        // Update prompt text
        document.getElementById('prompt-text').textContent = prompt;
        
        // Calculate total characters and words from all responses
        const totalChars = responses.reduce((sum, response) => sum + (response?.length || 0), 0);
        const totalWords = responses.reduce((sum, response) => sum + (response?.split(/\s+/).length || 0), 0);
        
        // Update stats
        document.getElementById('char-count').textContent = totalChars;
        document.getElementById('word-count').textContent = totalWords;
        document.getElementById('token-count').textContent = Math.round(totalWords * 1.3); // Rough estimate
        
        // Update model info if available
        if (modelInfo) {
            document.getElementById('model-name').textContent = modelInfo.name || 'N/A';
            document.getElementById('model-vocab').textContent = modelInfo.vocab_size?.toLocaleString() || 'N/A';
            document.getElementById('model-params').textContent = modelInfo.total_params?.toLocaleString() || 'N/A';
            document.getElementById('model-layers').textContent = modelInfo.n_layer || 'N/A';
            document.getElementById('model-device').textContent = modelInfo.device || 'N/A';
            document.getElementById('model-context').textContent = modelInfo.n_positions?.toLocaleString() || 'N/A';
            document.getElementById('model-embedding').textContent = modelInfo.n_embd?.toLocaleString() || 'N/A';
        }
    }

    async function generateResponses(prompt) {
        try {
            const temperatures = [0.1, 0.5, 0.9];
            const responses = [];
            let modelInfo = null;

            // Generate responses for each temperature
            for (const temp of temperatures) {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                        prompt,
                        temperature: temp,
                        custom_prompt: "make an argentine poem departing from prompt, using each phrase as a module with multiple interconnections with next verses and words, semantic, syntatic, phonetic, oneiric, narrative , or nonsense. Give a sense of unity to each strophe entangling meanings , alternating between logical and unreallistic abstract symbolics"
                    }),
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                if (data.status === 'success' && data.responses && data.responses.length > 0) {
                    responses.push(data.responses[0]); // Take the first response from the array
                    modelInfo = data.model_info; // Store model info from the first response
                } else {
                    throw new Error('No valid response received from the API');
                }
            }

            return { responses, modelInfo };
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }

    const handleSubmit = async () => {
        const prompt = userInput.value.trim();
        if (!prompt) return;

        // Disable input and button
        userInput.disabled = true;
        sendButton.disabled = true;

        // Show loading state
        loadingContainer.style.display = 'flex';
        loadingText.textContent = 'Generando respuestas...';

        try {
            // Add user message to chat history
            addMessageToChat('user', prompt);

            // Generate responses for each temperature
            const { responses, modelInfo } = await generateResponses(prompt);

            // Show generation info and responses
            generationInfo.style.display = 'block';
            responsesContainer.style.display = 'grid';

            // Update responses
            updateResponses(responses);

            // Update stats and model info
            updateStats(responses, prompt, modelInfo);

            // Clear input
            userInput.value = '';
        } catch (error) {
            console.error('Error:', error);
            addMessageToChat('error', 'Error al generar respuestas. Por favor, intenta de nuevo.');
        } finally {
            // Re-enable input and button
            userInput.disabled = false;
            sendButton.disabled = false;
            userInput.focus();

            // Hide loading state
            loadingContainer.style.display = 'none';
        }
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

    function addMessageToChat(type, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        messageDiv.textContent = content;
        chatHistory.appendChild(messageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function updateResponses(responses) {
        const responseElements = document.querySelectorAll('.response-content');
        responseElements.forEach((element, index) => {
            if (responses[index]) {
                element.textContent = responses[index];
            } else {
                element.textContent = 'No se pudo generar una respuesta.';
            }
        });
    }

    // Initialize
    userInput.focus();
}); 