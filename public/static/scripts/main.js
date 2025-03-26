// Add this at the beginning of the file, before any other code
window.toggleInfo = function() {
    const infoModal = document.getElementById('infoModal');
    const modalOverlay = document.getElementById('modalOverlay');
    if (infoModal && modalOverlay) {
        infoModal.classList.toggle('visible');
        modalOverlay.classList.toggle('visible');
    }
};

document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements
    const elements = {
        userInput: document.getElementById('userInput'),
        generateButton: document.getElementById('generateButton'),
        clearButton: document.getElementById('clearButton'),
        responsesContainer: document.getElementById('responsesContainer'),
        errorMessage: document.getElementById('errorMessage'),
        inputContainer: document.querySelector('.input-container'),
        infoModal: document.getElementById('infoModal'),
        modalOverlay: document.getElementById('modalOverlay'),
        generationStats: document.getElementById('generationStats'),
        chatHistory: document.getElementById('chatHistory'),
        generationInfo: document.getElementById('generationInfo'),
        loadingContainer: document.getElementById('loadingContainer'),
        loadingText: document.getElementById('loadingText')
    };

    // Only initialize if required elements exist
    if (!elements.userInput || !elements.generateButton) {
        console.error('Required elements not found');
        return;
    }

    // Hide elements only if they exist
    if (elements.generationInfo) {
        elements.generationInfo.style.display = 'none';
    }
    if (elements.responsesContainer) {
        elements.responsesContainer.style.display = 'none';
    }
    if (elements.loadingContainer) {
        elements.loadingContainer.style.display = 'none';
    }

    // Stats elements - only get if they exist
    const stats = {
        charCount: document.getElementById('char-count'),
        wordCount: document.getElementById('word-count'),
        tokenCount: document.getElementById('token-count'),
        promptText: document.getElementById('prompt-text')
    };

    const loadingMessages = [
        "Procesando tu solicitud...",
        "Analizando el contexto...",
        "Generando motores poéticos...",
        "Aplicando reglas lingüísticas...",
        "Refinando el texto generado..."
    ];

    let isGenerating = false;
    let isFirstGeneration = true;
    let generationHistory = [];

    const simulateLoading = async () => {
        if (!elements.loadingContainer || !elements.loadingText) return;

        const duration = Math.floor(Math.random() * (8000 - 5000 + 1)) + 5000;
        const intervalDuration = 2000;
        let messageIndex = 0;

        elements.loadingContainer.style.display = 'flex';

        return new Promise(resolve => {
            const updateMessage = () => {
                if (elements.loadingText) {
                    elements.loadingText.textContent = loadingMessages[messageIndex];
                    messageIndex = (messageIndex + 1) % loadingMessages.length;
                }
            };

            updateMessage();
            const messageInterval = setInterval(updateMessage, intervalDuration);

            setTimeout(() => {
                clearInterval(messageInterval);
                if (elements.loadingContainer) {
                    elements.loadingContainer.style.display = 'none';
                }
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

    function getModelInfo() {
        return JSON.parse(localStorage.getItem('modelInfo') || '{}');
    }

    function setModelInfo(info) {
        localStorage.setItem('modelInfo', JSON.stringify(info));
    }

    function updateModelInfo(info) {
        const modelInfo = getModelInfo();
        Object.assign(modelInfo, info);
        setModelInfo(modelInfo);
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
            updateModelInfo(modelInfo);
            document.getElementById('model-name').textContent = modelInfo.name || 'N/A';
            document.getElementById('model-vocab').textContent = modelInfo.vocab_size?.toLocaleString() || 'N/A';
            document.getElementById('model-params').textContent = modelInfo.total_params?.toLocaleString() || 'N/A';
            document.getElementById('model-layers').textContent = modelInfo.n_layer || 'N/A';
            document.getElementById('model-device').textContent = modelInfo.device || 'N/A';
            document.getElementById('model-context').textContent = modelInfo.n_positions?.toLocaleString() || 'N/A';
            document.getElementById('model-embedding').textContent = modelInfo.n_embd?.toLocaleString() || 'N/A';
        }
    }

    function addToHistory(prompt, responses) {
        const template = document.getElementById('generationTemplate');
        if (!template) return;
        
        // Get column elements
        const columns = {
            low: document.getElementById('lowTempColumn'),
            med: document.getElementById('medTempColumn'),
            high: document.getElementById('highTempColumn')
        };
        
        // Add each response to its corresponding column
        responses.forEach(response => {
            const clone = template.content.cloneNode(true);
            clone.querySelector('.prompt-text').textContent = `Prompt: ${prompt}`;
            clone.querySelector('.generated-text').textContent = response.text;
            
            // Determine which column to use based on temperature
            let column;
            if (response.temperature === 0.1) column = columns.low;
            else if (response.temperature === 0.5) column = columns.med;
            else if (response.temperature === 0.9) column = columns.high;
            
            if (column) {
                // Insert after the header
                const header = column.querySelector('.response-column-header');
                if (header) {
                    header.insertAdjacentElement('afterend', clone.firstElementChild);
                } else {
                    column.appendChild(clone);
                }
            }
        });
    }

    async function generateResponses(prompt) {
        try {
            const response = await fetch('http://localhost:8004/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: prompt,
                    max_length: 512
                }),
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.detail || 'Error generating text');
            }

            return data;
        } catch (error) {
            console.error('Generation error:', error);
            throw error;
        }
    }

    async function fetchModelInfo() {
        try {
            const response = await fetch('http://localhost:8004/model-info');
            if (!response.ok) {
                throw new Error('Failed to fetch model info');
            }
            const modelInfo = await response.json();
            updateModelInfo(modelInfo);
            displayModelInfo(modelInfo);
        } catch (error) {
            console.error('Error fetching model info:', error);
        }
    }

    function displayModelInfo(info) {
        if (!info) return;
        
        const elements = {
            name: document.getElementById('model-name'),
            vocab: document.getElementById('model-vocab'),
            params: document.getElementById('model-params'),
            layers: document.getElementById('model-layers'),
            device: document.getElementById('model-device'),
            context: document.getElementById('model-context'),
            embedding: document.getElementById('model-embedding')
        };

        if (elements.name) elements.name.textContent = info.name || 'ArPoChat';
        if (elements.vocab) elements.vocab.textContent = info.vocab_size?.toLocaleString() || '50,257';
        if (elements.params) elements.params.textContent = info.total_params?.toLocaleString() || '124,439,808';
        if (elements.layers) elements.layers.textContent = info.layers || '12';
        if (elements.device) elements.device.textContent = info.device || 'CPU';
        if (elements.context) elements.context.textContent = info.max_context || '1024';
        if (elements.embedding) elements.embedding.textContent = info.embedding_size || '768';
    }

    async function handleSubmit() {
        if (!elements.userInput || !elements.generateButton) return;

        const prompt = elements.userInput.value.trim();
        if (!prompt) return;

        try {
            elements.generateButton.disabled = true;
            elements.generateButton.textContent = 'Generando...';
            elements.errorMessage?.classList.remove('visible');
            elements.loadingContainer.style.display = 'flex';

            const response = await generateResponses(prompt);
            
            if (response?.generated_texts) {
                addToHistory(prompt, response.generated_texts);
                elements.userInput.value = '';
                
                // Update stats
                const totalChars = response.generated_texts.reduce((sum, item) => sum + item.text.length, 0);
                const totalWords = response.generated_texts.reduce((sum, item) => sum + item.text.split(/\s+/).length, 0);
                
                if (stats.charCount) stats.charCount.textContent = totalChars;
                if (stats.wordCount) stats.wordCount.textContent = totalWords;
                if (stats.tokenCount) stats.tokenCount.textContent = Math.round(totalWords * 1.3);
                if (stats.promptText) stats.promptText.textContent = prompt;
                
                if (isFirstGeneration) {
                    elements.inputContainer.classList.add('shifted');
                    elements.responsesContainer.classList.add('visible');
                    isFirstGeneration = false;
                }
            }
        } catch (error) {
            showError(error.message);
        } finally {
            elements.generateButton.disabled = false;
            elements.generateButton.textContent = 'Generar';
            elements.loadingContainer.style.display = 'none';
        }
    }

    function handleClear() {
        if (elements.userInput && elements.generateButton) {
            elements.userInput.value = '';
            elements.generateButton.disabled = true;
        }
    }

    function updateGenerationStats(prompt, text) {
        if (!elements.generationStats) return;
        
        const stats = {
            characters: text.length,
            words: text.split(/\s+/).length,
            tokens: Math.round(text.length / 4), // Rough estimate
        };
        
        elements.generationStats.innerHTML = `
            <p>Caracteres: ${stats.characters}</p>
            <p>Palabras: ${stats.words}</p>
            <p>Tokens (est.): ${stats.tokens}</p>
            <p>Último prompt: ${prompt}</p>
        `;
    }

    function showError(message) {
        if (!elements.errorMessage) return;
        elements.errorMessage.textContent = message;
        elements.errorMessage.classList.add('visible');
    }

    function scrollToLatest() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    }

    // Event Listeners
    elements.userInput.addEventListener('input', (event) => {
        if (elements.generateButton) {
            elements.generateButton.disabled = !event.target.value.trim();
        }
    });

    elements.generateButton.addEventListener('click', handleSubmit);

    if (elements.clearButton) {
        elements.clearButton.addEventListener('click', handleClear);
    }

    // Add keyboard event listeners
    document.addEventListener('keydown', (event) => {
        // Ctrl+I for info panel
        if (event.ctrlKey && event.key.toLowerCase() === 'i') {
            event.preventDefault(); // Prevent default browser behavior
            toggleInfo();
        }
    });

    // Add Enter key handler for input
    elements.userInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault(); // Prevent newline
            if (!elements.generateButton.disabled) {
                handleSubmit();
            }
        }
    });

    // Initialize
    elements.generateButton.disabled = true;
    elements.userInput.focus();

    // Fetch model info when page loads
    fetchModelInfo();
}); 