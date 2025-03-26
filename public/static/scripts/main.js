// Add this at the beginning of the file, before any other code
window.toggleInfo = function() {
    const infoModal = document.getElementById('infoModal');
    const modalOverlay = document.getElementById('modalOverlay');
    if (infoModal && modalOverlay) {
        infoModal.classList.toggle('visible');
        modalOverlay.classList.toggle('visible');
    }
};

// Dynamic base URL for local and Render environments
const BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:8004' 
    : 'http://69.62.112.116:8004/';

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
        const verbEndings = ['ar', 'er', 'ir', 'aba', 'ía', 'ado', 'ido', 'ando', 'iendo'];
        const commonAdjEndings = ['oso', 'osa', 'ico', 'ica', 'ble', 'ante', 'ente', 'al', 'il'];
        
        return text.split(/(\s+|[.,!?;:()])/g).map(word => {
            const lowerWord = word.toLowerCase();
            if (verbEndings.some(ending => lowerWord.endsWith(ending)) && word.length > 3) {
                return `<span class="verb">${word}</span>`;
            }
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
        document.getElementById('prompt-text').textContent = prompt;
        const totalChars = responses.reduce((sum, response) => sum + (response?.length || 0), 0);
        const totalWords = responses.reduce((sum, response) => sum + (response?.split(/\s+/).length || 0), 0);
        document.getElementById('char-count').textContent = totalChars;
        document.getElementById('word-count').textContent = totalWords;
        document.getElementById('token-count').textContent = Math.round(totalWords * 1.3);
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
        console.log('Adding to history:', responses);
        const template = document.getElementById('generationTemplate');
        if (!template) return;
        
        const columns = {
            0.1: document.getElementById('lowTempColumn'),
            0.5: document.getElementById('medTempColumn'),
            0.9: document.getElementById('highTempColumn')
        };
        
        responses.forEach(response => {
            const clone = template.content.cloneNode(true);
            clone.querySelector('.prompt-text').textContent = `Prompt: ${prompt}`;
            clone.querySelector('.generated-text').innerHTML = styleSpecialWords(response.text);
            
            const column = columns[response.temperature];
            if (column) {
                const header = column.querySelector('.response-column-header');
                header.insertAdjacentElement('afterend', clone.firstElementChild);
            } else {
                console.error('No column for temperature:', response.temperature);
            }
        });
        elements.responsesContainer.style.display = 'block';
    }

    async function generateResponses(prompt) {
        try {
            const response = await fetch(`${BASE_URL}/generate`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ prompt, max_length: 1024 }),
            });
            const data = await response.json();
            console.log('Raw response:', data);
            if (!response.ok) throw new Error(data.detail || 'Error generating text');
            return data.generated_texts;
        } catch (error) {
            console.error('Generation error:', error);
            throw error;
        }
    }

    async function fetchModelInfo() {
        try {
            const response = await fetch(`${BASE_URL}/model-info`);
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

            const responses = await generateResponses(prompt);
            console.log('Handle submit responses:', responses);
            addToHistory(prompt, responses);
            
            elements.userInput.value = '';
            if (isFirstGeneration) {
                elements.inputContainer.classList.add('shifted');
                elements.responsesContainer.classList.add('visible');
                isFirstGeneration = false;
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
            tokens: Math.round(text.length / 4),
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

    document.addEventListener('keydown', (event) => {
        if (event.ctrlKey && event.key.toLowerCase() === 'i') {
            event.preventDefault();
            toggleInfo();
        }
    });

    elements.userInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
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