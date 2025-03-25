document.addEventListener('DOMContentLoaded', () => {
    const promptInput = document.getElementById('promptInput');
    const generateBtn = document.getElementById('generateBtn');
    const responseContainer = document.getElementById('response');
    const loading = document.getElementById('loading');

    // Auto-resize textarea
    promptInput.addEventListener('input', () => {
        promptInput.style.height = 'auto';
        promptInput.style.height = promptInput.scrollHeight + 'px';
    });

    // Handle generation
    generateBtn.addEventListener('click', async () => {
        const prompt = promptInput.value.trim();
        if (!prompt) return;

        try {
            loading.classList.remove('hidden');
            responseContainer.innerHTML = '';
            
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt,
                    temperature: 0.7,
                    max_length: 200,
                    num_return_sequences: 1
                })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            
            // Format and display the response
            const formattedResponse = formatResponse(data);
            responseContainer.innerHTML = formattedResponse;
            
        } catch (error) {
            console.error('Error:', error);
            responseContainer.innerHTML = `<div class="error">Lo siento, hubo un error generando la respuesta.</div>`;
        } finally {
            loading.classList.add('hidden');
        }
    });

    // Handle Enter key
    promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            generateBtn.click();
        }
    });

    function formatResponse(data) {
        if (!data.generated_text) {
            return `<div class="error">No se pudo generar texto.</div>`;
        }
        
        return `
            <div class="response-header">
                <div class="prompt-used">Prompt: "${data.prompt || ''}"</div>
            </div>
            <div class="generated-text">${data.generated_text}</div>
        `;
    }
}); 