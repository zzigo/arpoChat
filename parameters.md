Increased max_length to 2000 tokens for longer poems
Set min_length to 500 tokens to ensure substantial responses
Adjusted temperature to 0.7 for a better balance between creativity and coherence
Increased top_k to 100 and top_p to 0.9 for more diverse vocabulary
Increased repetition_penalty to 1.5 to prevent repetitive phrases
Disabled early_stopping to allow full-length generation
Added no_repeat_ngram_size of 3 to prevent repetition of longer phrases
Increased length_penalty to 1.2 to encourage longer sequences
Added proper memory management with CUDA cache clearing
Improved text cleanup and punctuation handling
These changes should result in:
A cleaner, more organized UI with foldable sections
Longer, more coherent responses (300-500 words minimum)
Better diversity in vocabulary and structure
Less repetition and more natural flow
Proper memory management for stable generation
