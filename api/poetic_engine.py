import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoeticEngine:
    def __init__(self, model_name="gpt2", device=None):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.initialize_model()

    def initialize_model(self):
        """Initialize the model and tokenizer."""
        try:
            logger.info(f"Initializing model {self.model_name} on {self.device}")
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name).to(self.device)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
                
            logger.info("Model initialization successful")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def prepare_dataset(self, text_file_path):
        """Prepare the dataset for training."""
        try:
            logger.info(f"Preparing dataset from {text_file_path}")
            dataset = TextDataset(
                tokenizer=self.tokenizer,
                file_path=text_file_path,
                block_size=128
            )
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            return dataset, data_collator
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            raise

    def train(self, train_file_path, output_dir, num_epochs=3, batch_size=4, learning_rate=5e-5):
        """Train the model on the provided dataset."""
        try:
            dataset, data_collator = self.prepare_dataset(train_file_path)
            
            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                save_steps=500,
                save_total_limit=2,
                logging_dir=os.path.join(output_dir, 'logs'),
                logging_steps=100,
                evaluation_strategy="steps",
                eval_steps=500,
                warmup_steps=500,
                weight_decay=0.01,
                fp16=torch.cuda.is_available()
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=dataset
            )

            logger.info("Starting training...")
            trainer.train()
            
            # Save the model and tokenizer
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"Training completed. Model saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def save_model(self, output_dir):
        """Save the model and tokenizer to the specified directory."""
        try:
            logger.info(f"Saving model to {output_dir}")
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path):
        """Load a trained model from the specified directory."""
        try:
            logger.info(f"Loading model from {model_path}")
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
                
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    engine = PoeticEngine()
    
    # Training example
    train_file = "path/to/your/poetry/dataset.txt"
    output_dir = "path/to/save/model"
    
    engine.train(
        train_file_path=train_file,
        output_dir=output_dir,
        num_epochs=3,
        batch_size=4,
        learning_rate=5e-5
    ) 