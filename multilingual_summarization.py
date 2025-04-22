from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class MultilingualSummarizer:
    def __init__(self):
        """
        Initialize the summarizer and translator
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize BART summarizer - known for high quality summaries
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn",
            device=0 if self.device == "cuda" else -1
        )
        
        # Initialize the translation pipeline
        self.translator = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-en-es",
            device=0 if self.device == "cuda" else -1
        )
        
    def summarize_and_translate(self, text, max_length=150, min_length=50):
        """
        Generate a summary of the input text and translate it to Spanish
        
        Args:
            text (str): Input text to summarize
            max_length (int): Maximum length of the summary in tokens
            min_length (int): Minimum length of the summary in tokens
            
        Returns:
            dict: Original summary and translated summary
        """
        try:
            # Clean the input text - remove extra whitespace and newlines
            text = " ".join(text.split())
            
            # Generate summary with BART
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                length_penalty=2.0,  # Encourage longer summaries
                num_beams=4,  # Beam search for better quality
                early_stopping=True
            )
            original_summary = summary[0]['summary_text']
            
            # Translate the summary to Spanish
            translated = self.translator(original_summary)[0]['translation_text']
            
            return {
                'original_summary': original_summary,
                'translated_summary': translated
            }
        except Exception as e:
            print(f"Error during summarization or translation: {str(e)}")
            return None

def main():
    # Example texts in different languages
    texts = {
        "en": """Bitcoin uses more electricity annually than the whole of Argentina, analysis by Cambridge University suggests. 
        "Mining" for the cryptocurrency is power-hungry, involving heavy computer calculations to verify transactions. 
        Cambridge researchers say it consumes around 121.36 terawatt-hours (TWh) a year - and is unlikely to fall unless 
        the value of the currency slumps. Critics say electric-car firm Tesla's decision to invest heavily in Bitcoin undermines 
        its environmental image. The cryptocurrency has also been criticized by environmental groups for using vast amounts of energy 
        and accelerating climate change. Bitcoin's value hit a record high of $48,000 following Tesla's announcement that it had 
        invested $1.5bn in the cryptocurrency and pledged to start accepting it as payment for cars."""
    }
    
    # Initialize the summarizer
    print("Initializing summarizer and translator...")
    summarizer = MultilingualSummarizer()
    
    print("\nCross-lingual Summarization Results:")
    print("-" * 80)
    
    # Process each text
    for lang, text in texts.items():
        print(f"\nOriginal text ({lang}):")
        print(text)
        print("\nSummaries:")
        results = summarizer.summarize_and_translate(text)
        if results:
            print("\nEnglish Summary:", results['original_summary'])
            print("\nSpanish Translation:", results['translated_summary'])
        print("-" * 80)

if __name__ == "__main__":
    main() 