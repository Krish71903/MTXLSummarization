from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, pipeline
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
import matplotlib.pyplot as plt

class EvaluationMetrics:
    def __init__(self):
        """Initialize ROUGE and BLEU scorers"""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method4
        
    def compute_rouge(self, reference, candidate):
        """Compute ROUGE scores"""
        scores = self.rouge_scorer.score(reference, candidate)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
        
    def compute_bleu(self, reference, candidate):
        """Compute BLEU score"""
        reference_tokens = [reference.split()]
        candidate_tokens = candidate.split()
        return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=self.smoothing_function)
        
    def evaluate_summaries(self, references, candidates):
        """Evaluate multiple summaries and return results as DataFrame"""
        results = []
        for lang, (ref, cand) in zip(references.keys(), zip(references.values(), candidates.values())):
            rouge_scores = self.compute_rouge(ref, cand)
            bleu_score = self.compute_bleu(ref, cand)
            
            results.append({
                'language': lang,
                'reference': ref,
                'candidate': cand,
                'rouge1': rouge_scores['rouge1'],
                'rouge2': rouge_scores['rouge2'],
                'rougeL': rouge_scores['rougeL'],
                'bleu': bleu_score
            })
            
        return pd.DataFrame(results)
        
    def plot_scores(self, df):
        """Plot ROUGE and BLEU scores"""
        plt.figure(figsize=(15, 5))
        
        # Plot ROUGE scores
        plt.subplot(1, 2, 1)
        df.plot(x='language', y=['rouge1', 'rouge2', 'rougeL'], kind='bar')
        plt.title('ROUGE Scores')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Plot BLEU scores
        plt.subplot(1, 2, 2)
        df.plot(x='language', y='bleu', kind='bar', color='skyblue')
        plt.title('BLEU Scores')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.show()

class MultilingualSummarizer:
    def __init__(self):
        """
        Initialize the summarizer and translator
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize BART summarizer
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn",
            device=0 if self.device == "cuda" else -1
        )
        
        # Initialize M2M100 translator
        model_name = "facebook/m2m100_418M"
        self.translator_model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        self.translator_tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        
        if self.device == "cuda":
            self.translator_model = self.translator_model.to("cuda")
        
        # Language codes for M2M100
        self.lang_codes = {
            'English': 'en',
            'Spanish': 'es',
            'French': 'fr',
            'Korean': 'ko',
            'Portuguese': 'pt',
            'Punjabi': 'pa'
        }
        
    def generate_summary(self, text, max_length=90, min_length=50):
        """
        Generate a summary of the input text using BART
        """
        try:
            # Clean the input text
            text = " ".join(text.split())
            
            # Generate summary
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Error during summarization: {str(e)}")
            return None
        
    def translate_text(self, text, target_lang_code):
        """
        Translate text to target language using M2M100
        """
        try:
            # Set source language to English
            self.translator_tokenizer.src_lang = "en"
            
            # Encode the text
            encoded = self.translator_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            if self.device == "cuda":
                encoded = {k: v.to("cuda") for k, v in encoded.items()}
            
            # Generate translation
            generated_tokens = self.translator_model.generate(
                **encoded,
                forced_bos_token_id=self.translator_tokenizer.get_lang_id(target_lang_code),
                max_length=200,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True
            )
            
            # Decode the generated tokens
            translated = self.translator_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            return translated[0]
        except Exception as e:
            print(f"Error translating to {target_lang_code}: {str(e)}")
            return f"Translation failed for {target_lang_code}"
        
    def process_text(self, text):
        """
        Process text: first summarize, then translate the summary
        
        Args:
            text (str): Input text to process
            
        Returns:
            dict: Summary and translations in all target languages
        """
        try:
            # Generate summary
            summary = self.generate_summary(text)
            if not summary:
                return None
            
            # Generate translations for all languages
            translations = {}
            for lang_name, lang_code in self.lang_codes.items():
                if lang_name != "English":  # Skip English as it's our source
                    translated = self.translate_text(summary, lang_code)
                    translations[lang_name] = translated
                else:
                    translations[lang_name] = summary
            
            return {
                'summary': summary,
                'translations': translations
            }
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            return None

def main():
    # Example text and reference summaries
    text = """Bitcoin uses more electricity annually than the whole of Argentina, analysis by Cambridge University suggests. 
    Mining for the cryptocurrency is power-hungry, involving heavy computer calculations to verify transactions. 
    Cambridge researchers say it consumes around 121.36 terawatt-hours (TWh) a year - and is unlikely to fall unless 
    the value of the currency slumps. Critics say electric-car firm Tesla's decision to invest heavily in Bitcoin undermines 
    its environmental image."""
    
    # Reference summaries for evaluation
    reference_summaries = {
        'English': "Bitcoin's annual electricity consumption exceeds Argentina's, according to Cambridge University. The cryptocurrency's mining process requires significant energy, consuming 121.36 TWh yearly.",
        'Spanish': "El consumo anual de electricidad de Bitcoin supera al de Argentina, según la Universidad de Cambridge. El proceso de minería de la criptomoneda requiere energía significativa, consumiendo 121.36 TWh al año.",
        'French': "La consommation annuelle d'électricité de Bitcoin dépasse celle de l'Argentine, selon l'Université de Cambridge. Le processus de minage de la cryptomonnaie nécessite une énergie importante, consommant 121.36 TWh par an.",
        'Korean': "케임브리지 대학에 따르면 비트코인의 연간 전력 소비량이 아르헨티나를 초과합니다. 암호화폐의 채굴 과정은 상당한 에너지를 필요로 하며, 연간 121.36 TWh를 소비합니다.",
        'Portuguese': "O consumo anual de eletricidade do Bitcoin excede o da Argentina, de acordo com a Universidade de Cambridge. O processo de mineração da criptomoeda requer energia significativa, consumindo 121.36 TWh por ano.",
        'Punjabi': "ਕੈਮਬ੍ਰਿਜ ਯੂਨੀਵਰਸਿਟੀ ਦੇ ਅਨੁਸਾਰ, ਬਿਟਕੋਇਨ ਦੀ ਸਾਲਾਨਾ ਬਿਜਲੀ ਦੀ ਖਪਤ ਅਰਜਨਟੀਨਾ ਨਾਲੋਂ ਵੱਧ ਹੈ। ਕ੍ਰਿਪਟੋਕਰੰਸੀ ਦੀ ਮਾਈਨਿੰਗ ਪ੍ਰਕਿਰਿਆ ਨੂੰ ਵਿਸ਼ੇਸ਼ ਊਰਜਾ ਦੀ ਲੋੜ ਹੁੰਦੀ ਹੈ, ਸਾਲਾਨਾ 121.36 TWh ਦੀ ਖਪਤ ਕਰਦੀ ਹੈ।"
    }
    
    # Initialize the processor and evaluator
    print("Initializing summarizer, translator, and evaluator...")
    processor = MultilingualSummarizer()
    evaluator = EvaluationMetrics()
    
    print("\nMultilingual Summarization and Translation Results:")
    print("-" * 80)
    
    print("\nOriginal text:")
    print(text)
    
    results = processor.process_text(text)
    if results:
        print("\nEnglish Summary:")
        print(results['summary'])
        print("\nTranslations of the Summary:")
        
        # Print translations in order
        for lang_name in ['English', 'Spanish', 'French', 'Korean', 'Portuguese', 'Punjabi']:
            print(f"\n{lang_name}:")
            print(results['translations'][lang_name])
            print("-" * 80)
        
        # Evaluate summaries
        print("\nEvaluating summaries...")
        evaluation_results = evaluator.evaluate_summaries(reference_summaries, results['translations'])
        
        # Print evaluation results
        print("\nEvaluation Results:")
        print(evaluation_results[['language', 'rouge1', 'rouge2', 'rougeL', 'bleu']])
        
        # Plot scores
        print("\nGenerating score plots...")
        evaluator.plot_scores(evaluation_results)
        
        # Save results to CSV
        output_path = "evaluation_results.csv"
        evaluation_results.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main() 