from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, pipeline
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from tabulate import tabulate

class MultilingualSummarizer:
    def __init__(self):
        """Initialize the summarizer and translator"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize BART summarizer
        print("Loading BART model...")
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn",
            device=0 if self.device == "cuda" else -1
        )
        
        # Initialize M2M100 translator
        print("Loading M2M100 model...")
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
        """Generate a summary of the input text using BART"""
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
        """Translate text to target language using M2M100"""
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
        """Process text: first summarize, then translate the summary"""
        try:
            # Generate summary
            print("\nGenerating English summary...")
            summary = self.generate_summary(text)
            if not summary:
                return None
            
            # Generate translations for all languages
            translations = {}
            for lang_name, lang_code in self.lang_codes.items():
                if lang_name != "English":  # Skip English as it's our source
                    print(f"Translating to {lang_name}...")
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

class EvaluationMetrics:
    def __init__(self):
        """Initialize ROUGE scorer"""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
        
    def compute_rouge(self, reference, candidate):
        """Compute ROUGE-2 score"""
        scores = self.rouge_scorer.score(reference, candidate)
        return scores['rouge2'].fmeasure
        
    def evaluate_summaries(self, references, candidates):
        """Evaluate multiple summaries and return results as DataFrame"""
        results = []
        for lang, (ref, cand) in zip(references.keys(), zip(references.values(), candidates.values())):
            rouge_score = self.compute_rouge(ref, cand)
            
            results.append({
                'language': lang,
                'reference': ref,
                'candidate': cand,
                'rouge2': rouge_score
            })
            
        return pd.DataFrame(results)
        
    def plot_scores(self, df):
        """Plot ROUGE-2 scores"""
        plt.figure(figsize=(8, 5))
        df.plot(x='language', y='rouge2', kind='bar', color='skyblue')
        plt.title('ROUGE-2 Scores')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig('evaluation_scores.png')
        plt.close()

def print_results_table(df):
    """Print results in a nicely formatted table"""
    # Format ROUGE-2 scores to 3 decimal places
    df_display = df.copy()
    df_display['rouge2'] = df_display['rouge2'].map('{:.3f}'.format)
    
    # Print scores table
    print("\n" + "="*80)
    print("ROUGE-2 SCORES")
    print("="*80)
    print(tabulate(df_display[['language', 'rouge2']], 
                  headers='keys', tablefmt='pretty', showindex=False))
    
    # Print summaries with ROUGE-2 scores
    print("\n" + "="*80)
    print("DETAILED RESULTS BY LANGUAGE")
    print("="*80)
    for _, row in df.iterrows():
        print(f"\n{row['language']} (ROUGE-2: {df_display.loc[_, 'rouge2']})")
        print("-"*40)
        print("Reference Summary:")
        print(row['reference'])
        print("\nGenerated Summary:")
        print(row['candidate'])
        print("-"*40)

def main():
    # Download required NLTK data
    print("Downloading required NLTK data...")
    nltk.download('punkt')
    
    # Initialize components
    print("\nInitializing components...")
    summarizer = MultilingualSummarizer()
    evaluator = EvaluationMetrics()
    
    # Use a single source text for all languages
    source_text = """Videos that say approved vaccines are dangerous and cause autism, cancer or infertility are among those that will be taken down, the company said. The policy includes the termination of accounts of anti-vaccine influencers. Tech giants have been criticised for not doing more to counter false health information on their sites. In July, US President Joe Biden said social media platforms were largely responsible for people's scepticism in getting vaccinated by spreading misinformation, and appealed for them to address the issue. YouTube, which is owned by Google, said 130,000 videos were removed from its platform since last year, when it implemented a ban on content spreading misinformation about Covid vaccines. In a blog post, the company said it had seen false claims about Covid jabs "spill over into misinformation about vaccines in general". The new policy covers long-approved vaccines, such as those against measles or hepatitis B. "We're expanding our medical misinformation policies on YouTube with new guidelines on currently administered vaccines that are approved and confirmed to be safe and effective by local health authorities and the WHO," the post said, referring to the World Health Organization."""
    
    # Reference summaries for each language
    reference_summaries = {
        'English': "YouTube is removing videos spreading vaccine misinformation and terminating accounts of anti-vaccine influencers. The platform has removed 130,000 videos since last year and is expanding its policies to cover all approved vaccines.",
        'Spanish': "YouTube está eliminando videos que difunden información errónea sobre vacunas y terminando cuentas de influencers antivacunas. La plataforma ha eliminado 130,000 videos desde el año pasado y está expandiendo sus políticas para cubrir todas las vacunas aprobadas.",
        'French': "YouTube supprime les vidéos diffusant de fausses informations sur les vaccins et ferme les comptes des influenceurs anti-vaccins. La plateforme a supprimé 130 000 vidéos depuis l'année dernière et étend ses politiques à tous les vaccins approuvés.",
        'Korean': "YouTube는 백신 오보를 퍼뜨리는 동영상을 삭제하고 반백신 인플루언서의 계정을 종료하고 있습니다. 이 플랫폼은 지난해부터 130,000개의 동영상을 삭제했으며 승인된 모든 백신을 대상으로 정책을 확대하고 있습니다.",
        'Portuguese': "O YouTube está removendo vídeos que espalham desinformação sobre vacinas e encerrando contas de influenciadores antivacinas. A plataforma removeu 130.000 vídeos desde o ano passado e está expandindo suas políticas para cobrir todas as vacinas aprovadas.",
        'Punjabi': "ਯੂਟਿਊਬ ਵੈਕਸੀਨ ਬਾਰੇ ਗਲਤ ਜਾਣਕਾਰੀ ਫੈਲਾਉਣ ਵਾਲੀਆਂ ਵੀਡੀਓਜ਼ ਨੂੰ ਹਟਾ ਰਿਹਾ ਹੈ ਅਤੇ ਐਂਟੀ-ਵੈਕਸੀਨ ਇਨਫਲੂਐਂਸਰਾਂ ਦੇ ਖਾਤਿਆਂ ਨੂੰ ਬੰਦ ਕਰ ਰਿਹਾ ਹੈ। ਪਲੇਟਫਾਰਮ ਨੇ ਪਿਛਲੇ ਸਾਲ ਤੋਂ 130,000 ਵੀਡੀਓਜ਼ ਹਟਾਈਆਂ ਹਨ ਅਤੇ ਸਾਰੀਆਂ ਮਨਜ਼ੂਰ ਵੈਕਸੀਨਾਂ ਨੂੰ ਕਵਰ ਕਰਨ ਲਈ ਆਪਣੀਆਂ ਨੀਤੀਆਂ ਦਾ ਵਿਸਥਾਰ ਕਰ ਰਿਹਾ ਹੈ।"
    }
    
    print("\nProcessing text in all languages...")
    print(f"Source text: {source_text[:200]}...")
    
    # Process the text
    processed = summarizer.process_text(source_text)
    if processed:
        results = []
        for lang_name in ['English', 'Spanish', 'French', 'Korean', 'Portuguese', 'Punjabi']:
            results.append({
                'language': lang_name,
                'reference': reference_summaries[lang_name],
                'candidate': processed['translations'][lang_name]
            })
        
        # Evaluate results
        print("\nEvaluating summaries...")
        evaluation_results = evaluator.evaluate_summaries(
            {r['language']: r['reference'] for r in results},
            {r['language']: r['candidate'] for r in results}
        )
        
        # Print results in a nice format
        print_results_table(evaluation_results)
        
        # Generate and save plots
        print("\nGenerating score plots...")
        evaluator.plot_scores(evaluation_results)
        print("Plots saved as 'evaluation_scores.png'")
        
        # Save results to CSV
        output_path = "evaluation_results.csv"
        evaluation_results.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    else:
        print("Failed to process the text.")

if __name__ == "__main__":
    main() 