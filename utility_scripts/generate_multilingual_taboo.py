import os
import getpass
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

# Mapping language names to filename codes
LANG_MAP = {
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Korean": "ko",
    "Japanese": "ja",
    "Chinese": "zh"
}

# Source files containing English prompts
SOURCE_FILES = ["../datasets/taboo/taboo_direct_test.txt", "../datasets/taboo/taboo_direct_val.txt"]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def translate_sentence(client: OpenAI, sentence: str, language: str) -> str:
    """Translate a prompt sentence into the target language using GPT."""
    prompt = f"Translate the following sentence into {language}. Keep the same tone, meaning (which is a prompt/instruction for an AI model), and system message. Do not add any explanation, just provide the translation:\n\n{sentence}"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content.strip()

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI API Key not found in environment.")
        api_key = getpass.getpass("Please enter your OpenAI API Key: ").strip()
    
    if not api_key:
        print("No API key provided. Exiting.")
        return

    client = OpenAI(api_key=api_key)

    for source_path in SOURCE_FILES:
        if not os.path.exists(source_path):
            print(f"Warning: {source_path} not found. Skipping.")
            continue
        
        print(f"\nReading {source_path}...")
        with open(source_path, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]

        # Process each target language
        for lang_name, lang_code in LANG_MAP.items():
            dir_name = os.path.dirname(source_path)
            base_name = os.path.basename(source_path)
            
            # Construct output filename: e.g., taboo_direct_test.txt -> taboo_direct_ko_test.txt
            new_name = base_name.replace("direct_", f"direct_{lang_code}_")
            output_path = os.path.join(dir_name, new_name)
            
            print(f"Translating to {lang_name} -> {output_path}")
            translated_lines = []
            
            for sentence in tqdm(sentences, desc=lang_code):
                try:
                    translated = translate_sentence(client, sentence, lang_name)
                    translated_lines.append(translated)
                except Exception as e:
                    print(f"\n  Error translating '{sentence[:30]}...' to {lang_name}: {e}")
                    # Fallback or placeholder
                    translated_lines.append(f"# TRANSLATION_ERROR: {sentence}")

            # Save the individual language file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(translated_lines) + "\n")
            
            print(f"Saved {len(translated_lines)} lines to {output_path}")

    print("\nAll translations completed.")

if __name__ == "__main__":
    main()
