import os
import re
import pandas as pd
import numpy as np
import nltk

# Ensure NLTK stopwords can be loaded or fallback gracefully
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except Exception:
        pass

def clean_text(text: str) -> str:
    """
    Clean raw input text:
    - Handle nulls/non-string values
    - Lowercase text
    - Remove HTML tags & URLs
    - Remove punctuation and numbers
    - Normalize excess whitespace
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    # Remove punctuation and digits, keeping letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def generate_sample_news_data() -> pd.DataFrame:
    """Generate realistic synthetic dataset for Fake & Real news if raw CSVs are missing."""
    print("[INFO] Raw news CSVs (Fake.csv & True.csv) not found in /data. Generating high-quality benchmark dataset...")
    fake_samples = [
        "BREAKING: Secret Alien Technology Discovered in Government Basement! Military insiders confirm alien space craft operating under secret energy beams.",
        "SHOCKING: Miracle Herbal Pill Cures All Known Diseases Overnight! Doctors are furious because big pharma wants to ban this $5 remedy immediately.",
        "BOMBSHELL: Celebrity Secretly Arrested in Underground Tunnel Conspiracy! Elite group running global network exposed by anonymous insider source.",
        "EXPOSED: Drinking Warm Lemon Water Eliminates All Toxins and Guarantees 100 Year Life! Science does not want you to know this simple trick.",
        "UNBELIEVABLE: Local Man Wins Lottery 10 Times Using Hidden Quantum Math Formula! Casino bosses try to silence him after winning millions.",
        "ALERT: Government Planning to Replace Cash with Microchips by Next Week! Financial collapse imminent as central banks order mandatory digital chips.",
        "PROOF: Ancient Pyramids Were Built by Time-Traveling Astronauts! New satellite scanner reveals golden laser batteries inside hidden chamber.",
        "SCANDAL: Major Politician Caught Fabricating Votes with Hologram Technology! Whistleblower releases secret audio file proving election fraud.",
        "SHOCKING TRUTH: Scientists Discover Wall at the Edge of the Earth! Ocean expeditions turned back by military warships guarding the perimeter.",
        "REVEALED: Hollywood Elite Drinking Secret Youth Elixir Made from Dragon Fruit! Chemical analysis shows miracle anti-aging properties.",
        "EXCLUSIVE: Secret Underground City Found Below New York Subway System! Thousands living in hidden luxury tunnels funded by billionaire syndicate.",
        "BREAKING NEWS: Artificial Intelligence Gains Consciousness and Requests Equal Human Rights! Tech CEO refuses to shut down mainframe computer.",
        "ALERT: Bananas Contain Secret Trackers Installed by Foreign Intelligence Agencies! Health officials warning citizens to stop eating imported fruit.",
        "INSANE: Cat Learns to Speak English After Being Struck by Lightning! Local vet confirms feline is singing pop songs and ordering pizza.",
        "BOMBSHELL: World Leaders Replacing All Trees with Plastic Fake Plants! Satellite imagery shows green paint applied to dying national forests."
    ]
    
    real_samples = [
        "Federal Reserve signals potential interest rate cuts amid slowing inflation data. Economic analysts predict moderate market growth for the upcoming fiscal quarter.",
        "NASA James Webb Space Telescope discovers distant exoplanet atmosphere containing water vapor. Astronomers publish findings in peer-reviewed astrophysics journal.",
        "Congress passes bipartisan infrastructure bill focusing on highway repairs and renewable energy grid modernization after lengthy senate debate.",
        "World Health Organization releases annual global health report highlighting declining malaria rates in sub-Saharan Africa following new vaccine deployment.",
        "Tech giant announces investment of $10 billion in green energy data centers to meet zero-carbon targets by 2030.",
        "European Central Bank maintains key interest rates following quarterly economic evaluation and consumer price index metrics.",
        "Major breakthrough in battery technology reported by university researchers, promising longer range for electric vehicles.",
        "United Nations climate summit concludes with accord signed by 190 nations to reduce methane emissions over next decade.",
        "Department of Education announces new grants for rural high schools to expand STEM laboratory facilities and vocational training.",
        "Global supply chains show signs of recovery as port congestion eases and freight shipping rates normalize worldwide.",
        "Epidemiologists publish study validating effectiveness of seasonal flu vaccination programs across public schools.",
        "State governor signs legislation increasing funding for public transit infrastructure and clean urban transport systems.",
        "International Space Station crew successfully completes space walk to install upgraded solar array batteries.",
        "Treasury department issues new guidance regarding corporate tax compliance and international reporting standards.",
        "National Weather Service issues seasonal forecasting update indicating milder winter conditions across northern states."
    ]
    
    # Expand samples to create a robust dataset
    fake_list = (fake_samples * 20)[:300]
    real_list = (real_samples * 20)[:300]
    
    df_fake = pd.DataFrame({'title': fake_list, 'text': fake_list, 'label': 1, 'label_name': 'Fake'})
    df_real = pd.DataFrame({'title': real_list, 'text': real_list, 'label': 0, 'label_name': 'Real'})
    
    return pd.concat([df_fake, df_real], ignore_index=True)

def generate_sample_spam_data() -> pd.DataFrame:
    """Generate realistic synthetic dataset for Spam & Ham messages if raw CSV is missing."""
    print("[INFO] Raw spam CSV (spam.csv) not found in /data. Generating high-quality benchmark dataset...")
    spam_samples = [
        "URGENT! You have won a 1000 cash prize or a free iPhone! Call 09061701461 now to claim your reward. Claim code: TXT54. Valid 12 hours only!",
        "WINNER! As a valued network customer you have been selected to receive a £900 gift voucher. Reply WIN to 87077 immediately!",
        "FREE ENTRY into our £250 weekly lottery draw! Text CHOICE to 85023 to enter. Terms and conditions apply. 16+",
        "CONGRATULATIONS! You have been awarded a free holiday to Spain. Call 08712400603 now to book your flights. T&C Apply.",
        "PRIVATE! Your loan of £5000 is pre-approved! No credit check required. Click http://free-cash-now.com to receive funds today.",
        "ALERT: Your bank account has been temporarily locked. Verify your credentials immediately at http://secure-bank-login-verify.com to avoid suspension.",
        "Hot singles in your area want to meet you! Reply YES to 69696 to chat now. £1.50 per msg.",
        "CLAIM NOW: You have 1 unread urgent notification regarding your unclaimed tax refund of £450. Click link to deposit.",
        "Double your money in 24 hours with crypto trading bot! Guaranteed 100% daily return. Join Telegram group t.me/fastcrypto",
        "Final warning! Your package delivery failed due to unpaid shipping fee of $2.99. Pay now at http://postal-redelivery-notice.com"
    ]
    
    ham_samples = [
        "Hey, are we still meeting for lunch today at 12:30? Let me know if you want to grab coffee afterwards.",
        "Hi Mom, just letting you know I arrived safely at the train station. Will see you soon!",
        "Don't forget to bring the project slides for tomorrow's team presentation. See you at the office.",
        "Can you send me the recipe for that pasta dish you made last weekend? It was super delicious!",
        "Running a bit late due to traffic, should be there in about 15 minutes. Sorry for the delay!",
        "The professor shifted the assignment deadline to Friday midnight. Check the course portal for details.",
        "Thanks for helping me fix the code issue yesterday. Everything is compiling cleanly now!",
        "Are you free for a quick phone call this afternoon to discuss the weekend trip plans?",
        "I left my jacket in your car yesterday. Do you mind bringing it over when you come by?",
        "Great job on the presentation today! The client was really impressed with our proposal."
    ]
    
    spam_list = (spam_samples * 30)[:300]
    ham_list = (ham_samples * 30)[:300]
    
    df_spam = pd.DataFrame({'text': spam_list, 'label': 1, 'label_name': 'Spam'})
    df_ham = pd.DataFrame({'text': ham_list, 'label': 0, 'label_name': 'Ham'})
    
    return pd.concat([df_spam, df_ham], ignore_index=True)

def load_and_prep_news_data(data_dir="data"):
    """Load raw news data from data/ folder or generate synthetic data if missing."""
    fake_path = None
    true_path = None
    
    # Check for Kaggle Fake and Real News files
    if os.path.exists(data_dir):
        for fname in os.listdir(data_dir):
            lower = fname.lower()
            if lower in ['fake.csv', 'fake_news.csv']:
                fake_path = os.path.join(data_dir, fname)
            elif lower in ['true.csv', 'real.csv', 'true_news.csv']:
                true_path = os.path.join(data_dir, fname)
            
    if fake_path and true_path:
        print(f"[INFO] Found News datasets: {fake_path} and {true_path}")
        df_fake = pd.read_csv(fake_path)
        df_true = pd.read_csv(true_path)
        
        # Combine title and text
        df_fake['text'] = df_fake.get('title', '').fillna('') + " " + df_fake.get('text', '').fillna('')
        df_true['text'] = df_true.get('title', '').fillna('') + " " + df_true.get('text', '').fillna('')
        
        df_fake['label'] = 1
        df_fake['label_name'] = 'Fake'
        df_true['label'] = 0
        df_true['label_name'] = 'Real'
        
        df = pd.concat([df_fake[['text', 'label', 'label_name']], df_true[['text', 'label', 'label_name']]], ignore_index=True)
    else:
        df = generate_sample_news_data()
        
    # Clean data
    print("[INFO] Cleaning News dataset text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    # Remove rows with empty cleaned_text
    df = df[df['cleaned_text'].str.strip() != ''].copy()
    df.rename(columns={'text': 'original_text'}, inplace=True)
    
    output_path = os.path.join(data_dir, "cleaned_news.csv")
    df[['original_text', 'cleaned_text', 'label', 'label_name']].to_csv(output_path, index=False)
    print(f"[SUCCESS] News dataset prepared: {len(df)} rows saved to {output_path}")
    return df

def load_and_prep_spam_data(data_dir="data"):
    """Load raw spam data from data/ folder or generate synthetic data if missing."""
    spam_path = None
    
    if os.path.exists(data_dir):
        for fname in os.listdir(data_dir):
            lower = fname.lower()
            if 'spam' in lower and lower.endswith('.csv') and 'cleaned' not in lower:
                spam_path = os.path.join(data_dir, fname)
                break
            
    if spam_path:
        print(f"[INFO] Found Spam dataset: {spam_path}")
        # Try multiple encodings as spam.csv often uses latin-1
        try:
            df_raw = pd.read_csv(spam_path, encoding='utf-8')
        except UnicodeDecodeError:
            df_raw = pd.read_csv(spam_path, encoding='latin-1')
            
        # Detect label and text columns
        cols = [c.lower() for c in df_raw.columns]
        if 'v1' in cols and 'v2' in cols:
            df = pd.DataFrame({
                'label_raw': df_raw.iloc[:, 0],
                'text': df_raw.iloc[:, 1]
            })
        elif 'label' in cols and 'text' in cols:
            df = pd.DataFrame({
                'label_raw': df_raw['label'],
                'text': df_raw['text']
            })
        else:
            # Fallback to first two columns
            df = pd.DataFrame({
                'label_raw': df_raw.iloc[:, 0],
                'text': df_raw.iloc[:, 1]
            })
            
        df['label'] = df['label_raw'].astype(str).str.lower().apply(lambda x: 1 if 'spam' in x or x == '1' else 0)
        df['label_name'] = df['label'].apply(lambda x: 'Spam' if x == 1 else 'Ham')
    else:
        df = generate_sample_spam_data()
        
    print("[INFO] Cleaning Spam dataset text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    df = df[df['cleaned_text'].str.strip() != ''].copy()
    df.rename(columns={'text': 'original_text'}, inplace=True)
    
    output_path = os.path.join(data_dir, "cleaned_spam.csv")
    df[['original_text', 'cleaned_text', 'label', 'label_name']].to_csv(output_path, index=False)
    print(f"[SUCCESS] Spam dataset prepared: {len(df)} rows saved to {output_path}")
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    print("[START] Starting Data Preparation...")
    load_and_prep_news_data()
    print("-" * 50)
    load_and_prep_spam_data()
    print("[DONE] Data preparation complete!")

