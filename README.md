# LLM-Based Summarization App: Earnings Call Transcripts

### 1. Purpose
- **Objective**: Summarize earnings call transcripts into concise, actionable summaries.  
- **Focus**: Highlight key financial metrics and strategic insights.

### 2. Key Components
#### a. Data Preprocessing
- Scraped transcripts from sources like Motley Fool using **BeautifulSoup**.

#### b. Summarization Pipeline
- Leveraged **Azure OpenAI's GPT-3.5-turbo** with the following techniques:
  - **Token-based chunking** for breaking down large transcripts.
  - **"Stuffing"** to generate summaries for individual chunks.
  - **"Refining"** to ensure overall summary cohesion.

#### c. Interface & Deployment
- Developed a **Streamlit app** with the following features:
  - **Cost tracking** for monitoring API expenses.
  - Support for **URL-based input** to streamline transcript processing.

#### d. Cost Optimization
- Monitored token usage to ensure **budget-friendly operations** without compromising summary quality.

### 3. Outcome
- A streamlined application for creating concise, insightful summaries of earnings call transcripts.
- Provides users with a **scalable**, **cost-efficient tool** for financial analysis.


###  Steps to Run the App
1. **Clone the Repository**
   ```bash
   git clone https://github.com/mahendra057/Earnings-call-Transcript-LLM-Summarizer.git
   cd Earnings-call-Transcript-LLM-Summarizer

2. pip install -r requirements.txt

3. streamlit run app.py

