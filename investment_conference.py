import streamlit as st
import time
import yfinance as yf
import re
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from groq import Groq
from xai_sdk import Client as XAIClient

# ---------------- API KEYS (REPLACE THESE) ----------------
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
GOOGLE_API_KEY = "AIza..."
GROQ_API_KEY = "gsk_..."
XAI_API_KEY = "xai-..."

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
xai_client = XAIClient(api_key=XAI_API_KEY)

# LLM Agents
agents = {
    "GPT-4o": {"client": openai_client, "model": "gpt-4o", "style": "Aggressive Growth Investor"},
    "Claude-3.5-Sonnet": {"client": anthropic_client, "model": "claude-3-5-sonnet-20241022", "style": "Value & Quality Investor"},
    "Gemini-1.5-Pro": {"client": genai.GenerativeModel("gemini-1.5-pro"), "model": "gemini-1.5-pro", "style": "Risk-Averse Balanced Investor"},
    "Groq-Llama-3.1-70B": {"client": groq_client, "model": "llama-3.1-70b-versatile", "style": "Momentum & Data-Driven Quant"},
    "Grok-4": {"client": xai_client, "model": "grok-4-1-fast-reasoning", "style": "Truth-Seeking Contrarian"},
}

# Moderator for consensus & street vibe
moderator_agent = {"client": openai_client, "model": "gpt-4o-mini", "style": "Neutral Moderator & Street Sentiment Analyst"}

# Unified LLM response function
def get_llm_response(agent, prompt, stream=False):
    client = agent["client"]
    model = agent["model"]
    style = agent["style"]
    full_prompt = f"You are a {style}. Respond briefly (2-4 sentences) and directly to: {prompt}"
    
    try:
        if "openai" in str(type(client)).lower() or isinstance(client, Groq):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=300,
                temperature=0.7,
                stream=stream
            )
            if stream:
                return response
            return response.choices[0].message.content.strip()

        elif isinstance(client, Anthropic):
            response = client.messages.create(
                model=model,
                max_tokens=300,
                messages=[{"role": "user", "content": full_prompt}],
                stream=stream
            )
            if stream:
                return response
            return response.content[0].text.strip()

        elif "generativeai" in str(type(client)).lower():
            response = client.generate_content(full_prompt, stream=stream)
            if stream:
                return response
            return response.text.strip()

        elif "xai" in str(type(client)).lower():
            chat = client.chat.create(model=model)
            chat.add_message("user", full_prompt)
            response = chat.complete()
            return response.content.strip()

    except Exception as e:
        return f"Error with {style}: {str(e)}"

# Real-time stock data
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period="5d")
        price = hist['Close'].iloc[-1] if not hist.empty else "N/A"
        change_pct = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0
        news = [item['title'] for item in stock.news[:3]] if hasattr(stock, 'news') else []
        return {"price": round(price, 2), "change_pct": round(change_pct, 2), "news": news}
    except:
        return {"price": "Error", "change_pct": 0, "news": []}

# Extract tickers from text
def extract_tickers(text):
    return list(set(re.findall(r'\b[A-Z]{2,5}\b', text.upper())))

# Streamlit App
st.title("Multi-LLM Investment Conference App")
st.markdown("Ask about stocks/portfolio. 5 LLMs give takes → quick consensus + **Street Vibe** (Wall Street + X/Reddit/StockTwits/social) → optional full debate.")

# Portfolio Memory (local browser session)
if "portfolio" not in st.session_state:
    st.session_state.portfolio = "Example: MU 15%, AVGO 12%, NBIS 10%, CRDO 8%, ASML 6%, dry powder $30k. Aggressive AI infra focus."
portfolio = st.text_area("Your Portfolio (saved locally):", value=st.session_state.portfolio, height=120)
st.session_state.portfolio = portfolio

# Query
query = st.text_input("Your question:")

if st.button("Get Quick Consensus") and query:
    with st.spinner("Consulting the council..."):
        tickers = extract_tickers(query + " " + portfolio)
        market_data = {t: get_stock_data(t) for t in tickers}
        data_str = "\n".join([f"{t}: ${d['price']} ({d['change_pct']}%), News: {', '.join(d['news'])}" for t, d in market_data.items()])
        
        prompt = f"Portfolio: {portfolio}\nMarket data: {data_str if data_str else 'None'}\nQuery: {query}"

        rationales = {}
        for name, agent in agents.items():
            rationales[name] = get_llm_response(agent, prompt)
        
        st.subheader("Brief Rationales")
        for name, rationale in rationales.items():
            with st.expander(f"📊 {name} ({agents[name]['style']})"):
                st.write(rationale)
        
        # Street Vibe (Wall Street + social media / X / Reddit / StockTwits)
        vibe_prompt = f"Query: '{query}'\nRationales: {rationales}\nMarket data: {data_str}\n\nProvide a concise cumulative summary of the current **street vibe** on relevant stocks/sectors (early February 2026). Include Wall Street/analyst consensus, plus social media sentiment from X (Twitter), Reddit, StockTwits, and comment sections. Highlight key themes, tone (bullish/cautious/fearful), recurring opinions, risks/opportunities, and notable trader reactions."
        street_vibe = get_llm_response(moderator_agent, vibe_prompt)
        st.subheader("Street Vibe (Wall Street + Social Media Consensus)")
        st.info(street_vibe)
        
        # Quick Consensus
        mod_prompt = f"Summarize majority view, agreements/disagreements, action rec: {rationales}"
        consensus = get_llm_response(moderator_agent, mod_prompt)
        st.info(f"**Quick Consensus:** {consensus}")

# Full Debate
if st.button("Run Full Debate (3-5 min)") and query:
    st.subheader("Extended Debate")
    debate_container = st.empty()
    debate_log = "Debate starting...\n\n"
    
    tickers = extract_tickers(query + " " + portfolio)
    data_str = "\n".join([f"{t}: ${d['price']} ({d['change_pct']}%), News: {', '.join(d['news'])}" for t, d in {t: get_stock_data(t) for t in tickers}.items()])
    debate_prompt = f"Portfolio: {portfolio}\nData: {data_str}\nQuery: {query}\nDebate with others. Build toward consensus."
    
    for round_num in range(1, 4):
        debate_log += f"**Round {round_num}**\n"
        for name, agent in agents.items():
            response = get_llm_response(agent, debate_prompt + "\nPrevious: " + debate_log[-1500:])
            debate_log += f"**{name}**: {response}\n\n"
            debate_container.markdown(debate_log)
            time.sleep(1.2)
        debate_log += "---\n"
    
    # Final Street Vibe recap
    vibe_prompt = f"From debate: {debate_log[-2500:]}\nSummarize cumulative street vibe (Wall Street + X/Reddit/StockTwits/social) and final consensus."
    final_vibe = get_llm_response(moderator_agent, vibe_prompt)
    debate_log += f"**Street Vibe Recap:** {final_vibe}"
    debate_container.markdown(debate_log)

# Quick Stock Lookup
ticker = st.text_input("Quick stock lookup:")
if ticker:
    data = get_stock_data(ticker)
    st.write(f"**{ticker.upper()}** — Price: ${data['price']} | Change: {data['change_pct']}%")
    if data['news']:
        st.write("Recent news:")
        for n in data['news']:
            st.write(f"- {n}")
          
