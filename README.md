# 📊 SwarmTrader: AI Multi-Agent Market Analyst

**SwarmTrader** is an advanced financial analysis tool powered by a "swarm" of autonomous AI agents using **LangGraph** and **Google Gemini**. It orchestrates multiple agents to fetch stock data, analyze news, normalize global financial metrics, and generate a comprehensive investment report with a Buy/Sell/Hold recommendation.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange)
![Gemini](https://img.shields.io/badge/AI-Gemini%202.5%20Flash-green)

## 🚀 Features

* **🤖 Multi-Agent Architecture**: Uses a directed state graph to coordinate specialized agents (Ticker Resolver, Financials, Market Data, News, and Master Analyst).
* **🌍 Global Ticker Resolution**: Automatically identifies stock tickers for companies across various exchanges (NYSE, NSE, LSE, TSX, etc.) and handles suffix logic (e.g., Reliance -> `RELIANCE.NS`).
* **💱 Smart Currency Normalization**:
    * **Region Detection**: Distinguishes between Western (Millions/Billions) and Asian (Lakhs/Crores/Wan/Yi) number systems.
    * **Math & Currency Conversion**: Automatically converts mixed currencies (INR, JPY, GBP) to **USD** for a fair global market cap comparison.
* **📰 Market-Moving News Analysis**: Scrapes recent headlines and filters specifically for high-impact events (Earnings, Regulatory Actions, M&A) rather than general noise.
* **📈 Interactive UI**: Built with Streamlit, featuring real-time agent status indicators, interactive Plotly price charts, and dynamic SWOT analysis cards.

## 🏗️ Architecture

The application is built on **LangGraph**, creating a workflow where state is passed between agents:

1.  **Ticker Resolver**: Identifies the correct symbol (e.g., "Tata Motors" -> `TATAMOTORS.NS`).
2.  **Financials Agent**: Uses Google Search to find fundamental ratios (P/E, Market Cap, Revenue) and normalizes the data to USD.
3.  **Market Data Agent**: Fetches historical price data and volume via `yahooquery`.
4.  **News Agent**: Aggregates recent news and summarizes sentiment/impact.
5.  **Company Details Agent**: Retrieves CEO, Sector, Industry, and founding details.
6.  **Master Analyst**: The final LLM node that synthesizes all collected data into a structured report with a sentiment score and strategic recommendation.

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.9+
* A Google Cloud Project with:
    * **Gemini API Key**
    * **Google Custom Search JSON API** enabled
    * **Search Engine ID (CSE ID)**

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/swarmtrader.git](https://github.com/yourusername/swarmtrader.git)
cd swarmtrader
```
