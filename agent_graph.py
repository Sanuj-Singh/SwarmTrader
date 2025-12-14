import yfinance as yf
import json
import re
import streamlit as st
import operator
import datetime
from yahooquery import Ticker
from typing import TypedDict, List, Annotated
from langchain_core.messages import HumanMessage
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END


GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GOOGLE_SEARCH_API_KEY = st.secrets["GOOGLE_SEARCH_API_KEY"]
GOOGLE_CSE_ID = st.secrets["GOOGLE_CSE_ID"]
@st.cache_resource
def get_agents():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    )
    search_tool = GoogleSearchAPIWrapper(
        google_api_key=GOOGLE_SEARCH_API_KEY,
        google_cse_id=GOOGLE_CSE_ID
    )
    return llm, search_tool

llm, search_tool = get_agents()

class AgentState(TypedDict):
    company_name: str
    ticker: str
    financial_data: dict
    market_data: dict
    news_data: dict
    company_details: dict
    final_report: dict
    messages: Annotated[List[str], operator.add]

# --- 2. TICKER RESOLUTION finding ticker from company name also global rules for different exchanges---

def lookup_ticker(company_name: str):
    print(f"   ... Finding ticker for '{company_name}' ...")
    try:
        prompt = f"""
        Act as a financial data expert. Identify the correct Yahoo Finance stock ticker for the company: '{company_name}'.

        Follow these GLOBAL TICKER RULES:
        1.  **USA (NYSE/NASDAQ):** Ticker only (e.g., Apple -> AAPL).
        2.  **India:** Append '.NS' for NSE (e.g., Reliance -> RELIANCE.NS).
        3.  **United Kingdom:** Append '.L' (e.g., Tesco -> TSCO.L).
        4.  **Canada:** Append '.TO' for TSX (e.g., Shopify -> SHOP.TO).
        5.  **Europe:** Append '.DE' (Frankfurt), '.PA' (Paris), or '.AS' (Amsterdam).
        6.  **Asia:** Append '.HK' (Hong Kong), '.T' (Tokyo), '.SS' (Shanghai).
        7.  **Australia:** Append '.AX'.

        CRITICAL INSTRUCTIONS:
        - Select the **primary listing** with the highest trading volume.
        - If the company is private or cannot be found, return "null".
        - **OUTPUT:** Return STRICT JSON only. No markdown, no conversational text.

        ### EXAMPLES:
        Input: "Samsung Electronics"
        Output: {{"ticker": "005930.KS"}}

        Input: "LVMH"
        Output: {{"ticker": "MC.PA"}}

        Input: "Commonwealth Bank"
        Output: {{"ticker": "CBA.AX"}}

        Input: "Microsoft"
        Output: {{"ticker": "MSFT"}}

        Input: "{company_name}"
        Output:
        """
        response = llm.invoke(prompt)

        match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            ticker = data.get("ticker", "UNKNOWN").upper().strip()
           #removes unwanted characters from ticker 
            ticker = re.sub(r'[^A-Z0-9.]', '', ticker)
            return ticker
        return "UNKNOWN"
    except Exception as e:
        print(f"Ticker Error: {e}")
        return "UNKNOWN"




#function to fetch fundamentals using google search 

def fetch_fundamentals(company_name:str,ticker: str):
    """
    (Financial Agent Tool) USES GOOGLE SEARCH instead of APIs.
    Bypasses network blocks by reading text from the web.
    """
    print(f"--- [Financials] Searching Google for {ticker} data ---")

    query = f"{company_name} stock share price, market cap, P/E ratio, revenue, net income, beta, dividend yield, 52 week high,52 week low, volume"
    try:
        search_results = search_tool.run(query)
        print(f"{search_results}")
    except Exception as e:
        print(f"Search failed: {e}")
        search_results = "No search results found."
    # Build prompt for LLM to extract and convert financial metrics
    global_number_systems = [
        {
            "region": "India/South Asia",
            "systems": ["Lakh", "Crore", "Arab"],
            "conversion_rules": {"lakh": 100000, "crore": 10000000, "arab": 1000000000}
        },
        {
            "region": "East Asia (China/Japan/Korea)",
            "systems": ["Wan", "Man", "Yi", "Oku", "Cho"],
            "conversion_rules": {
                "wan": 10000, "man": 10000,
                "yi": 100000000, "oku": 100000000,
                "cho": 1000000000000
            }
        },
        {
            "region": "International/Western",
            "systems": ["Million", "Billion", "Trillion"],
            "conversion_rules": {"million": 1000000, "billion": 1000000000, "trillion": 1000000000000}
        }
    ]

    prompt =f"""

You are a Senior Global Financial Data Analyst.
Your goal is to extract current financial fundamentals for the specific company identified by Ticker: {ticker}.

### 1. INPUT DATA
**Reference - Regional Number Systems:**
{global_number_systems}

**Source Text (Search Results):**
{search_results}

---

### 2. CRITICAL EXECUTION PROTOCOLS

**Protocol A: Entity Disambiguation (The "Who" Filter)**
1.  **Strict Matching:** You must ONLY extract data for the company matching {ticker}.
2.  **Competitor Trap:** The source text may list competitors (e.g., comparing TCS to Infosys). IGNORE metrics belonging to other companies.
3.  **Ticker Verification:** If the ticker ends in `.NS` or `.BO`, it is an Indian entity. If `.L`, it is UK. If `.T`, it is Japanese. Adjust currency logic accordingly.

**Protocol B: The "Lakh-Crore" Math Trap (The "How" Filter)**
*Common LLM Error:* Confusing "Lakh Crore" with "Trillions".
* **Rule:** 1 Lakh Crore INR != 1 Trillion USD.
* **Math:** 1 Lakh Crore = 1,00,000,00,00,000 (10^12) INR = 100 Billion INR.
* **Conversion:** (Value in Lakh Crore * 10^12) / Exchange Rate or directly (Value in Lakh Crore * 100 Billion) / Exchange Rate
* **Sanity Check:** No Indian company exceeds $300B USD (approx). No Global company exceeds $4T USD. If your calculation exceeds these limits, YOUR MATH IS WRONG. Recalculate.

**Protocol C: Currency Normalization**
1.  **Detect Currency:** Look for symbols (₹, $, €, £, ¥, KRW, GBX).
2.  **Date Prioritization:** Prioritize 2024-2025 (TTM) data. Ignore data older than 2023.
3.  **Conversion Logic:**
    * **Weak Currencies (INR, JPY, KRW):** DIVIDE by rate (e.g., INR/84, JPY/150).
    * **Strong Currencies (GBP, EUR):** MULTIPLY by rate (e.g., GBP*1.27).
    * **UK Specific:** If priced in Pence (GBX), divide by 100 to get GBP, then convert to USD.
    * **Market Cap Normalization:** ALWAYS convert Market Cap to USD regardless of source currency for fair global comparison.

**Protocol D: Number Format Understanding**
* Indian System: 1 Lakh = 100,000; 1 Crore = 10,000,000; 1 Lakh Crore = 1,00,000,00,00,000
* International: 1 Million = 1,000,000; 1 Billion = 1,000,000,000; 1 Trillion = 1,000,000,000,000
* Conversion: 1 Lakh Crore INR = 100 Billion INR

**Protocol E: Global Market Consistency**
1. Market Cap should always be presented in USD for global comparison
2. Revenue and Income figures should be converted to USD
3. 52W High/Low should remain in native currency
4. Always include USD conversion for global market cap ranking

---

### 3. EXTRACTION STEPS (Chain of Thought)
Perform these steps internally before generating JSON:
1.  **Identify** the specific company name associated with {ticker}.
2.  **Extract** raw strings (e.g., "21.5 Lakh Cr", "450 Billion Yen", "€12.3 Billion").
3.  **Normalize** to standard numerical format (e.g., 2,150,000,000,000 for Lakh Crore).
4.  **Convert** to USD using current approximate rates:
    * INR: ~84
    * JPY: ~150
    * GBP: ~1.27
    * EUR: ~1.10
    * KRW: ~1,330
    * Adjust as necessary for precision
5.  **Format** output (B = Billion, M = Million) with exactly 2 decimal places.

---

### 4. REQUIRED OUTPUT (Strict JSON)
Return ONLY this JSON object. If a specific metric is not found or ambiguous, set it to "N/A".
ALL monetary values EXCEPT 52W High and 52W Low should be converted to USD for global market comparison.

{{
  "meta": {{
    "target_company": "Name of company identified",
    "detected_currency": "Original currency symbol (e.g. INR, JPY, USD)",
    "exchange_rate_used": "Exchange rate applied for conversion to USD (e.g. 84.5)",
    "market_cap_usd": "Market cap converted to USD for global comparison",
    "math_scratchpad": "SHOW YOUR WORK. Ex: (21 Lakh Crore * 100 Billion) / 84 = $250B"
  }},
  "metrics": {{
    "Market Cap": "Value in USD ONLY (e.g. 249.03B) for global market comparison",
    "Revenue TTM": "Value in USD (e.g. 12.5B)",
    "Net Income": "Value in USD",
    "Beta": "Float value",
    "PE Ratio": "Float value",
    "EPS TTM": "Value in USD",
    "Dividend Yield": "Percentage %",
    "52W High": "Value in NATIVE currency (NOT converted to USD)",
    "52W Low": "Value in NATIVE currency (NOT converted to USD)",
    "Volume": "Value in M (Millions)",
    "Shares Outstanding": "Value in B or M"
  }}
}}
"""

    try:
        response = llm.invoke(prompt)

        # Clean JSON
        clean_json = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        metrics = data.get("metrics", {})

        # MOCK CHART DATA (Since search doesn't give arrays easily)
        # We return empty charts so the UI doesn't crash
        chart_data = {
            "years": ["2020", "2021", "2022", "2023"],
            "revenue": [0, 0, 0, 0],
            "net_income": [0, 0, 0, 0]
        }

        return {"metrics": metrics, "chart_data": chart_data}

    except Exception as e:
        print(f"Extraction Error: {e}")
        # TO ENSURE UI STABILITY, return empty metrics on failure
        return {
            "metrics": {k: "N/A" for k in ["Market Cap", "Revenue", "Net Income", "Beta", "P/E Ratio","Share Price" , "52W High"]},
            "chart_data": {}
        }
# Price History Tool Fetcher
def get_stock_price(ticker: str):
    print("Market Data Tool) Uses yahooquery for price history.")
    if "UNKNOWN" in ticker: return {"error": "Invalid Ticker"}

    try:
        Yahoticker = Ticker(ticker)
        df = Yahoticker.history(period='1y', interval='1d')

        #
        if isinstance(df, dict) or df.empty:  # Check for empty or error dict
            return {"error": "No data found"}

        # Reset index to get Date as a column (sometimes it's 'Date' or 'date')
        df = df.reset_index()

        # Ensure 'date' column exists (sometimes it's 'Date' or 'date')
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'Date'})

        # Format for Frontend (Plotly)
        history_data = df[['Date', 'close', 'volume']].rename(columns={'close': 'Close', 'volume': 'Volume'}).to_dict(
            'records')

        # Format for LLM
        llm_context = df.tail(10).to_string()

        return {
            "history_data": history_data,
            "llm_context": llm_context
        }
    except Exception as e:
        print(f"   [Error] YahooQuery Price failed: {e}")
        return {"error": str(e)}

def get_company_news(ticker: str):
    Yticker=Ticker(ticker)
    
    try:
        raw_news=Yticker.news(count=15)
#            query= f"{company_name} latest earnings results guidance downgrade upgrade ,regulatory action lawsuit investigation merger acquisition leadership change,financial scandal controversy analyst ratings major partnership stock forecast impact and latest news"
 #           raw_results = search_tool.run(query)
        news_list = []

        if isinstance( raw_news,dict):
            news_list = raw_news.get(ticker, [])
            if not news_list:
                # If the dict has values, take the first value found (robustness for single ticker queries)
                values = list(raw_news.values())
                if values and isinstance(values[0], list):
                    news_list = values[0]
        elif isinstance(raw_news, list):
            news_list = raw_news
        if not isinstance(news_list, list):
            news_list = []
        news_context=""
        if news_list:
            for item in news_list:
                if not isinstance(item, dict):
                    continue
                title = item.get("title", "")
                summary = item.get("summary","")[:500]  # Limit summary length
                published_time = item.get("providerPublishTime", "")
                pub_date=""
                if published_time:
                    try:
                        pub_date = datetime.datetime.fromtimestamp(published_time).strftime('%Y-%m-%d')
                    except:
                        pub_date="Recent"
                news_context +=f"- [{pub_date}]::: {title}\n {summary}\n"
        else:
            news_context="No recent news found."
        prompt = f"""Analyze the following data for ({ticker}):

        RECENT NEWS HEADLINES:
        {news_context}

        Task:
        Summarize ONLY market-moving events:
        - Earnings & guidance
        - Analyst upgrades/downgrades
        - M&A
        - Regulatory, lawsuits, investigations
        - Leadership changes
        Return STRICT JSON ONLY. No markdown, no extra text:{{
        "News": {{
            "news_summary": "Your summary here...",
            "impact_level": "HIGH | MEDIUM | LOW"
            
            }}
        }}
        """

        response = llm.invoke(prompt)

            # Clean JSON
        clean_news_json = response.content.replace("```json", "").replace("```", "").strip()
        news_data = json.loads(clean_news_json)
        news_output=news_data.get("News",news_data)
            
        return {
                "News":news_output,
                 }
    except Exception as e:
            print(f"Error fetching news: {e}")
            return {
            "News": {
                "news_summary": f"Could not fetch news due to error: {str(e)}"
            }
        }

#function to get company details like CEO founded industry sector
def get_company_details(company_name: str,ticker: str):
    print(f"--- [Company Details] Fetching details for {ticker} ---")
    companyquery=f"{company_name} CEO, founded, year, industry, and sector"
    try:
        management_results = search_tool.run(companyquery)
        print(f"{management_results}")
    except Exception as e:
        print(f"Search failed: {e}")
    prompt = f"""
        ### SOURCE DATA (From Search):
    {management_results}
    Extract the following company details from the text below:
    1. CEO Name
    2. Year Founded
    3. Industry
    4. Sector
    Return STRICT JSON ONLY. No markdown, no extra text.
    {{"company_details":{{
        "CEO": "Name of CEO",
        "founded": "Year Founded",
        "industry": "Industry Name",
        "sector": "Sector Name"
    }}
    }}

    """
    try:
        response = llm.invoke(prompt)

        # Clean JSON
        clean_json = response.content.replace("```json", "").replace("```", "").strip()
        comdata = json.loads(clean_json)
        company_details = comdata.get("company_details", {})
        return {
            "CEO": company_details.get("CEO", "N/A"),
            "founded": company_details.get("founded", "N/A"),
            "industry": company_details.get("industry", "N/A"),
            "sector": company_details.get("sector", "N/A")
        }
    except:
        return {"CEO": "N/A", "founded": "N/A", "industry": "N/A", "sector": "N/A"}

# --- 3. AGENT NODES ---
def ticker_node(state: AgentState):
    return {"ticker": lookup_ticker(state['company_name'])}


def financials_agent(state: AgentState):
    return {"financial_data": fetch_fundamentals(state['company_name'],state['ticker'])}


def market_data_agent(state: AgentState):
    return {"market_data": get_stock_price(state['ticker'])}


def news_agent(state: AgentState):
    news_and_details = get_company_news(state['ticker'])
    return {"news_data": news_and_details.get("News", {})}

def Company_details_agent(state: AgentState):
    return {"company_details": get_company_details(state["company_name"],state['ticker'])}
# --- MASTER ANALYST NODE  COMBINES ALL DATA  FOR FINAL REPORT---
def analyst_node(state: AgentState):
    print(f"--- [Analyst] Analyzing data for {state['company_name']} ---")

    metrics = state.get('financial_data', {}).get('metrics', {})
    price_txt = state.get('market_data', {}).get('llm_context', 'No Price data')
    news_payload = state.get('news_data', {})
    company_details = state.get('company_details', {})

    prompt = f"""You are a Senior Financial Analyst. Analyze {state['company_name']} ({state['ticker']}).

    ### DATA:
    COMPANY DETAILS:
    Market_data : {price_txt}
    Fundamentals: {metrics}
    news: {news_payload}
    company details: {company_details}


    ### INSTRUCTIONS:
    1. Analyze the data to determine a Buy/Sell/Hold recommendation.
    2. **STRICT TEXT RULE:** Use ONLY plain text. NO Markdown, NO asterisks (**), NO bolding, NO bullet points characters.
    3. JSON OUTPUT ONLY:
    {{
        "sentiment_score": 50,
        "confidence_score": 50,
        "recommendation": "BUY",
        "volatility": "Medium",
        "swot": {{
            "strengths": ["Factor 1", "Factor 2"],
            "weaknesses": ["Factor 1", "Factor 2"],
            "opportunities": ["Factor 1", "Factor 2"],
            "threats": ["Factor 1", "Factor 2"]
        }},
        "companies_details": {{

            "CEO": "Name",
            "founded": "Year",
            "industry": "Industry Name",
            "sector": "Sector Name",


        }}
        "summary": "Write a clean, professional paragraph here without any special formatting."
    }}
    """

    parsed_report = {}

    def clean_text(text):
        if isinstance(text, str):
            return text.replace("**", "").replace("*", "").replace("#", "").replace("_", "").strip()
        return text

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if match:
            parsed_report = json.loads(match.group(0))

            # --- CLEANUP STEP ---
            # Manually clean every field to ensure plain text
            parsed_report['summary'] = clean_text(parsed_report.get('summary', ''))


            # Clean the SWOT lists
            swot = parsed_report.get('swot', {})
            for key in ['strengths', 'weaknesses', 'opportunities', 'threats']:
                if key in swot and isinstance(swot[key], list):
                    swot[key] = [clean_text(item) for item in swot[key]]
            
            company_details = parsed_report.get('companies_details', {})
            
            for key in ['CEO', 'founded', 'industry', 'sector']:
                if key in company_details and isinstance(company_details[key], list):
                    company_details[key] =[clean_text(itemm) for itemm in company_details[key]]
            if 'companies_details' in parsed_report:
                parsed_report['company_details'] = parsed_report.pop('companies_details')
        else:
            raise ValueError("No JSON found")

    except Exception as e:
        print(f"Analyst Error: {e}")
        parsed_report = {
            "recommendation": "HOLD",
            "sentiment_score": 50,
            "confidence_score": 0,
            "summary": "Analysis failed. Please try again.",
            "swot": {"strengths": [], "weaknesses": [], "opportunities": [], "threats": []},
            "company_details": {"CEO": "N/A", "founded": "N/A", "industry": "N/A", "sector": "N/A"},
            "Fundamentals": metrics,
            "News": news_payload
        }
    print(company_details)
    return {"final_report": parsed_report}
# --- 4. WORKFLOW DEFINITION ---

workflow = StateGraph(AgentState)
workflow.add_node("ticker_resolver", ticker_node)
workflow.add_node("financials_agent", financials_agent)
workflow.add_node("market_data_agent", market_data_agent)
workflow.add_node("news_agent", news_agent)
workflow.add_node("company_details_agent", Company_details_agent)
workflow.add_node("master_analyst", analyst_node)

workflow.set_entry_point("ticker_resolver")
workflow.add_edge("ticker_resolver", "financials_agent")
workflow.add_edge("financials_agent", "market_data_agent")
workflow.add_edge("market_data_agent", "news_agent")
workflow.add_edge("news_agent", "company_details_agent")
workflow.add_edge("company_details_agent", "master_analyst")
workflow.add_edge("master_analyst", END)

app = workflow.compile()
# ---  RUN FUNCTION ---
def run_analysis(name_input: str):
    inputs = {"company_name": name_input, "messages": []}
    for output in app.stream(inputs):
        yield output

