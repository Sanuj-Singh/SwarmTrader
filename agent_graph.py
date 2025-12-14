import yfinance as yf
import json
import re
import streamlit as st
import operator
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

def fetch_fundamentals(ticker: str):
    """
    (Financial Agent Tool) USES GOOGLE SEARCH instead of APIs.
    Bypasses network blocks by reading text from the web.
    """
    print(f"--- [Financials] Searching Google for {ticker} data ---")

    query = f"{ticker} stock share price market cap P/E ratio revenue net income beta dividend yield 52 week high low volume"
    try:
        search_results = search_tool.run(query)
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

    prompt = f"""
    You are a Global Financial Data Analyst.

    ### REFERENCE: REGIONAL NUMBER SYSTEMS
    Use these rules if the source text uses regional terms (e.g., "50 Crore"):
    {global_number_systems}

    ### SOURCE DATA (From Search):
    {search_results}

    ### YOUR TASK:
    Extract financial metrics for the company.
    ###  MATH & LOGIC RULES TO FOLLOW :
    1.  **Reliance Industries Check:** The current Market Cap is approx **₹21 Lakh Crore** (INR).
        * *Wrong Math:* 21 Lakh Crore = $21 Trillion USD ( FALSE - This is > US GDP)
        * *Correct Math:* (21 * 10^5 * 10^7) / 84 INR_Rate ≈ **$250 Billion USD** (CORRECT)
    2.  **Date Check:** Prioritize data from **2024-2025**. Ignore "2022" data unless it's the only option.
    3.  **Currency Conversion:**
        * **INR/JPY/KRW:** Divide by Exchange Rate (e.g., Value / 84).
        * **EUR/GBP:** Multiply by Exchange Rate (e.g., Value * 1.10).

    ### EXECUTION STEPS:
    1.  **Extract Raw Native Value:** (e.g., "21,04,299 Crore INR").
    2.  **Convert to Full Integer:** (21,04,299 * 10,000,000 = 21,042,990,000,000).
    3.  **Apply Exchange Rate:** (21,042,990,000,000 / 84.5 = 249,029,467,455).
    4.  **Format:** "249.03B".

    ### REQUIRED OUTPUT (JSON):
    {{
        "meta": {{
            "detected_currency": "e.g. INR",
            "exchange_rate_used": "e.g. 84.5",
            "math_scratchpad": "e.g. 21.04 Lakh Crore / 84.5 = 249B USD"  <-- CRITICAL: WRITE YOUR MATH HERE
        }},
        "metrics": {{
            "Market Cap": "Value in USD (e.g. 249.03B)",
            "Revenue TTM": "Value in USD",
            "Net Income": "Value in USD",
            "Beta": "Value",
            "PE Ratio": "Value",
            "EPS TTM": "Value in USD",
            "Dividend": "Value %",
            "52W High": "Value (Native)",
            "52W Low": "Value (Native)",
            "Volume": "Value in M",
            "Shares Out": "Value"
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

def get_company_news(company_name: str, ticker: str):

        try:
            query= f"{company_name} who is the current CEO, when was it founded, which industry and sector. latest earnings results guidance downgrade upgrade ,regulatory action lawsuit investigation merger acquisition leadership change,financial scandal controversy analyst ratings major partnership stock forecast impact and latest news"
            raw_results = search_tool.run(query)
            prompt = f"""Extract the following details for {company_name} from the search results below.who is the current CEO, when was it founded, which industry and sector.If a field is missing, use "N/A".

                SEARCH RESULTS:
                {raw_results}

                Extract these specific details and return valid JSON only:

               "News": {{

                    "news_summary": "summary of latest news"
                }}

                """

            response = llm.invoke(prompt)

            # Clean JSON
            clean_news_json = response.content.replace("```json", "").replace("```", "").strip()
            news_data = json.loads(clean_news_json)
            news_output=news_data.get("News",{})
            
            return {
                "News":news_output,
                
                    }
        except Exception as e:
            print(f"Error fetching news: {e}")
            return "new Error"


#function to get company details like CEO founded industry sector
def get_company_details(ticker: str):
    print(f"--- [Company Details] Fetching details for {ticker} ---")
    try:
        ticker = yf.Ticker(ticker)
        info = ticker.info
        return {
            "CEO": info.get("ceoCompName", info.get("CEO", "Information Not Available")),
            "founded": info.get("startDate", "Information Not Available"),
            "industry": info.get("industry", "Information Not Available"),
            "sector": info.get("sector", "Information Not Available")
        }
    except:
        return {
            "CEO": "Information Not Available",
            "founded": "Information Not Available", 
            "industry": "Information Not Available",
            "sector": "Information Not Available"
        }

# --- 3. AGENT NODES ---
def ticker_node(state: AgentState):
    return {"ticker": lookup_ticker(state['company_name'])}


def financials_agent(state: AgentState):
    return {"financial_data": fetch_fundamentals(state['ticker'])}


def market_data_agent(state: AgentState):
    return {"market_data": get_stock_price(state['ticker'])}


def news_agent(state: AgentState):
    news_and_details = get_company_news(state['company_name'], state['ticker'])
    return {"news_data": news_and_details.get("News", {})}

def Company_details_agent(state: AgentState):
    return {"company_details": get_company_details(state['ticker'])}
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
workflow.add_node("master_analyst", analyst_node)

workflow.set_entry_point("ticker_resolver")
workflow.add_edge("ticker_resolver", "financials_agent")
workflow.add_edge("financials_agent", "market_data_agent")
workflow.add_edge("market_data_agent", "news_agent")
workflow.add_edge("news_agent", "master_analyst")
workflow.add_edge("master_analyst", END)

app = workflow.compile()
# ---  RUN FUNCTION ---
def run_analysis(name_input: str):
    inputs = {"company_name": name_input, "messages": []}
    for output in app.stream(inputs):
        yield output

