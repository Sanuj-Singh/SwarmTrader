import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import time
from datetime import datetime, timedelta
import numpy as np
from agent_graph import run_analysis

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="SwarmTrader",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# 2. CUSTOM CSS
st.markdown("""
<style>
    /* Global Background */
    .stApp { background-color: #f4f6f9; }

    /* --- SIDEBAR STYLING --- */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }

    /* --- CARD STYLING --- */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e6e9ef;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100%;
    }
    .metric-label {
        font-size: 12px;
        color: #6c757d;
        font-weight: 500;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 18px;
        color: #212529;
        font-weight: 700;
    }
    .metric-icon {
        font-size: 20px;
        margin-right: 10px;
        background: #f8f9fa;
        padding: 8px;
        border-radius: 50%;
    }

    /* --- SIGNAL CARDS --- */
    .signal-card-green {
        background-color: #dbf2e3; 
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 15px;
        height: 100px;
        color: #155724;
    }
    .signal-card-red {
        background-color: #f8d7da; 
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 15px;
        height: 100px;
        color: #721c24;
    }
    .signal-card-orange {
        background-color: #fff3cd; 
        border: 1px solid #ffeeba;
        border-radius: 8px;
        padding: 15px;
        height: 100px;
        color: #856404;
    }
    .signal-title {
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .signal-value {
        font-size: 28px;
        font-weight: 800;
    }

    /* --- PROGRESS BAR CUSTOM --- */
    .progress-container {
        width: 100%;
        background-color: #e9ecef;
        border-radius: 4px;
        height: 8px;
        margin-top: 10px;
    }
    .progress-bar {
        height: 8px;
        border-radius: 4px;
        transition: width 0.5s ease-in-out;
    }
    /* Add this inside your st.markdown style block */
    .info-card {
        background-color: #ffffff;
        border: 1px solid #e6e9ef;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)


# 3. HELPER FUNCTIONS
def create_fundamental_card(icon, label, value):
    return st.markdown(f"""
    <div class="metric-card">
        <div style="display: flex; align-items: center;">
            <span class="metric-icon">{icon}</span>
            <div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)


def render_agent_status(name, status):
    if status == "idle":
        color, icon, text = "#999999", "⚪", "Idle"
    elif status == "running":
        color, icon, text = "#ff9800", "🟠", "Processing..."
    elif status == "done":
        color, icon, text = "#28a745", "🟢", "Completed"
    elif status == "error":
        color, icon, text = "#dc3545", "🔴", "Failed"
    else:
        color, icon, text = "#999999", "⚪", "Unknown"

    return f"""
    <div style="background-color: white; border: 1px solid {color}; 
                border-radius: 5px; padding: 8px; margin-bottom: 5px; 
                display: flex; align-items: center;">
        <span style="font-size: 18px; margin-right: 10px;">{icon}</span>
        <div>
            <div style="font-weight: bold; color: #333; font-size: 14px;">{name}</div>
            <div style="font-size: 12px; color: {color};">{text}</div>
        </div>
    </div>
    """


# 4. SIDEBAR SETUP
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    company_input = st.text_input("Enter Company Name", value="", key="ticker_input")
    run_btn =st.button("Generate Analysis ↗", type="primary", width="stretch")

    st.markdown("---")
    st.markdown("**Agent Status**")

    p_resolve = st.empty()
    p_fin = st.empty()
    p_market = st.empty()
    p_news = st.empty()
    p_details = st.empty()
    p_analyst = st.empty()
    # LOGIC: If we already have data, show all as DONE (Green)
    # Otherwise, show them as IDLE (Grey)
    initial_status = "done" if 'data' in st.session_state else "idle"

    p_resolve.markdown(render_agent_status("Ticker Resolver", initial_status), unsafe_allow_html=True)
    p_fin.markdown(render_agent_status("Financials Agent", initial_status), unsafe_allow_html=True)
    p_market.markdown(render_agent_status("Market Data Agent", initial_status), unsafe_allow_html=True)
    p_news.markdown(render_agent_status("News Agent", initial_status), unsafe_allow_html=True)
    p_details.markdown(render_agent_status("Company Details", initial_status), unsafe_allow_html=True)
    p_analyst.markdown(render_agent_status("Master Analyst", initial_status), unsafe_allow_html=True)
# 5. EXECUTION LOGIC
if run_btn:
    # Reset UI to running state
    p_resolve.markdown(render_agent_status("Ticker Resolver", "running"), unsafe_allow_html=True)
    p_fin.markdown(render_agent_status("Financials Agent", "idle"), unsafe_allow_html=True)
    p_market.markdown(render_agent_status("Market Data Agent", "idle"), unsafe_allow_html=True)
    p_news.markdown(render_agent_status("News Agent", "idle"), unsafe_allow_html=True)
    p_details.markdown(render_agent_status("Company Details", "idle"), unsafe_allow_html=True)
    p_analyst.markdown(render_agent_status("Master Analyst", "idle"), unsafe_allow_html=True)
    final_state = {}

    try:
        with st.spinner("Coordinating Multi-Agent Swarm..."):
            for chunk in run_analysis(company_input):

                # LOOP through the chunk to update state and status
                for agent_name, agent_data in chunk.items():

                    # Merge data
                    final_state.update(agent_data)

                    # Update Indicators dynamically
                    if agent_name == "ticker_resolver":
                        p_resolve.markdown(render_agent_status("Ticker Resolver", "done"), unsafe_allow_html=True)
                        p_fin.markdown(render_agent_status("Financials Agent", "running"), unsafe_allow_html=True)

                    elif agent_name == "financials_agent":
                        p_fin.markdown(render_agent_status("Financials Agent", "done"), unsafe_allow_html=True)
                        p_market.markdown(render_agent_status("Market Data Agent", "running"), unsafe_allow_html=True)

                    elif agent_name == "market_data_agent":
                        p_market.markdown(render_agent_status("Market Data Agent", "done"), unsafe_allow_html=True)
                        p_news.markdown(render_agent_status("News Agent", "running"), unsafe_allow_html=True)

                    elif agent_name == "news_agent":
                        p_news.markdown(render_agent_status("News Agent", "done"), unsafe_allow_html=True)
                        p_details.markdown(render_agent_status("Company Details", "running"), unsafe_allow_html=True)

                    elif agent_name == "company_details_agent":
                        p_details.markdown(render_agent_status("Company Details", "done"), unsafe_allow_html=True)
                        p_analyst.markdown(render_agent_status("Master Analyst", "running"), unsafe_allow_html=True)

                    elif agent_name == "master_analyst":
                        p_analyst.markdown(render_agent_status("Master Analyst", "done"), unsafe_allow_html=True)
        # Save to session state
        if final_state:
            st.session_state['data'] = final_state
            st.rerun()  # This reloads the page, and Section 4 above will now render everything as "Green/Done"

    except Exception as e:
        st.error(f"Error: {str(e)}")
        p_analyst.markdown(render_agent_status("Master Analyst", "error"), unsafe_allow_html=True)
        st.stop()

# 6. MAIN CONTENT RENDERING
st.title(" AI Multi-Agent Market Analyst")
st.caption("Global Financial Intelligence | Powered by LangGraph & Gemini")

if 'data' in st.session_state and st.session_state['data']:
    data = st.session_state['data']

    # --- SAFE DATA EXTRACTION ---
    # We use .get() everywhere to prevent crashes if AI misses a field
    report = data.get('final_report', {})
    metrics = data.get('financial_data', {}).get('metrics', {})
    rec = report.get('recommendation', 'HOLD').upper()
    confidence = report.get('confidence_score', 0)  # e.g. 85
    sentiment = report.get('sentiment_score', 50)  # e.g. 75
    volatility = report.get('volatility', 'Medium')
    company_detail = report.get("company_details", {})


    # Determine Colors based on Signal
    if "BUY" in rec:
        rec_color_class = "signal-card-green"
    elif "SELL" in rec:
        rec_color_class = "signal-card-red"
    else:
        rec_color_class = "signal-card-orange"
    st.markdown("##### General Information")
    coll1, coll2, coll3, coll4 = st.columns(4)
    with coll1:
        st.markdown(f"""
            <div class="info-card">
                <div class="signal-title">CEO</div>
                <div class="signal-value">{company_detail.get("CEO", "N/A")}</div>
            </div>
        """, unsafe_allow_html=True)

    with coll2:
        st.markdown(f"""
            <div class="info-card">
                <div class="signal-title">Founded</div>
                <div class="signal-value">{company_detail.get("founded")}</div>
            </div>
        """, unsafe_allow_html=True)

    with coll3:
        st.markdown(f"""
            <div class="info-card">
                <div class="signal-title">Industry</div>
                <div class="signal-value">{company_detail.get("industry")}</div>
            </div>
        """, unsafe_allow_html=True)

    with coll4:
        st.markdown(f"""
            <div class="info-card">
                <div class="signal-title">Sector</div>
                <div class="signal-value">{company_detail.get("sector")}</div>
            </div>
        """, unsafe_allow_html=True)

    # --- SECTION 1: DYNAMIC QUANTITATIVE SIGNALS ---
    st.markdown("##### Section 1 — Quantitative Signals")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
            <div class="{rec_color_class}">
                <div class="signal-title">Recommendation</div>
                <div class="signal-value">{rec}</div>
                <div style="font-size: 11px; margin-top:5px;"> </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        # Determine sentiment color (Green > 50, Red < 50)
        bar_color = "#28a745" if sentiment >= 50 else "#dc3545"
        st.markdown(f"""
            <div class="metric-card" style="height: 100px;">
                <div style="display:flex; justify-content:space-between;">
                    <span class="signal-title">Sentiment Score</span>
                    <span>{"" if sentiment >= 50 else ""}</span>
                </div>
                <div class="signal-value">{sentiment}<span style="font-size:16px; color:#999;">/100</span></div>
                <div class="progress-container">
                    <div class="progress-bar" style="width: {sentiment}%; background-color: {bar_color};"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="metric-card" style="height: 100px;">
                <div class="signal-title">Confidence %</div>
                <div class="signal-value">{confidence}%</div>
                <div style="font-size: 12px; color: green;">↑ AI Confidence</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class="signal-card-orange">
                <div class="signal-title">Volatility Flag</div>
                <div class="signal-value" style="font-size: 22px;">{volatility}</div>
                <div style="font-size: 11px; margin-top:5px;">〰 Historical volatility observed.</div>
            </div>
        """, unsafe_allow_html=True)

    # --- SECTION 2: FUNDAMENTALS GRID ---
    st.markdown("##### Section 2 — Key Fundamentals Grid")

    # Row 1
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        create_fundamental_card("", "Market Cap", metrics.get("Market Cap", "N/A"))
    with r1c2:
        create_fundamental_card("", "PE Ratio", metrics.get("P/E Ratio", "N/A"))
    with r1c3:
        create_fundamental_card("", "EPS TTM", metrics.get("EPS TTM", "N/A"))
    with r1c4:
        create_fundamental_card("", "Beta", metrics.get("Beta", "N/A"))

    # Row 2
    st.write("")
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        create_fundamental_card("", "52W High", metrics.get("52W High", "N/A"))
    with r2c2:
        create_fundamental_card("", "52W Low", metrics.get("52W Low", "N/A"))
    with r2c3:
        create_fundamental_card("", "Volume", metrics.get("Volume", "N/A"))
    with r2c4:
        create_fundamental_card("", "Dividend", metrics.get("Dividend", "N/A"))

    # --- SECTION 3: CHARTS & REPORT ---
    st.write("")
    col_chart, col_report = st.columns([2, 1])

    with col_chart:
        st.markdown("##### Stock Price History (1 Year)")

        # Get Real Price Data
        history = data.get('market_data', {}).get('history_data', [])

        if history:
            df_hist = pd.DataFrame(history)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_hist['Date'],
                y=df_hist['Close'],
                mode='lines',
                name='Close',
                line=dict(color='#0066cc', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 102, 204, 0.1)'
            ))

            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    type="date"
                ),
                yaxis=dict(title="Price ($)", showgrid=True, gridcolor='#f0f0f0'),
                margin=dict(l=0, r=0, t=0, b=0),
                height=350,
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No historical price data available.")

    with col_report:
        st.markdown("#####  Analyst Report")
        with st.container(border=True):
            st.markdown("**Multi-Agent Consensus Report**")
            st.caption(f"Date: {datetime.now().strftime('%Y-%m-%d')}")

            # DYNAMIC SUMMARY
            summary = report.get('summary', "No summary available.")
            st.write(summary)

            st.divider()
            st.markdown("** SWOT Analysis**")

            swot = report.get('swot', {})

            # Using Plain Text bullets to fix formatting
            st.markdown("**Strengths**")
            for item in swot.get('strengths', [])[:3]:
                st.markdown(f"• {item}")

            st.markdown("**Risks / Threats**")
            for item in swot.get('threats', [])[:3]:
                st.markdown(f"• {item}")

else:
    # Landing Page State
    st.info(" Enter a company name in the sidebar and click 'Generate Analysis' to begin.")