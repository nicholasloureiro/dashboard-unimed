import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser
import os
from datetime import datetime, timedelta
import json
import re
from functools import lru_cache
import time

# Classe personalizada para exibir gráficos no Streamlit
class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)
        
    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return result
    
    def format_plot(self, result):
        if isinstance(result["value"], plt.Figure):
            # Convert matplotlib figure to Plotly figure
            import io
            import base64
            
            # Save matplotlib figure to a buffer
            buf = io.BytesIO()
            result["value"].savefig(buf, format='png', transparent=True)
            buf.seek(0)
            
            # Display as image
            st.image(buf, use_column_width=True)
        elif isinstance(result["value"], str) and result["value"].startswith("data:image"):
            # Handle base64 encoded images
            st.image(result["value"])
        elif isinstance(result["value"], str):
            # Check for Plotly JSON in the string
            match = re.search(r'<plotly>(.*?)</plotly>', result["value"], re.DOTALL)
            if match:
                try:
                    fig_json = json.loads(match.group(1))
                    fig = go.Figure(fig_json)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error parsing Plotly figure: {e}")
                    # Fallback: display the raw content
                    st.write(result["value"])
            elif result["value"].endswith(".html") or "temp_chart" in result["value"]:
                # Handle HTML files, especially temp_chart.html
                try:
                    # Check if it's a file path
                    if os.path.exists(result["value"]):
                        with open(result["value"], 'r') as f:
                            html_content = f.read()
                        # Render HTML content
                        st.components.v1.html(html_content, height=400)
                    else:
                        # If not a file path, try to render as HTML directly
                        st.components.v1.html(result["value"], height=400)
                except Exception as e:
                    st.error(f"Error rendering HTML: {e}")
                    st.write(result["value"])
            else:
                # For any other string output
                try:
                    # Try to interpret the string as JSON (may contain a Plotly figure)
                    try:
                        json_data = json.loads(result["value"])
                        if isinstance(json_data, dict) and "data" in json_data and "layout" in json_data:
                            # This looks like a Plotly figure
                            fig = go.Figure(json_data)
                            st.plotly_chart(fig, use_container_width=True)
                            return result
                    except json.JSONDecodeError:
                        pass
                    
                    # Try to display as an image
                    try:
                        st.image(result["value"])
                    except:
                        # Fall back to displaying as text
                        st.write(result["value"])
                except Exception as e:
                    st.write(result["value"])
        else:
            # For other types like Plotly figures
            try:
                if hasattr(result["value"], "to_html") or hasattr(result["value"], "update_layout"):
                    # This is likely a Plotly figure
                    st.plotly_chart(result["value"], use_container_width=True)
                else:
                    # For anything else, use the generic write method
                    st.write(result["value"])
            except Exception as e:
                st.error(f"Error displaying plot: {e}")
                st.write(result["value"])
        return result
    
    def format_other(self, result):
        # Check if the result might be a Plotly figure
        try:
            if isinstance(result["value"], str):
                # Try to extract Plotly JSON if enclosed in tags
                match = re.search(r'<plotly>(.*?)</plotly>', result["value"], re.DOTALL)
                if match:
                    try:
                        fig_json = json.loads(match.group(1))
                        fig = go.Figure(fig_json)
                        st.plotly_chart(fig, use_container_width=True)
                        return result
                    except Exception:
                        # If extraction fails, continue with normal display
                        pass
        except Exception:
            pass
        
        # Default display
        st.write(result["value"])
        return result


def real_br_money_mask(my_value):
    a = '{:,.2f}'.format(float(my_value))
    b = a.replace(',','v')
    c = b.replace('.',',')
    return c.replace('v','.')

# Função para conectar ao banco de dados SQLite
def get_data():
    conn = sqlite3.connect("medical_data.db")
    query = """
    SELECT a.*, 
           p.name AS provider_name, p.type AS provider_type, 
           pt.name AS patient_name, pt.age AS patient_age,
           pr.code AS procedure_code, pr.name AS procedure_name,
           m.code AS material_code, m.name AS material_name, 
           md.code AS medication_code, md.name AS medication_name, 
           hs.admission_date, hs.discharge_date, hs.department,
           ptc.name AS protocol_name, r.score AS recommendation_score
    FROM alerts a
    LEFT JOIN providers p ON a.provider_id = p.provider_id
    LEFT JOIN patients pt ON a.patient_id = pt.patient_id
    LEFT JOIN procedures pr ON a.procedure_id = pr.procedure_id
    LEFT JOIN materials m ON a.material_id = m.material_id
    LEFT JOIN medications md ON a.medication_id = md.medication_id
    LEFT JOIN hospitalizations hs ON a.hospitalization_id = hs.hospitalization_id
    LEFT JOIN protocols ptc ON pr.protocol_id = ptc.protocol_id
    LEFT JOIN recommendations r ON a.patient_id = r.patient_id 
                                AND a.provider_id = r.provider_id 
                                AND a.hospital_id = r.hospital_id
    """  # Ajuste conforme necessário
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Carregar dados
df = get_data()

# Configuração da página
st.set_page_config(page_title="Dashboard Unimed", layout="wide")

# Custom CSS to match the corporate design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-color: #009C6E;
        --secondary-color: #f8f9fa;
        --accent-color: #00A651;
        --text-color: #2c3e50;
        --light-gray: #e9ecef;
        --medium-gray: #adb5bd;
        --dark-gray: #495057;
        --danger-color: #dc3545;
        --warning-color: #ffc107;
        --success-color: #28a745;
        --info-color: #17a2b8;
        --border-radius: 15px;
        --box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Global styles */
    .main, .stApp {
        background-color: #f5f7fa;
        color: var(--text-color);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1wrcr25 {
        background-color: black !important;
        border-right: 1px solid var(--light-gray);
        box-shadow: var(--box-shadow);
    }
    
    /* Header */
    h1 {
        font-size: 24px;
        font-weight: 600;
        color: var(--text-color);
        letter-spacing: -0.5px;
    }
    
    /* KPI Cards */
    .kpi-card {
        background-color: white;
        border-radius: var(--border-radius);
        padding: 20px;
        box-shadow: var(--box-shadow);
        margin-bottom: 20px;
        border-top: 3px solid var(--primary-color);
        transition: transform 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
    }
    
    .kpi-title {
        font-size: 13px;
        color: var(--dark-gray);
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 8px;
        color: var(--text-color);
    }
    
    .kpi-trend {
        font-size: 12px;
        color: var(--dark-gray);
        display: flex;
        align-items: center;
    }
    
    .kpi-trend.positive {
        color: var(--success-color);
    }
    
    .kpi-trend.negative {
        color: var(--danger-color);
    }
    
    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
        font-size: 16px;
        font-weight: 600;
        color: var(--text-color);
        border-bottom: 1px solid var(--light-gray);
        padding-bottom: 8px;
    }
    
    .section-header i {
        margin-right: 10px;
        color: var(--primary-color);
    }
    
    /* Insights */
    .insight-item {
        border-left: 3px solid var(--primary-color);
        padding: 12px 16px;
        margin-bottom: 15px;
        background-color: white;
        border-radius: 0 var(--border-radius) var(--border-radius) 0;
        box-shadow: var(--box-shadow);
    }
    
    .insight-text {
        font-size: 14px;
        margin-bottom: 5px;
        line-height: 1.5;
    }
    
    .insight-meta {
        font-size: 12px;
        color: var(--medium-gray);
    }
    
    /* Alerts Table */
    .dataframe {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: var(--box-shadow);
    }
    
    .dataframe th, .dataframe td {
        padding: 12px 16px;
        text-align: left;
        border-bottom: 1px solid var(--light-gray);
    }
    
    .dataframe th {
        font-weight: 600;
        color: var(--dark-gray);
        font-size: 13px;
        background-color: #f8f9fa;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .dataframe td {
        font-size: 14px;
        background-color: white;
    }
    
    .dataframe tr:hover td {
        background-color: #f5f7fa;
    }
    
    /* Search bar */
    .search-container {
        margin-bottom: 16px;
        
    }
    
    /* Example questions */
    .example-questions {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 20px;
    }
    
    .example-question {
        background-color: #f0f4f8;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 13px;
        color: var(--dark-gray);
        cursor: pointer;
        border: 1px solid var(--light-gray);
        transition: all 0.2s ease;
    }
    
    .example-question:hover {
        background-color: #e0e7ff;
        border-color: #c7d2fe;
        transform: translateY(-2px);
    }
    
    .example-question.selected {
        background-color: var(--primary-color);
        color: white !important;
        border-color: var(--primary-color);
    }
    
    /* Filter */
    .filter-container {
        margin-bottom: 20px;
        background-color: white;
        padding: 16px;
        border-radius: var(--border-radius);
        box-shadow: var(--box-shadow);
    }
    
    .filter-label {
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
        color: var(--dark-gray);
    }
    
    /* Hide Streamlit elements */
    #MainMenu, footer, header {
        visibility: hidden;
    }
    
    div.block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    
    /* Chart styling */
    .chart-container {
        background-color: white;
        border-radius: var(--border-radius);
        padding: 20px;
        box-shadow: var(--box-shadow);
    }
    
    /* Value formatting */
    .value-positive {
        color: var(--success-color);
        font-weight: 500;
    }
    
    .value-negative {
        color: var(--danger-color);
        font-weight: 500;
    }
    
    .value-risk {
        color: var(--danger-color);
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #4CAF50;
        color: white !important;
        border: none;
        border-radius: var(--border-radius);
        padding: 8px 16px;
        font-weight: 500;
        transition: background-color 0.2s ease;
    }
    
    .stButton button:hover {
        background-color: #007a56;  /* Darker green for hover */
        color: white;
    }
    
    /* Input fields */
    .stTextInput input {
        border-radius: var(--border-radius);
        border: 1px solid var(--light-gray);
        padding: 8px 12px;
    }
    
    .stTextInput input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px rgba(0,86,179,0.2);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)
# Sidebar with Unimed logo
with st.sidebar:
    st.sidebar.image("unimed.png", width=300)

    st.markdown(
    """
    <style>
        [data-testid="stSidebar"] img {
            margin-top: -100px;  /* Adjust this value as needed */
            margin-bottom: -200px;
            margin-left: -40px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
    
    st.markdown(
    """
    <hr style="border: 1px solid #009C6E; margin-top:-13px; margin-bottom: 15px; ">
    """,
    unsafe_allow_html=True,
)

    # Sidebar menu
    st.markdown("""
    <div style="padding: 5px 0;">
        <div style="display: flex; align-items: center; padding: 10px; background-color: var(--light-gray); border-radius: 6px; border-left: 3px solid #009C6E; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <i class="fas fa-chart-line" style="color: #009C6E; margin-right: 10px;"></i>
            <span style="color: #2c3e50; font-size: 14px; font-weight: 500;">Dashboard</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main content
st.markdown('<h1 style="font-size: 24px; margin-bottom: 25px; color: #2c3e50; font-weight: 600; border-bottom: 2px solid #009C6E; padding-bottom: 10px; margin-top:-20px;">Dashboard</h1>', unsafe_allow_html=True)

# Initialize the session state for selected question
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'selected_question' not in st.session_state:
    st.session_state.selected_question = None

# Define the example questions
example_questions = [
    "Me mostre onde eu estou perdendo mais receita",
    "Liste os usuários que geram fraudes de maior valor",
    "Faça um gráfico com os 5 provedores com mais fraudes"
]

# Initialize session state if not already set
if "query" not in st.session_state:
    st.session_state.query = ""
if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

# Search bar
col1, col2 = st.columns([6, 1])

# Place the text input in the first column
with col1:
    user_query = st.text_input(
        "",
        value=st.session_state.query,
        placeholder="O que gostaria de saber?",
        label_visibility="collapsed",
        key="user_query_input"
    )

    # Update session state when input changes
    if user_query != st.session_state.query:
        st.session_state.query = user_query

# Place the button in the second column
with col2:
    search_button = st.button("Perguntar ✨", type="primary")


# Create columns for example questions (3 per row)
question_cols = st.columns(3)

# Create clickable example questions
for i, question in enumerate(example_questions):
    with question_cols[i % 3]:  # Distribute buttons into columns
        # Create a unique key for each button
        question_key = f"question_{i}"

        # Determine if this question is currently selected
        is_selected = st.session_state.selected_question == i

        # Create a button for each example question
        if st.button(
            question,
            key=question_key,
            type="secondary",
            use_container_width=True,
            help=f"Clique para perguntar: {question}"
        ):
            # When clicked:
            st.session_state.query = question  # Update query
            st.session_state.selected_question = i  # Mark as selected
            st.rerun()  # Rerun UI

# Add CSS to style the selected button
if st.session_state.selected_question is not None:
    selected_idx = st.session_state.selected_question
    st.markdown(f"""
        <style>
            [data-testid="baseButton-secondary"]:nth-of-type({selected_idx + 1}) {{
                background-color: #00A651 !important;
                color: white !important;
            }}
        </style>
    """, unsafe_allow_html=True)

# Configuração do PandasAI
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    llm = OpenAI(api_token=api_key, model="gpt-4", temperature=0,
    system_message=("Você é um assistente de análise de dados médicos. "
                "Responda SEMPRE em português brasileiro, em tom profissional mas acessível. "
                "Não use termos em inglês a menos que sejam termos técnicos sem tradução adequada. "
                "Dê respostas concisas e diretas, focadas nos dados. "
                "Quando for solicitado a criar gráficos, SEMPRE use a biblioteca Plotly e não matplotlib. "
                "Para todos os gráficos, use a cor #009C6E como cor principal. "
                "Sempre retorne o código do gráfico Plotly dentro de tags <plotly></plotly> para que ele seja renderizado corretamente."
            ))
    smart_df = SmartDataframe(df, config={
        "llm": llm,
        "language": "pt-br",
        "response_parser": StreamlitResponse  # Adicionando o novo parser
    })
    
    if user_query and search_button:
        st.markdown("""
        <div class="section">
            <div class="section-header" style="background-color: white; padding: 12px; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 15px; display: flex; align-items: center;">
                <i class="fas fa-robot" style="color: #009C6E; font-size: 16px; margin-right: 10px;"></i>
                <span style="font-size: 16px; font-weight: 600; color: #2c3e50;">Resposta da IA</span>
            </div>
      """, unsafe_allow_html=True)
        
        try:
            # Modify the query to ensure Plotly is used for charts
            if "gráfico" in user_query.lower() or "grafico" in user_query.lower() or "visualização" in user_query.lower() or "visualizacao" in user_query.lower():
                user_query += " Use Plotly para criar o gráfico com a cor #009C6E como cor principal e retorne o código dentro de tags <plotly></plotly>"
            
            # Enviar pergunta ao PandasAI
            # Com o StreamlitResponse configurado, ele já irá renderizar o resultado apropriadamente
            smart_df.chat(f"Responda em portugues: {user_query}")
            # Não precisamos fazer nada adicional aqui, pois o parser já trata a exibição
        except Exception as e:
            st.error(f"Erro ao processar a pergunta: {str(e)}")
            
        st.markdown("</div></div>", unsafe_allow_html=True)
else:
    st.error("API Key não encontrada. Configure a variável de ambiente OPENAI_API_KEY.")

# Function to query SQLite database
def query_db(query, params=()):
    conn = sqlite3.connect("medical_data.db")
    result = conn.execute(query, params).fetchone()[0]
    conn.close()
    return result if result else 0  # Avoid None values

# Define session state for date selection
if "start_date" not in st.session_state:
    st.session_state.start_date = datetime.today() - timedelta(days=7)
if "end_date" not in st.session_state:
    st.session_state.end_date = datetime.today()

st.markdown(
    """
    <style>
        /* Target the date input box */
        [data-testid="stSidebar"] [data-baseweb="input"] {
            background-color: var(--light-gray) !important;
            border-radius: 5px;
            border: 1px solid #ccc !important;
        }

        /* Change focus border color from red to green */
        [data-testid="stSidebar"] [data-baseweb="input"]:focus {
            border: 2px solid #009C6E !important;
            box-shadow: 0 0 5px #009C6E !important;
        }

        /* Target the calendar popup container */
        [role="dialog"] {
            background-color: var(--light-gray) !important;
            border: 1px solid #ccc !important;
            border-radius: 8px !important;
        }

        /* Fix selected date red color */
        [role="gridcell"][aria-selected="true"] {
            background-color: #009C6E !important;
            color: white !important;
            border-radius: 50% !important;
        }

        /* Stronger override for red background */
        [role="gridcell"][aria-selected="true"]::after {
            background-color: #009C6E !important;
            border-radius: 50% !important;
        }

        /* Remove red hover effect */
        [role="gridcell"]:hover {
            background-color: #00b482 !important;
            color: black !important;
            border-radius: 50% !important;
        }

        /* Fix today's date highlight color */
        [role="gridcell"][aria-current="date"] {
            border: 2px solid #009C6E !important;
            border-radius: 50% !important;
            color: black !important;
        }

        /* Ensure all text remains visible */
        [role="gridcell"] {
            color: #333 !important;
        }

    </style>
    """,
    unsafe_allow_html=True
)



st.sidebar.markdown("### Selecione o Período")
start_date = st.sidebar.date_input("Data Inicial", st.session_state.start_date)
end_date = st.sidebar.date_input("Data Final", st.session_state.end_date)

# Ensure start_date is before end_date
if start_date > end_date:
    st.sidebar.error("A data inicial não pode ser maior que a data final.")
else:
    # Update session state
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date

    # Convert dates to datetime format for database queries
    start_date_dt = datetime.combine(start_date, datetime.min.time())
    end_date_dt = datetime.combine(end_date, datetime.max.time())

    # Previous period calculation (same duration before start_date)
    period_duration = end_date_dt - start_date_dt
    start_date_previous = start_date_dt - period_duration
    end_date_previous = start_date_dt

    # Queries
    queries = {
        "Alertas Ativos": "SELECT COUNT(*) FROM alerts WHERE created_at BETWEEN ? AND ? AND alert_status = 'Ativo'",
        "Alertas Ativos (Last Period)": "SELECT COUNT(*) FROM alerts WHERE created_at BETWEEN ? AND ? AND alert_status = 'Ativo'",
        "Taxa de Confirmação": "SELECT ROUND(AVG(is_anomaly) * 100, 2) FROM alerts WHERE created_at BETWEEN ? AND ? AND alert_status = 'Ativo'",
        "Taxa de Confirmação (Last Period)": "SELECT ROUND(AVG(is_anomaly) * 100, 2) FROM alerts WHERE created_at BETWEEN ? AND ? AND alert_status = 'Ativo'",
        "Risco Total": "SELECT ROUND(SUM(risk_value), 2) FROM alerts WHERE created_at BETWEEN ? AND ? AND alert_status = 'Ativo'",
        "Risco Total (Last Period)": "SELECT ROUND(SUM(risk_value), 2) FROM alerts WHERE created_at BETWEEN ? AND ? AND alert_status = 'Ativo'"
    }

    # Fetch data dynamically
    current_alerts = query_db(queries["Alertas Ativos"], (start_date_dt, end_date_dt))
    previous_alerts = query_db(queries["Alertas Ativos (Last Period)"], (start_date_previous, end_date_previous))
    alerts_delta = current_alerts - previous_alerts

    current_confirmation = query_db(queries["Taxa de Confirmação"], (start_date_dt, end_date_dt))
    previous_confirmation = query_db(queries["Taxa de Confirmação (Last Period)"], (start_date_previous, end_date_previous))
    confirmation_delta = round(current_confirmation - previous_confirmation, 2)

    current_risk = query_db(queries["Risco Total"], (start_date_dt, end_date_dt))
    previous_risk = query_db(queries["Risco Total (Last Period)"], (start_date_previous, end_date_previous))
    risk_delta = round(current_risk - previous_risk, 2)

    # KPI Cards in a simpler style with dynamic data
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">ALERTAS ATIVOS</div>
            <div class="kpi-value">{current_alerts}</div>
            <div class="kpi-trend {'positive' if alerts_delta < 0 else 'negative'}">{alerts_delta:+}% vs período anterior</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">TAXA DE CONFIRMAÇÃO</div>
            <div class="kpi-value">{current_confirmation}%</div>
            <div class="kpi-trend {'positive' if confirmation_delta > 0 else 'negative'}">{confirmation_delta:+}% vs período anterior</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">RISCO TOTAL</div>
            <div class="kpi-value">R$ {real_br_money_mask(current_risk)}</div>
            <div class="kpi-trend {'positive' if risk_delta < 0 else 'negative'}">{risk_delta:+} vs período anterior</div>
        </div>
        """, unsafe_allow_html=True)

# Create two columns for side-by-side layout
col1, col2 = st.columns(2)

# First container with insights - using PandasAI
with col1:
    st.markdown("""
    <div class="section-header" style="background-color: white; padding: 12px; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 15px;">
        <i class="fas fa-lightbulb" style="color: #009C6E; font-size: 16px;"></i>
        <span style="font-size: 16px; font-weight: 600; color: #2c3e50;">Insights</span>
    </div>
    """, unsafe_allow_html=True)
    
    if api_key:
        try:
            insights_query = "Liste os principais insights dos dados de alertas médicos, em portugues e em markdown, nao utilie graficos"
            # Não precisamos armazenar o resultado, pois o parser já trata a exibição
            smart_df.chat(insights_query)
        except Exception as e:
            st.error(f"Erro ao gerar insights: {str(e)}")

# Função para criar gráfico de distribuição de alertas com cache
@lru_cache(maxsize=10)
def create_alert_distribution_chart(df_json):
    """
    Cria um gráfico de distribuição de alertas por tipo com cache para melhor desempenho.
    O parâmetro df_json é uma string JSON do dataframe para permitir o cache.
    """
    start_time = time.time()
    
    # Converter JSON para DataFrame
    df = pd.read_json(df_json)
    
    # Calcular a distribuição
    alert_counts = df['alert_type'].value_counts().reset_index()
    alert_counts.columns = ['Tipo de Alerta', 'Contagem']
    total = alert_counts['Contagem'].sum()
    alert_counts['Porcentagem'] = (alert_counts['Contagem'] / total * 100).round(1)
    
    # Criar gráfico otimizado
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=alert_counts['Tipo de Alerta'],
        y=alert_counts['Porcentagem'],
        marker_color='#009C6E',
        text=alert_counts['Porcentagem'].apply(lambda x: f'{x}%'),
        textposition='auto'
    ))
    
    # Otimizar layout para melhor desempenho
    fig.update_layout(
        title='Distribuição de Alertas por Tipo (%)',
        xaxis_title='Tipo de Alerta',
        yaxis_title='Porcentagem (%)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='closest',
        xaxis=dict(tickangle=-45),
        yaxis=dict(range=[0, max(alert_counts['Porcentagem']) * 1.1])
    )
    
    # Otimizar configuração para renderização mais rápida
    config = {
        'staticPlot': True,  # Modo estático para melhor desempenho
        'displayModeBar': False,
        'responsive': True
    }
    
    end_time = time.time()
    print(f"Tempo para gerar gráfico: {end_time - start_time:.2f} segundos")
    
    return fig, config

# Second container with distribution chart - using cached function
with col2:
    st.markdown("""
    <div class="section-header" style="background-color: white; padding: 12px; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 15px;">
        <i class="fas fa-chart-bar" style="color: #009C6E; font-size: 16px;"></i>
        <span style="font-size: 16px; font-weight: 600; color: #2c3e50;">Distribuição de Alertas por Tipo</span>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Usar a função com cache para melhor desempenho
        with st.spinner("Gerando gráfico..."):
            # Converter DataFrame para JSON para permitir cache
            df_json = df.to_json()
            fig, config = create_alert_distribution_chart(df_json)
            st.plotly_chart(fig, use_container_width=True, config=config)
    except Exception as e:
        st.error(f"Erro ao gerar gráfico: {str(e)}")
        
        # Fallback para PandasAI se o método otimizado falhar
        if api_key:
            try:
                chart_query = "Crie um gráfico de barras simples usando Plotly mostrando a distribuição em porcentagem dos tipos de alertas. Use a cor #009C6E para as barras. Coloque os elementos do gráfico em português e use background transparente. Retorne o código do gráfico dentro de tags <plotly></plotly>"
                smart_df.chat(chart_query)
            except Exception as e2:
                st.error(f"Erro ao gerar gráfico alternativo: {str(e2)}")

# Alertas em tempo real
st.markdown("""
<div class="section-header" style="margin-top: 25px; background-color: white; padding: 12px; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 15px;">
    <i class="fas fa-bell" style="color: #009C6E; font-size: 16px;"></i>
    <span style="font-size: 16px; font-weight: 600; color: #2c3e50;">Alertas em Tempo Real</span>
</div>
""", unsafe_allow_html=True)

# Use real data from the database
alertas = df[["alert_type", "description", "risk_value"]].rename(
    columns={"alert_type": "Nome", "description": "Descrição", "risk_value": "Valor em risco"}
)

# Keep the risk value as a float without formatting
# This will allow Streamlit to sort and filter the column properly
st.dataframe(alertas, use_container_width=True, hide_index=True)

# Add Font Awesome
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)