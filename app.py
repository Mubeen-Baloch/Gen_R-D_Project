"""
Streamlit Frontend for Uncertainty-Aware Scientific Claim Synthesis.
Provides a comprehensive dashboard for interacting with the multi-agent framework.
"""

import os
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from loguru import logger
from streamlit_agraph import agraph, Node, Edge, Config

# ─── Configure Logging ─────────────
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("./data/pipeline.log", level="DEBUG")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import get_settings
from src.knowledge_graph.dskg import DSKG
from src.orchestration.autonomy_loop import AutonomyLoop
from src.stores.claim_store import ClaimStore
from src.stores.conflict_registry import ConflictRegistry
from src.stores.gap_store import GapStore
from src.evaluation.metrics import MetricsEvaluator

# ─── Page Configuration ─────────────
st.set_page_config(
    page_title="Uncertainty-Aware Synthesis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-1px);
    }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2563eb;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .gap-card {
        background-color: white;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .conflict-card {
        background-color: white;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────
if "settings" not in st.session_state:
    st.session_state.settings = get_settings()
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "Topic Input"

# ─── Helper Functions ─────────────
def run_pipeline(topic: str, max_papers: int, iterations: int):
    """Run the autonomy loop in a separate thread to prevent UI blocking."""
    try:
        settings = st.session_state.settings
        settings.max_papers = max_papers
        settings.max_autonomy_iterations = iterations
        
        loop = AutonomyLoop(settings)
        state = loop.execute(topic)
        st.session_state.pipeline_state = state
        st.session_state.pipeline_error = None
    except Exception as e:
        logger.exception("Pipeline failed")
        st.session_state.pipeline_error = str(e)
    finally:
        st.session_state.is_running = False

def render_metric_card(label: str, value: str):
    """Render a styled metric card."""
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
    """, unsafe_allow_html=True)

# ─── Sidebar Navigation ─────────────
st.sidebar.title("🔬 Research Agent")
st.sidebar.markdown("Uncertainty-Aware Scientific Claim Synthesis")

pages = [
    "Topic Input",
    "Dashboard",
    "Knowledge Graph",
    "Conflict Registry",
    "Gap Explorer",
    "Literature Review",
    "Evaluation",
]

for page in pages:
    if st.sidebar.button(
        page, 
        use_container_width=True, 
        type="primary" if st.session_state.current_page == page else "secondary"
    ):
        st.session_state.current_page = page
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Configuration")
api_key = st.sidebar.text_input("Gemini API Key", value=st.session_state.settings.google_api_key, type="password")
if api_key != st.session_state.settings.google_api_key:
    st.session_state.settings.google_api_key = api_key
    st.sidebar.success("API Key updated!")

# ─── Page: Topic Input ─────────────
if st.session_state.current_page == "Topic Input":
    st.title("Generate Literature Review")
    st.markdown("Enter a research topic to start the multi-agent synthesis pipeline.")
    
    with st.form("topic_form"):
        topic = st.text_input("Research Topic", "Retrieval-Augmented Generation in Medical NLP")
        
        col1, col2 = st.columns(2)
        with col1:
            max_papers = st.slider("Max Papers to Analyze", 5, 50, 15)
        with col2:
            iterations = st.slider("Autonomy Iterations", 1, 5, 2)
            
        submitted = st.form_submit_button("Start Synthesis")
        
    if submitted:
        if not st.session_state.settings.google_api_key:
            st.error("Please configure your Gemini API Key in the sidebar.")
        else:
            st.session_state.is_running = True
            st.session_state.pipeline_state = None
            
            # Start background thread
            thread = threading.Thread(
                target=run_pipeline, 
                args=(topic, max_papers, iterations)
            )
            # Add Streamlit context to thread
            from streamlit.runtime.scriptrunner import add_script_run_ctx
            add_script_run_ctx(thread)
            thread.start()
            
            st.session_state.current_page = "Dashboard"
            st.rerun()

# ─── Page: Dashboard ─────────────
elif st.session_state.current_page == "Dashboard":
    st.title("System Dashboard")
    
    if st.session_state.is_running:
        st.info("Pipeline is currently running. This may take several minutes depending on the number of papers.")
        with st.spinner("Agents are analyzing the literature..."):
            # Poll for updates
            time.sleep(2)
            st.rerun()
            
    elif getattr(st.session_state, "pipeline_error", None):
        st.error(f"Pipeline failed: {st.session_state.pipeline_error}")
        
    elif st.session_state.pipeline_state:
        state = st.session_state.pipeline_state
        st.success(f"Synthesis complete for: **{state.topic}**")
        
        # Summary Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("Papers", str(len(state.papers)))
        with col2:
            render_metric_card("Claims Extracted", str(len(state.claims)))
        with col3:
            render_metric_card("Contradictions", str(len(state.conflicts)))
        with col4:
            render_metric_card("Gaps Identified", str(len(state.gaps)))
            
        st.markdown("---")
        
        # Agent Messages Log
        st.subheader("Agent Execution Log")
        log_df = pd.DataFrame([
            {
                "Time": msg.timestamp.strftime("%H:%M:%S"),
                "Sender": msg.sender,
                "Recipient": msg.recipient,
                "Type": msg.message_type.value
            }
            for msg in state.message_bus[-20:] # Last 20 messages
        ])
        if not log_df.empty:
            st.dataframe(log_df, use_container_width=True, hide_index=True)
            
    else:
        st.warning("No synthesis data available. Please go to 'Topic Input' and run the pipeline.")

# ─── Page: Knowledge Graph ─────────────
elif st.session_state.current_page == "Knowledge Graph":
    st.title("Dynamic Scientific Knowledge Graph")
    
    if not st.session_state.pipeline_state:
        st.warning("No data available. Run the pipeline first.")
    else:
        # Try to load DSKG
        dskg_path = st.session_state.settings.dskg_path
        if Path(dskg_path).exists():
            dskg = DSKG(dskg_path)
            dskg.load()
            
            stats = dskg.get_stats()
            col1, col2 = st.columns(2)
            with col1:
                render_metric_card("Total Nodes", str(stats["total_nodes"]))
            with col2:
                render_metric_card("Total Edges", str(stats["total_edges"]))
                
            st.markdown("### Graph Visualization")
            
            # Subgraph selection to prevent browser crash
            node_type = st.selectbox("Center Graph On", ["Concept", "Method", "Claim"], index=0)
            
            # Extract subgraph for visualization
            nodes = []
            edges = []
            
            # Limit nodes to avoid rendering lag
            max_nodes = 100
            count = 0
            
            colors = {
                "concept": "#3b82f6",
                "method": "#10b981",
                "claim": "#f59e0b",
                "paper": "#6b7280",
            }
            
            for node_id, data in dskg.graph.nodes(data=True):
                if count >= max_nodes:
                    break
                ntype = data.get("node_type", "unknown")
                label = data.get("name", data.get("title", data.get("claim_text", node_id)[:30]))
                
                nodes.append(Node(
                    id=node_id,
                    label=label,
                    size=25 if ntype == "concept" else 15,
                    color=colors.get(ntype, "#9ca3af"),
                ))
                count += 1
                
            for u, v, data in dskg.graph.edges(data=True):
                if any(n.id == u for n in nodes) and any(n.id == v for n in nodes):
                    edges.append(Edge(
                        source=u,
                        target=v,
                        label=data.get("edge_type", ""),
                        color="#d1d5db"
                    ))
                    
            config = Config(
                width="100%",
                height=600,
                directed=True,
                physics=True,
                hierarchical=False,
            )
            
            agraph(nodes=nodes, edges=edges, config=config)
            
        else:
            st.warning("DSKG file not found.")

# ─── Page: Conflict Registry ─────────────
elif st.session_state.current_page == "Conflict Registry":
    st.title("Conflict Registry")
    st.markdown("Contested claims and methodological disputes automatically identified across the literature.")
    
    if not st.session_state.pipeline_state or not st.session_state.pipeline_state.conflicts:
        st.warning("No conflicts identified or pipeline not run.")
    else:
        conflicts = st.session_state.pipeline_state.conflicts
        
        # Filters
        types = list(set(c.conflict_type.value for c in conflicts))
        selected_types = st.multiselect("Filter by Conflict Type", types, default=types)
        
        filtered_conflicts = [c for c in conflicts if c.conflict_type.value in selected_types]
        
        for i, conflict in enumerate(filtered_conflicts):
            st.markdown(f"""
            <div class="conflict-card">
                <h3>{conflict.conflict_type.value.replace("_", " ").title()} Conflict</h3>
                <p><strong>Claim A (Paper {conflict.source_paper_a_id}):</strong> {conflict.pair.claim_a_text}</p>
                <p><strong>Claim B (Paper {conflict.source_paper_b_id}):</strong> {conflict.pair.claim_b_text}</p>
                <hr>
                <p><strong>Resolution:</strong> {conflict.reconciliation_statement}</p>
            </div>
            """, unsafe_allow_html=True)

# ─── Page: Gap Explorer ─────────────
elif st.session_state.current_page == "Gap Explorer":
    st.title("Formalized Research Gaps")
    st.markdown("Evidence-grounded gaps mapped to falsifiable hypotheses.")
    
    if not st.session_state.pipeline_state or not st.session_state.pipeline_state.gaps:
        st.warning("No gaps identified or pipeline not run.")
    else:
        gaps = st.session_state.pipeline_state.gaps
        
        # Sort by confidence
        gaps = sorted(gaps, key=lambda g: g.confidence, reverse=True)
        
        for i, gap in enumerate(gaps):
            st.markdown(f"""
            <div class="gap-card">
                <h3>{gap.gap_type.value.replace("_", " ").title()} ({gap.gap_class.value})</h3>
                <p><strong>Confidence Score:</strong> {gap.confidence:.2f}</p>
                <p><strong>Statement:</strong> {gap.gap_statement}</p>
                <p><strong>Falsifiability:</strong> {gap.falsifiability}</p>
                <p><small><strong>Topic Cluster:</strong> {gap.topic_cluster}</small></p>
            </div>
            """, unsafe_allow_html=True)

# ─── Page: Literature Review ─────────────
elif st.session_state.current_page == "Literature Review":
    st.title("Synthesized Literature Review")
    
    if not st.session_state.pipeline_state or not st.session_state.pipeline_state.review:
        st.warning("No review generated or pipeline not run.")
    else:
        review = st.session_state.pipeline_state.review
        
        # Action buttons
        col1, col2 = st.columns([1, 5])
        with col1:
            st.download_button(
                "📥 Download Markdown",
                data=str(review),
                file_name=f"review_{review.topic.replace(' ', '_')}.md",
                mime="text/markdown",
            )
            
        st.markdown("---")
        
        # Display Review
        st.markdown(f"# Literature Review: {review.topic}")
        
        for section in review.sections:
            st.markdown(f"## {section.title}")
            st.markdown(section.content)
            
        st.markdown(review.contested_claims_section)
        st.markdown(review.research_gaps_section)
        st.markdown(review.temporal_narrative)
        
        st.markdown("## Conclusion")
        st.markdown(review.conclusion)

# ─── Page: Evaluation ─────────────
elif st.session_state.current_page == "Evaluation":
    st.title("System Evaluation Metrics")
    st.markdown("Quantitative assessment of the synthesized output.")
    
    if not st.session_state.pipeline_state:
        st.warning("No data available. Run the pipeline first.")
    else:
        state = st.session_state.pipeline_state
        metrics = MetricsEvaluator.get_all_metrics(state)
        
        # Radar Chart
        df = pd.DataFrame(dict(
            r=list(metrics.values()),
            theta=list(metrics.keys())
        ))
        
        fig = px.line_polar(df, r='r', theta='theta', line_close=True, range_r=[0, 1.0])
        fig.update_traces(fill='toself', line_color='#2563eb')
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Metric Scores")
            for name, score in metrics.items():
                st.metric(name, f"{score:.2f}", help="Score from 0.0 to 1.0")
                
        # Critic Evaluations
        if state.evaluations:
            st.markdown("---")
            st.subheader("Critic Agent Feedback Iterations")
            
            for eval in state.evaluations:
                with st.expander(f"Iteration {eval.iteration} - Score: {eval.overall_quality:.2f}", expanded=(eval.iteration == state.current_iteration)):
                    st.markdown(f"**Feedback:** {eval.feedback}")
                    st.markdown("**Directives:**")
                    for d in eval.improvement_directives:
                        st.markdown(f"- {d}")
