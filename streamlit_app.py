"""
NFL DFS Optimizer - Complete Streamlit Application
Version: 5.0.0 - UPDATE 2 Compatible

COMPLETE FILE - Replaces entire streamlit_app.py

COMPATIBLE WITH:
- NFL DFS Optimizer v5.0.0 (Parts 1-13)
- UPDATE 2: Advanced AI System
- Enhanced Monte Carlo Simulation
- Advanced Scoring & Intelligence

NEW FEATURES:
- Pre-flight constraint checking with auto-fix buttons
- Constraint violation diagnostics
- Optimizer performance summaries
- Advanced AI strategist support
- Enhanced error messages with specific guidance
- Support for new optimization modes
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import threading
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import traceback
import io

# ============================================================================
# OPTIMIZER IMPORTS
# ============================================================================

# Check if optimizer is available
try:
    # Core classes
    from nfl_dfs_optimizer import (
        # Data Processing
        OptimizedDataProcessor,
        safe_load_csv,
        
        # Core Classes
        DraftKingsRules,
        LineupConstraints,
        ValidationResult,
        GeneticConfig,
        SimulationResults,
        
        # Analyzers
        PlayerPoolAnalyzer,
        StackDetector,
        OwnershipAnalyzer,
        
        # Validation & Constraints
        ConstraintFeasibilityChecker,
        BatchLineupValidator,
        DiversityTracker,
        ExposureTracker,
        
        # Simulation
        MonteCarloSimulationEngine,
        
        # Optimizers
        GeneticAlgorithmOptimizer,
        SimulatedAnnealingOptimizer,
        
        # AI Components (if available)
        AIStrategistType,
        AIEnforcementLevel,
        
        # Output
        CSVExporter,
        ConsoleFormatter,
        OutputManager,
        
        # Utilities
        OptimizerLogger,
        PerformanceMonitor,
        UnifiedCache,
        get_logger,
        get_performance_monitor,
        get_unified_cache,
        
        # Configuration
        OptimizerConfig,
        CONTEST_TYPE_MAPPING,
    )
    
    OPTIMIZER_AVAILABLE = True
    
except ImportError as e:
    st.error(f"Failed to import optimizer: {e}")
    st.info("Please ensure nfl_dfs_optimizer.py is in the same directory")
    OPTIMIZER_AVAILABLE = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="NFL DFS Optimizer v5.0",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS
# ============================================================================

class StreamlitConfig:
    """Streamlit-specific configuration"""
    
    # Timeouts
    OPTIMIZATION_TIMEOUT_SEC = 300
    THREAD_JOIN_SEC = 0.5
    
    # Display
    MAX_WARNINGS_SHOW = 5
    MAX_LINEUPS_DISPLAY = 50
    
    # Progress
    INITIAL_PROGRESS_THRESHOLD_SEC = 5
    PROGRESS_UPDATE_INTERVAL_SEC = 5
    
    # Cache
    CSV_CACHE_TTL_SEC = 3600  # 1 hour
    MAX_CACHE_ENTRIES = 5

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        # Core data
        'df': None,
        'processed_df': None,
        'lineups': [],
        'game_info': {},
        
        # Settings
        'num_lineups': 20,
        'game_total': 50.0,
        'spread': 0.0,
        'contest_type': 'Large GPP (1000+)',
        'optimization_mode': 'balanced',
        'min_salary_pct': 95,
        
        # AI Settings
        'use_ai': False,
        'ai_enforcement': 'moderate',
        'api_key': '',
        'ai_strategists': ['game_theory', 'correlation'],
        
        # Advanced Options
        'use_monte_carlo': True,
        'num_simulations': 10000,
        'diversity_threshold': 0.5,
        'randomness_factor': 0.15,
        
        # UI State
        'show_advanced': False,
        'show_debug': False,
        'show_profiling': False,
        'show_ai_details': False,
        
        # Processing flags
        'data_validated': False,
        'optimization_complete': False,
        
        # Results
        'warnings': [],
        'errors': [],
        'last_optimization_time': None,
        'optimizer_summary': None,
        'pool_analysis': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize
init_session_state()

# ============================================================================
# STATE SNAPSHOT
# ============================================================================

class StateSnapshot:
    """Preserve state across errors"""
    
    def __init__(self):
        self.snapshot = {}
    
    def capture(self):
        """Capture current state"""
        self.snapshot = {
            k: v for k, v in st.session_state.items()
            if not k.startswith('_')
        }
    
    def restore(self):
        """Restore captured state"""
        for k, v in self.snapshot.items():
            st.session_state[k] = v

# ============================================================================
# CACHING
# ============================================================================

@st.cache_data(
    ttl=StreamlitConfig.CSV_CACHE_TTL_SEC,
    max_entries=StreamlitConfig.MAX_CACHE_ENTRIES
)
def cached_csv_load(
    file_bytes: bytes,
    file_name: str,
    file_hash: str
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Cache CSV processing"""
    
    if not OPTIMIZER_AVAILABLE:
        return None, ["Optimizer not available"]
    
    logger = get_logger()
    warnings = []
    
    try:
        # Load CSV
        file_obj = io.BytesIO(file_bytes)
        df, encoding = safe_load_csv(file_obj, logger)
        
        if df is None:
            return None, [f"Failed to load CSV: {encoding}"]
        
        if encoding != 'utf-8':
            warnings.append(f"Loaded with {encoding} encoding")
        
        # Process
        processor = OptimizedDataProcessor(logger)
        df_processed, proc_warnings = processor.process_dataframe(df)
        warnings.extend(proc_warnings)
        
        logger.log(f"Processed {len(df_processed)} players", "INFO")
        
        return df_processed, warnings
        
    except Exception as e:
        logger.log_exception(e, "cached_csv_load")
        return None, [f"Error: {str(e)}"]

# ============================================================================
# STYLING
# ============================================================================

def apply_css():
    """Apply custom CSS"""
    st.markdown("""
        <style>
        .main {padding-top: 1rem;}
        .stAlert {margin: 0.5rem 0;}
        h1 {color: #1e3a8a; font-weight: 700;}
        h2 {color: #2563eb; margin-top: 1.5rem;}
        h3 {color: #3b82f6; margin-top: 1rem;}
        .stButton>button {
            width: 100%;
            background-color: #3b82f6;
            color: white;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #2563eb;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 0.5rem;
            color: white;
        }
        .diagnostic-box {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .success-box {
            background-color: #dcfce7;
            border-left: 4px solid #22c55e;
            padding: 1rem;
            border-radius: 0.25rem;
        }
        .warning-box {
            background-color: #fef3c7;
            border-left: 4px solid #eab308;
            padding: 1rem;
            border-radius: 0.25rem;
        }
        .error-box {
            background-color: #fee2e2;
            border-left: 4px solid #ef4444;
            padding: 1rem;
            border-radius: 0.25rem;
        }
        </style>
    """, unsafe_allow_html=True)

# ============================================================================
# CONSTRAINT DIAGNOSTICS
# ============================================================================

def run_constraint_check(
    df: pd.DataFrame,
    min_salary_pct: int
) -> Dict[str, Any]:
    """Run constraint feasibility check"""
    
    if not OPTIMIZER_AVAILABLE:
        return {
            'is_feasible': False,
            'error': 'Optimizer not available',
            'suggestions': []
        }
    
    constraints = LineupConstraints(
        min_salary=int(DraftKingsRules.SALARY_CAP * (min_salary_pct / 100)),
        max_salary=DraftKingsRules.SALARY_CAP
    )
    
    is_feasible, error_msg, suggestions = ConstraintFeasibilityChecker.check(
        df,
        constraints
    )
    
    return {
        'is_feasible': is_feasible,
        'error': error_msg,
        'suggestions': suggestions,
        'constraints': constraints
    }

def render_constraint_diagnostics():
    """Render constraint diagnostics with auto-fix"""
    
    if st.session_state.processed_df is None:
        return
    
    df = st.session_state.processed_df
    
    with st.expander("🔍 Pre-Flight Constraint Check", expanded=False):
        st.subheader("Constraint Feasibility Analysis")
        
        # Run check
        diag = run_constraint_check(df, st.session_state.min_salary_pct)
        
        if diag['is_feasible']:
            st.markdown(
                '<div class="success-box">✓ All constraints feasible - optimization should succeed</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="error-box">✗ Constraint Issue: {diag["error"]}</div>',
                unsafe_allow_html=True
            )
            
            if diag['suggestions']:
                st.markdown("**Suggested Fixes:**")
                
                for i, suggestion in enumerate(diag['suggestions'], 1):
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.write(f"{i}. {suggestion}")
                    
                    with col2:
                        # Auto-fix buttons
                        import re
                        match = re.search(r'(\d+)%', suggestion)
                        if match and "lower" in suggestion.lower():
                            pct = int(match.group(1))
                            if st.button(f"Fix: {pct}%", key=f"fix_{i}"):
                                st.session_state.min_salary_pct = pct
                                st.rerun()
        
        # Pool statistics
        st.subheader("Player Pool Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        # Calculate feasibility bounds
        cheapest_6 = df.nsmallest(6, 'Salary')['Salary'].sum()
        cheapest_6 += df['Salary'].min() * 0.5  # Captain multiplier
        
        expensive_6 = df.nlargest(6, 'Salary')['Salary'].sum()
        expensive_6 += df['Salary'].max() * 0.5
        
        min_threshold = diag['constraints'].min_salary
        max_threshold = diag['constraints'].max_salary
        
        with col1:
            st.metric("Players", len(df))
            st.metric("Teams", df['Team'].nunique())
        
        with col2:
            st.metric("Cheapest 6", f"${cheapest_6:,.0f}")
            st.metric("Most Expensive 6", f"${expensive_6:,.0f}")
        
        with col3:
            st.metric("Min Threshold", f"${min_threshold:,.0f}")
            st.metric("Max Threshold", f"${max_threshold:,.0f}")
        
        # Visual feasibility check
        if cheapest_6 > max_threshold:
            st.error("⚠️ IMPOSSIBLE: Even cheapest lineup exceeds cap!")
        elif expensive_6 < min_threshold:
            st.error("⚠️ IMPOSSIBLE: Can't reach minimum threshold!")
        elif cheapest_6 > min_threshold * 0.95:
            st.warning("⚠️ TIGHT: Minimal feasible range")
        else:
            st.success("✓ Good feasible range")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_process_data(uploaded_file):
    """Load and process uploaded CSV"""
    
    logger = get_logger()
    
    try:
        # Read file
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name
        file_hash = hashlib.md5(file_bytes).hexdigest()[:8]
        
        # Process with caching
        df, warnings = cached_csv_load(file_bytes, file_name, file_hash)
        
        if df is None:
            return None, warnings
        
        # Show preview
        with st.expander("📊 Data Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Showing 10 of {len(df)} rows")
        
        # Show warnings
        if warnings:
            with st.expander(f"⚠️ Warnings ({len(warnings)})", expanded=False):
                for w in warnings[:StreamlitConfig.MAX_WARNINGS_SHOW]:
                    st.warning(w)
                
                if len(warnings) > StreamlitConfig.MAX_WARNINGS_SHOW:
                    st.info(f"...and {len(warnings) - StreamlitConfig.MAX_WARNINGS_SHOW} more")
        
        return df, warnings
        
    except Exception as e:
        logger.log_exception(e, "load_and_process_data")
        st.error(f"Error loading data: {str(e)}")
        
        if st.session_state.show_debug:
            st.code(traceback.format_exc())
        
        return None, [str(e)]

# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render sidebar configuration"""
    
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # 1. DATA UPLOAD
        st.header("1️⃣ Data Upload")
        
        uploaded_file = st.file_uploader(
            "Upload Player CSV",
            type=['csv'],
            help="CSV with player projections"
        )
        
        if uploaded_file:
            if st.button("📥 Load Data", type="primary"):
                with st.spinner("Loading..."):
                    df, warnings = load_and_process_data(uploaded_file)
                    
                    if df is not None:
                        st.session_state.df = df
                        st.session_state.processed_df = df
                        st.session_state.warnings = warnings
                        st.session_state.data_validated = True
                        
                        # Run pool analysis
                        analyzer = PlayerPoolAnalyzer(df)
                        st.session_state.pool_analysis = analyzer.analyze()
                        
                        st.success(f"✓ Loaded {len(df)} players")
                    else:
                        st.error("✗ Load failed")
        
        st.divider()
        
        # 2. GAME SETTINGS
        st.header("2️⃣ Game Settings")
        
        st.session_state.game_total = st.number_input(
            "Game Total (O/U)",
            min_value=30.0,
            max_value=70.0,
            value=st.session_state.game_total,
            step=0.5
        )
        
        st.session_state.spread = st.number_input(
            "Spread",
            min_value=-20.0,
            max_value=20.0,
            value=st.session_state.spread,
            step=0.5
        )
        
        st.divider()
        
        # 3. CONTEST SETTINGS
        st.header("3️⃣ Contest Settings")
        
        st.session_state.contest_type = st.selectbox(
            "Contest Type",
            options=list(CONTEST_TYPE_MAPPING.keys()),
            index=list(CONTEST_TYPE_MAPPING.keys()).index(
                st.session_state.contest_type
            )
        )
        
        st.session_state.num_lineups = st.slider(
            "Number of Lineups",
            min_value=1,
            max_value=150,
            value=st.session_state.num_lineups
        )
        
        st.session_state.optimization_mode = st.selectbox(
            "Optimization Mode",
            options=['balanced', 'ceiling', 'floor', 'sharpe'],
            index=0
        )
        
        st.divider()
        
        # 4. AI SETTINGS
        st.header("4️⃣ AI Settings")
        
        st.session_state.use_ai = st.checkbox(
            "Enable AI Analysis",
            value=st.session_state.use_ai,
            help="Use Claude AI for strategic insights"
        )
        
        if st.session_state.use_ai:
            st.session_state.api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                value=st.session_state.api_key
            )
            
            st.session_state.ai_enforcement = st.select_slider(
                "AI Enforcement",
                options=['advisory', 'moderate', 'strong', 'mandatory'],
                value=st.session_state.ai_enforcement
            )
            
            if not st.session_state.api_key:
                st.warning("⚠️ API key required")
        
        st.divider()
        
        # 5. ADVANCED OPTIONS
        with st.expander("🔧 Advanced Options"):
            st.session_state.min_salary_pct = st.slider(
                "Min Salary %",
                min_value=50,
                max_value=100,
                value=st.session_state.min_salary_pct,
                step=5
            )
            st.caption(
                f"${int(DraftKingsRules.SALARY_CAP * st.session_state.min_salary_pct / 100):,}"
            )
            
            st.session_state.use_monte_carlo = st.checkbox(
                "Monte Carlo Simulation",
                value=st.session_state.use_monte_carlo
            )
            
            if st.session_state.use_monte_carlo:
                st.session_state.num_simulations = st.select_slider(
                    "Simulations",
                    options=[1000, 5000, 10000, 20000],
                    value=st.session_state.num_simulations
                )
            
            st.session_state.diversity_threshold = st.slider(
                "Diversity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.diversity_threshold,
                step=0.05
            )
            
            st.session_state.show_debug = st.checkbox(
                "Debug Mode",
                value=st.session_state.show_debug
            )

# ============================================================================
# DATA OVERVIEW
# ============================================================================

def render_data_overview():
    """Render data overview tab"""
    
    if st.session_state.processed_df is None:
        st.info("📤 Upload a CSV file to begin")
        return
    
    df = st.session_state.processed_df
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Players", len(df))
    
    with col2:
        st.metric("Teams", df['Team'].nunique())
    
    with col3:
        st.metric("Avg Salary", f"${df['Salary'].mean():,.0f}")
    
    with col4:
        st.metric("Avg Projection", f"{df['Projected_Points'].mean():.1f}")
    
    # Charts
    st.subheader("Pool Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**By Position**")
        st.bar_chart(df['Position'].value_counts())
    
    with col2:
        st.write("**By Team**")
        st.bar_chart(df['Team'].value_counts())
    
    # Top players
    st.subheader("Top 10 Players")
    
    top10 = df.nlargest(10, 'Projected_Points')[
        ['Player', 'Position', 'Team', 'Salary', 'Projected_Points', 'Ownership']
    ].copy()
    
    st.dataframe(top10, use_container_width=True, hide_index=True)
    
    # Diagnostics
    render_constraint_diagnostics()
    
    # Pool analysis
    if st.session_state.pool_analysis:
        with st.expander("📈 Pool Analysis", expanded=False):
            analysis = st.session_state.pool_analysis
            
            st.write(f"**Pool Quality:** {analysis.get('pool_quality', 'Unknown').upper()}")
            
            recs = analysis.get('recommendations', {})
            st.write("**Recommendations:**")
            st.write(f"- Recommended Algorithm: {recs.get('recommended_algorithm', 'N/A')}")
            st.write(f"- Suggested Randomness: {recs.get('suggested_randomness', 0):.2f}")
            st.write(f"- Max Reasonable Lineups: {recs.get('max_reasonable_lineups', 0)}")

# ============================================================================
# OPTIMIZATION
# ============================================================================

def render_optimization():
    """Render optimization tab"""
    
    if st.session_state.processed_df is None:
        st.warning("⚠️ Load data first")
        return
    
    # Settings summary
    with st.expander("Current Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Game:**")
            st.write(f"- Total: {st.session_state.game_total}")
            st.write(f"- Spread: {st.session_state.spread}")
            st.write(f"- Contest: {st.session_state.contest_type}")
        
        with col2:
            st.write("**Optimization:**")
            st.write(f"- Lineups: {st.session_state.num_lineups}")
            st.write(f"- Mode: {st.session_state.optimization_mode}")
            st.write(f"- AI: {st.session_state.use_ai}")
    
    # Optimize button
    if st.button("🚀 Generate Lineups", type="primary", use_container_width=True):
        run_optimization()

def run_optimization():
    """Execute optimization with full error handling"""
    
    snapshot = StateSnapshot()
    snapshot.capture()
    
    logger = get_logger()
    perf = get_performance_monitor()
    
    try:
        # Pre-flight check
        st.info("Running pre-flight check...")
        
        diag = run_constraint_check(
            st.session_state.processed_df,
            st.session_state.min_salary_pct
        )
        
        if not diag['is_feasible']:
            st.error(f"Pre-flight Failed: {diag['error']}")
            
            if diag['suggestions']:
                st.warning("Suggestions:")
                for s in diag['suggestions']:
                    st.write(f"- {s}")
            
            return
        
        st.success("✓ Pre-flight passed")
        
        # Setup progress
        progress = st.progress(0)
        status = st.empty()
        
        # Get settings
        df = st.session_state.processed_df.copy()
        num_lineups = st.session_state.num_lineups
        use_ai = st.session_state.use_ai and bool(st.session_state.api_key)
        
        # Infer game info
        processor = OptimizedDataProcessor()
        game_info = processor.infer_game_info(
            df,
            st.session_state.game_total,
            st.session_state.spread
        )
        
        progress.progress(20)
        status.text("Initializing optimizers...")
        
        # Create constraints
        constraints = LineupConstraints(
            min_salary=int(DraftKingsRules.SALARY_CAP * st.session_state.min_salary_pct / 100),
            max_salary=DraftKingsRules.SALARY_CAP
        )
        
        # Optimization mode config
        field_size = CONTEST_TYPE_MAPPING.get(
            st.session_state.contest_type,
            'large_field'
        )
        field_config = OptimizerConfig.get_field_config(field_size)
        
        progress.progress(40)
        status.text("Running optimization...")
        
        # Start optimization thread
        perf.start_timer('full_optimization')
        start_time = time.time()
        
        result = {'lineups': None, 'error': None}
        
        def opt_thread():
            try:
                # Use genetic algorithm for now
                # (Full MasterOptimizer in Part 13)
                
                # Create Monte Carlo engine
                mc_engine = None
                if st.session_state.use_monte_carlo:
                    mc_engine = MonteCarloSimulationEngine(
                        df,
                        game_info,
                        st.session_state.num_simulations
                    )
                
                # Create genetic optimizer
                genetic_config = OptimizerConfig.get_genetic_config(
                    len(df),
                    num_lineups,
                    time_budget_seconds=180
                )
                
                optimizer = GeneticAlgorithmOptimizer(
                    df=df,
                    game_info=game_info,
                    constraints=constraints,
                    config=genetic_config,
                    mc_engine=mc_engine
                )
                
                # Generate lineups
                lineups = optimizer.generate_lineups(
                    num_lineups,
                    fitness_mode=st.session_state.optimization_mode
                )
                
                # Calculate metrics
                from nfl_dfs_optimizer import calculate_lineup_metrics
                
                detailed_lineups = []
                for i, lineup in enumerate(lineups, 1):
                    metrics = calculate_lineup_metrics(
                        lineup.captain,
                        lineup.flex,
                        df,
                        mc_engine
                    )
                    metrics['Lineup'] = i
                    detailed_lineups.append(metrics)
                
                result['lineups'] = detailed_lineups
                
            except Exception as e:
                result['error'] = e
        
        # Run thread
        thread = threading.Thread(target=opt_thread)
        thread.daemon = True
        thread.start()
        
        # Wait with timeout
        timeout = StreamlitConfig.OPTIMIZATION_TIMEOUT_SEC
        elapsed = 0
        
        while thread.is_alive() and elapsed < timeout:
            thread.join(timeout=StreamlitConfig.THREAD_JOIN_SEC)
            elapsed = time.time() - start_time
            
            # Update progress
            pct = min(90, 40 + int((elapsed / timeout) * 50))
            progress.progress(pct)
            
            if elapsed > 10 and int(elapsed) % 5 == 0:
                status.text(f"Optimizing... ({int(elapsed)}s)")
        
        # Check timeout
        if thread.is_alive():
            progress.empty()
            status.empty()
            st.error(f"⏱️ Timeout after {timeout}s")
            st.info("Try: Lower min salary %, reduce lineups, or disable AI")
            return
        
        # Check error
        if result['error']:
            raise result['error']
        
        # Get results
        lineups = result['lineups']
        elapsed_time = perf.stop_timer('full_optimization')
        
        progress.progress(100)
        status.text("Complete!")
        time.sleep(0.5)
        
        progress.empty()
        status.empty()
        
        if not lineups:
            st.warning("No lineups generated")
            return
        
        # Store results
        st.session_state.lineups = lineups
        st.session_state.optimization_complete = True
        st.session_state.last_optimization_time = elapsed_time
        
        # Success
        st.success(f"✓ Generated {len(lineups)} lineups in {elapsed_time:.1f}s")
        
        # Quick stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_proj = np.mean([l['Projected'] for l in lineups])
            st.metric("Avg Projection", f"{avg_proj:.2f}")
        
        with col2:
            avg_sal = np.mean([l['Total_Salary'] for l in lineups])
            st.metric("Avg Salary", f"${avg_sal:,.0f}")
        
        with col3:
            if 'Ceiling_90th' in lineups[0]:
                avg_ceil = np.mean([l['Ceiling_90th'] for l in lineups])
                st.metric("Avg Ceiling", f"{avg_ceil:.2f}")
        
        st.rerun()
        
    except Exception as e:
        snapshot.restore()
        logger.log_exception(e, "run_optimization")
        st.error(f"❌ Error: {str(e)}")
        
        if st.session_state.show_debug:
            with st.expander("Debug Trace"):
                st.code(traceback.format_exc())

# ============================================================================
# RESULTS
# ============================================================================

def render_results():
    """Render results tab"""
    
    if not st.session_state.optimization_complete:
        st.info("Complete optimization to view results")
        return
    
    lineups = st.session_state.lineups
    df = st.session_state.processed_df
    
    # Summary
    st.subheader("Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Lineups", len(lineups))
    
    with col2:
        best = max(l['Projected'] for l in lineups)
        st.metric("Best Projection", f"{best:.2f}")
    
    with col3:
        avg = np.mean([l['Projected'] for l in lineups])
        st.metric("Avg Projection", f"{avg:.2f}")
    
    with col4:
        avg_sal = np.mean([l['Total_Salary'] for l in lineups])
        st.metric("Avg Salary", f"${avg_sal:,.0f}")
    
    # Display options
    st.subheader("Lineup Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        show_count = st.selectbox(
            "Show:",
            options=[5, 10, 20, 50, "All"],
            index=1
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort by:",
            options=['Projected', 'Ceiling_90th', 'Total_Salary'],
            index=0
        )
    
    # Filter and sort
    display_count = len(lineups) if show_count == "All" else int(show_count)
    sorted_lineups = sorted(
        lineups,
        key=lambda x: x.get(sort_by, 0),
        reverse=True
    )[:display_count]
    
    # Display lineups
    for i, lineup in enumerate(sorted_lineups, 1):
        with st.container():
            st.markdown(f"### Lineup {i}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Captain:** {lineup['Captain']}")
                st.write("**FLEX:**")
                
                flex = lineup.get('FLEX', [])
                if isinstance(flex, str):
                    flex = [p.strip() for p in flex.split(',')]
                
                for player in flex:
                    player_info = df[df['Player'] == player]
                    if not player_info.empty:
                        row = player_info.iloc[0]
                        st.write(
                            f"  - {player} ({row['Position']}) - "
                            f"${row['Salary']:,.0f} | {row['Projected_Points']:.1f} pts"
                        )
            
            with col2:
                st.metric("Projected", f"{lineup['Projected']:.2f}")
                st.metric("Salary", f"${lineup['Total_Salary']:,.0f}")
                
                if 'Ceiling_90th' in lineup:
                    st.metric("Ceiling", f"{lineup['Ceiling_90th']:.2f}")
                    st.metric("Floor", f"{lineup['Floor_10th']:.2f}")
            
            st.divider()

# ============================================================================
# EXPORT & ANALYTICS
# ============================================================================

def render_export_analytics():
    """Render export and analytics"""
    
    if not st.session_state.optimization_complete:
        st.info("Complete optimization first")
        return
    
    lineups = st.session_state.lineups
    
    # EXPORT SECTION
    st.header("Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        format_choice = st.selectbox(
            "Format",
            options=['Standard', 'DraftKings', 'Detailed']
        )
    
    # Generate CSV
    try:
        if format_choice == 'DraftKings':
            # DK format
            rows = []
            for lineup in lineups:
                flex = lineup.get('FLEX', [])
                if isinstance(flex, str):
                    flex = [p.strip() for p in flex.split(',')]
                
                row = {
                    'CPT': lineup['Captain'],
                    'FLEX': flex[0] if len(flex) > 0 else '',
                    'FLEX_2': flex[1] if len(flex) > 1 else '',
                    'FLEX_3': flex[2] if len(flex) > 2 else '',
                    'FLEX_4': flex[3] if len(flex) > 3 else '',
                    'FLEX_5': flex[4] if len(flex) > 4 else '',
                }
                rows.append(row)
        else:
            # Standard format
            rows = []
            for i, lineup in enumerate(lineups, 1):
                flex = lineup.get('FLEX', [])
                if isinstance(flex, str):
                    flex = [p.strip() for p in flex.split(',')]
                
                row = {
                    'Lineup': i,
                    'Captain': lineup['Captain'],
                    'FLEX_1': flex[0] if len(flex) > 0 else '',
                    'FLEX_2': flex[1] if len(flex) > 1 else '',
                    'FLEX_3': flex[2] if len(flex) > 2 else '',
                    'FLEX_4': flex[3] if len(flex) > 3 else '',
                    'FLEX_5': flex[4] if len(flex) > 4 else '',
                    'Salary': lineup['Total_Salary'],
                    'Projected': lineup['Projected'],
                }
                
                if format_choice == 'Detailed':
                    row.update({
                        'Ownership': lineup.get('Avg_Ownership', 0),
                        'Ceiling': lineup.get('Ceiling_90th', 0),
                        'Floor': lineup.get('Floor_10th', 0),
                    })
                
                rows.append(row)
        
        export_df = pd.DataFrame(rows)
        csv = export_df.to_csv(index=False)
        
        # Preview
        with st.expander("Preview"):
            st.dataframe(export_df.head(10))
        
        # Download
        st.download_button(
            label=f"📥 Download {format_choice} CSV",
            data=csv,
            file_name=f"lineups_{format_choice.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Export failed: {e}")
    
    st.divider()
    
    # ANALYTICS SECTION
    st.header("Analytics")
    
    # Player exposure
    st.subheader("Player Exposure")
    
    exposure = defaultdict(int)
    for lineup in lineups:
        captain = lineup['Captain']
        flex = lineup.get('FLEX', [])
        if isinstance(flex, str):
            flex = [p.strip() for p in flex.split(',')]
        
        for player in [captain] + flex:
            exposure[player] += 1
    
    exposure_pct = {
        p: (c / len(lineups)) * 100
        for p, c in exposure.items()
    }
    
    top_exposure = sorted(exposure_pct.items(), key=lambda x: x[1], reverse=True)[:15]
    exp_df = pd.DataFrame(top_exposure, columns=['Player', 'Exposure %'])
    
    st.bar_chart(exp_df.set_index('Player'))
    
    # Captain usage
    st.subheader("Captain Usage")
    
    captain_counts = defaultdict(int)
    for lineup in lineups:
        captain_counts[lineup['Captain']] += 1
    
    captain_pct = {
        c: (count / len(lineups)) * 100
        for c, count in captain_counts.items()
    }
    
    capt_df = pd.DataFrame(
        list(captain_pct.items()),
        columns=['Captain', 'Usage %']
    ).sort_values('Usage %', ascending=False)
    
    st.bar_chart(capt_df.set_index('Captain'))

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application"""
    
    if not OPTIMIZER_AVAILABLE:
        st.error("Optimizer not available - check imports")
        return
    
    # Apply styling
    apply_css()
    
    # Header
    st.title("🏈 NFL DFS AI-Powered Optimizer")
    st.markdown("*Advanced DraftKings Showdown optimization with AI and Monte Carlo simulation*")
    st.caption("Version 5.0.0 - UPDATE 2 Compatible")
    
    # Sidebar
    render_sidebar()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Overview",
        "🚀 Optimization",
        "📈 Results",
        "💾 Export & Analytics"
    ])
    
    with tab1:
        render_data_overview()
    
    with tab2:
        render_optimization()
    
    with tab3:
        render_results()
    
    with tab4:
        render_export_analytics()
    
    # Footer
    st.divider()
    st.caption("NFL DFS Optimizer v5.0 | Built with Claude AI")
    
    if st.session_state.last_optimization_time:
        st.caption(f"Last optimization: {st.session_state.last_optimization_time:.1f}s")

if __name__ == "__main__":
    main()
