"""
NFL DFS Optimizer - Complete Streamlit Application
Version: 3.2.1 - Context Manager Fix

COMPLETE FILE - Replaces entire streamlit_app.py

FIXED: Context manager protocol error in StreamlitStateSnapshot
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import threading
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Literal
from itertools import combinations
import traceback
import io

# Import from enhanced optimizer
from nfl_dfs_optimizer import (
    # Core functions
    optimize_showdown,
    safe_load_csv,
    validate_and_normalize_dataframe,
    validate_lineup_with_context,
    format_lineup_for_export,
    validate_export_format,
    calculate_lineup_similarity,
    ensure_lineup_diversity,
    
    # Classes
    OptimizedDataProcessor,
    MasterOptimizer,
    ValidationResult,
    StructuredLogger,
    PerformanceMonitor,
    PerformanceProfiler,
    LineupConstraints,
    
    # Enums
    ExportFormat,
    ValidationLevel,
    OptimizationMode,
    
    # Constants
    OptimizerConfig,
    DraftKingsRules,
    StreamlitConstants,
    CONTEST_TYPE_MAPPING,
    FIELD_SIZE_CONFIGS,
    
    # Utilities
    get_logger,
    get_performance_monitor,
    get_profiler,
    
    # Type hints
    OptimizationModeType,
    AIEnforcementType,
    ExportFormatType,
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="NFL DFS Optimizer",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

def ensure_session_state():
    """Initialize all session state variables at module level"""
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
        'use_ai': False,
        'ai_enforcement': 'Moderate',
        'api_key': '',
        'min_salary_pct': 95,
        
        # UI state
        'show_advanced': False,
        'show_debug': False,
        'show_profiling': False,
        
        # Processing flags
        'data_validated': False,
        'optimization_complete': False,
        
        # Warnings and errors
        'warnings': [],
        'errors': [],
        
        # Performance tracking
        'last_optimization_time': None,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize immediately
ensure_session_state()

# ============================================================================
# STATE SNAPSHOT FOR ERROR RECOVERY (FIXED)
# ============================================================================

class StreamlitStateSnapshot:
    """Preserve Streamlit state across errors with thread safety"""
    
    def __init__(self):
        self.snapshot: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def capture(self, keys: Optional[List[str]] = None) -> None:
        """Capture current session state"""
        with self._lock:
            if keys is None:
                keys = [k for k in st.session_state.keys() if not k.startswith('_')]
            
            self.snapshot = {
                key: st.session_state.get(key)
                for key in keys
            }
    
    def restore(self) -> None:
        """Restore captured state"""
        with self._lock:
            for key, value in self.snapshot.items():
                st.session_state[key] = value

# ============================================================================
# STREAMLIT CACHING DECORATORS
# ============================================================================

@st.cache_data(
    ttl=StreamlitConstants.Cache.CSV_PROCESSING_TTL_SEC,
    max_entries=5
)
def cached_csv_processing(
        file_bytes: bytes,
        file_name: str,
        file_hash: str
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Cache expensive CSV processing for 1 hour"""
    logger = get_logger()

    # Manual timing instead of profiler context manager
    start_time = time.time()

    file_obj = io.BytesIO(file_bytes)
    df_raw, encoding_info = safe_load_csv(file_obj, logger)

    if df_raw is None:
        return None, [f"Failed to load CSV: {encoding_info}"]

    warnings = []
    if encoding_info != 'utf-8':
        warnings.append(f"File loaded with {encoding_info} encoding")

    processor = OptimizedDataProcessor()
    df_processed, process_warnings = processor.process_dataframe(
        df_raw,
        ValidationLevel.MODERATE
    )

    warnings.extend(process_warnings)

    # Log timing
    elapsed = time.time() - start_time
    logger.log(f"CSV processing completed in {elapsed:.2f}s", "INFO")

    return df_processed, warnings

# ============================================================================
# STYLING
# ============================================================================

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
        <style>
        .main {padding-top: 2rem;}
        .stAlert {margin-top: 1rem; margin-bottom: 1rem;}
        h1 {color: #1f4788;}
        h2 {color: #2c5aa0; margin-top: 2rem;}
        h3 {color: #4a90e2;}
        .stButton>button {width: 100%;}
        .stProgress > div > div > div {background-color: #4a90e2;}
        </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA LOADING & VALIDATION
# ============================================================================

def load_and_validate_data(uploaded_file) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Load and validate with caching and profiling"""
    logger = get_logger()
    
    try:
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name
        
        # Generate file hash for proper cache key
        file_hash = hashlib.md5(file_bytes).hexdigest()[:8]
        
        df_processed, warnings = cached_csv_processing(file_bytes, file_name, file_hash)
        
        if df_processed is None:
            return None, warnings
        
        with st.expander("Raw Data Preview", expanded=False):
            st.dataframe(df_processed.head(10))
            st.caption(f"Showing 10 of {len(df_processed)} rows")
        
        if warnings:
            with st.expander(
                f"Processing Warnings ({len(warnings)})",
                expanded=len(warnings) <= StreamlitConstants.Display.MAX_WARNINGS_SHOW
            ):
                for warning in warnings[:StreamlitConstants.Display.MAX_WARNINGS_SHOW]:
                    st.warning(warning)
                
                if len(warnings) > StreamlitConstants.Display.MAX_WARNINGS_SHOW:
                    st.info(
                        f"... and {len(warnings) - StreamlitConstants.Display.MAX_WARNINGS_SHOW} more warnings"
                    )
        
        return df_processed, warnings
        
    except Exception as e:
        error_msg = f"Error loading data: {str(e)}"
        logger.log_exception(e, "load_and_validate_data")
        st.error(error_msg)
        
        if st.session_state.get('show_debug', False):
            st.code(traceback.format_exc())
        
        return None, [error_msg]

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

def render_sidebar():
    """Render sidebar with configuration options"""
    
    with st.sidebar:
        st.title("Configuration")
        
        # File Upload
        st.header("1. Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Player CSV",
            type=['csv'],
            help="Upload a CSV with player projections"
        )
        
        if uploaded_file is not None:
            if st.button("Load & Validate Data", type="primary"):
                with st.spinner("Loading and validating data..."):
                    df, warnings = load_and_validate_data(uploaded_file)
                    
                    if df is not None:
                        st.session_state.df = df
                        st.session_state.processed_df = df
                        st.session_state.warnings = warnings
                        st.session_state.data_validated = True
                        st.success(f"Loaded {len(df)} players")
                    else:
                        st.session_state.data_validated = False
                        st.error("Data validation failed")
        
        st.divider()
        
        # Game Settings
        st.header("2. Game Settings")
        
        game_total = st.number_input(
            "Game Total (O/U)",
            min_value=30.0,
            max_value=70.0,
            value=float(st.session_state.game_total),
            step=0.5,
            help="Expected total points in the game"
        )
        st.session_state.game_total = game_total
        
        spread = st.number_input(
            "Spread",
            min_value=-20.0,
            max_value=20.0,
            value=float(st.session_state.spread),
            step=0.5,
            help="Point spread (negative = favorite)"
        )
        st.session_state.spread = spread
        
        st.divider()
        
        # Contest Settings
        st.header("3. Contest Settings")
        
        contest_type = st.selectbox(
            "Contest Type",
            options=list(CONTEST_TYPE_MAPPING.keys()),
            index=list(CONTEST_TYPE_MAPPING.keys()).index(st.session_state.contest_type),
            help="Select your contest size"
        )
        st.session_state.contest_type = contest_type
        
        num_lineups = st.slider(
            "Number of Lineups",
            min_value=1,
            max_value=150,
            value=int(st.session_state.num_lineups),
            help="How many lineups to generate"
        )
        st.session_state.num_lineups = num_lineups
        
        optimization_mode: OptimizationModeType = st.selectbox(
            "Optimization Mode",
            options=['balanced', 'ceiling', 'floor', 'boom_or_bust'],
            index=['balanced', 'ceiling', 'floor', 'boom_or_bust'].index(
                st.session_state.optimization_mode
            ),
            help="Strategy for lineup optimization"
        )
        st.session_state.optimization_mode = optimization_mode
        
        st.divider()
        
        # AI Settings
        st.header("4. AI Settings (Optional)")
        
        use_ai = st.checkbox(
            "Enable AI Analysis",
            value=st.session_state.use_ai,
            help="Use Claude AI for strategic recommendations"
        )
        st.session_state.use_ai = use_ai
        
        if use_ai:
            api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                value=st.session_state.api_key,
                help="Your Anthropic API key (starts with 'sk-ant-')"
            )
            st.session_state.api_key = api_key
            
            ai_enforcement: AIEnforcementType = st.select_slider(
                "AI Enforcement Level",
                options=['Advisory', 'Moderate', 'Strong', 'Mandatory'],
                value=st.session_state.ai_enforcement,
                help="How strictly to enforce AI recommendations"
            )
            st.session_state.ai_enforcement = ai_enforcement
            
            if not api_key:
                st.warning("API key required for AI features")
        
        st.divider()
        
        # Advanced Options
        if st.checkbox("Advanced Options", value=st.session_state.show_advanced):
            st.session_state.show_advanced = True
            
            st.subheader("Salary Constraints")
            min_salary_pct = st.slider(
                "Minimum Salary % of Cap",
                min_value=50,
                max_value=100,
                value=int(st.session_state.min_salary_pct),
                step=5,
                help="Lower if optimization fails (default: 95%)"
            )
            st.session_state.min_salary_pct = min_salary_pct
            st.caption(
                f"Min salary: ${int(DraftKingsRules.SALARY_CAP * min_salary_pct / 100):,}"
            )
            
            st.subheader("Debug Options")
            
            show_debug = st.checkbox(
                "Show Debug Info",
                value=st.session_state.show_debug,
                help="Display detailed error traces"
            )
            st.session_state.show_debug = show_debug
            
            show_profiling = st.checkbox(
                "Show Performance Profiling",
                value=st.session_state.show_profiling,
                help="Display performance metrics"
            )
            st.session_state.show_profiling = show_profiling
        else:
            st.session_state.show_advanced = False

# ============================================================================
# DATA OVERVIEW SECTION
# ============================================================================

def render_data_overview():
    """Render complete data overview section"""
    
    st.header("Data Overview")
    
    if st.session_state.processed_df is None:
        st.info("Upload a CSV file in the sidebar to get started")
        return
    
    df = st.session_state.processed_df
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Players", len(df))
    
    with col2:
        teams = df['Team'].nunique()
        st.metric("Teams", teams)
    
    with col3:
        avg_salary = df['Salary'].mean()
        st.metric("Avg Salary", f"${avg_salary:,.0f}")
    
    with col4:
        avg_proj = df['Projected_Points'].mean()
        st.metric("Avg Projection", f"{avg_proj:.1f}")
    
    # Position breakdown
    st.subheader("Position Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pos_counts = df['Position'].value_counts()
        st.bar_chart(pos_counts)
    
    with col2:
        team_counts = df['Team'].value_counts()
        st.bar_chart(team_counts)
    
    # Top players
    st.subheader("Top 10 Players by Projection")
    
    top_players = df.nlargest(10, 'Projected_Points')[
        ['Player', 'Position', 'Team', 'Salary', 'Projected_Points', 'Ownership']
    ].copy()
    
    top_players['Salary'] = top_players['Salary'].apply(lambda x: f"${x:,.0f}")
    top_players['Projected_Points'] = top_players['Projected_Points'].apply(lambda x: f"{x:.1f}")
    top_players['Ownership'] = top_players['Ownership'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(top_players, use_container_width=True, hide_index=True)
    
    # DIAGNOSTIC SECTION WITH QUICK FIX
    with st.expander("Diagnostic Info (Expand if optimization fails)", expanded=False):
        st.subheader("Data Validation")
        
        # Check required columns
        required_cols = ['Player', 'Position', 'Team', 'Salary', 'Projected_Points']
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Required Columns:**")
            for col in required_cols:
                if col in df.columns:
                    st.success(f"✓ {col}")
                else:
                    st.error(f"✗ {col} MISSING")
        
        with col2:
            st.write("**Data Sample:**")
            st.write(f"First player: {df.iloc[0]['Player']}")
            st.write(f"Position: {df.iloc[0]['Position']}")
            st.write(f"Team: {df.iloc[0]['Team']}")
        
        # Check value ranges
        st.write("**Salary Analysis:**")
        st.write(f"- Range: ${df['Salary'].min():,.0f} - ${df['Salary'].max():,.0f}")
        st.write(f"- Average: ${df['Salary'].mean():,.0f}")
        st.write(f"- Median: ${df['Salary'].median():,.0f}")
        
        st.write("**Projection Analysis:**")
        st.write(f"- Range: {df['Projected_Points'].min():.1f} - {df['Projected_Points'].max():.1f}")
        st.write(f"- Average: {df['Projected_Points'].mean():.1f}")
        
        st.write("**Team Distribution:**")
        for team, count in df['Team'].value_counts().items():
            st.write(f"- {team}: {count} players")
        
        if df['Team'].nunique() < 2:
            st.error("WARNING: Only 1 team found! Need at least 2 teams for valid lineups")
        
        # Check if any lineup is theoretically possible
        st.write("**Salary Cap Feasibility:**")
        min_6_salary = df.nsmallest(6, 'Salary')['Salary'].sum()
        max_6_salary = df.nlargest(6, 'Salary')['Salary'].sum()
        current_threshold = int(DraftKingsRules.SALARY_CAP * (st.session_state.min_salary_pct / 100))
        
        st.write(f"- Cheapest 6-player lineup: ${min_6_salary:,.0f}")
        st.write(f"- Most expensive 6-player lineup: ${max_6_salary:,.0f}")
        st.write(f"- Salary cap: ${DraftKingsRules.SALARY_CAP:,.0f}")
        st.write(f"- Current min threshold ({st.session_state.min_salary_pct}%): ${current_threshold:,.0f}")
        
        if min_6_salary > DraftKingsRules.SALARY_CAP:
            st.error("IMPOSSIBLE: Cheapest 6 players exceed salary cap!")
        elif max_6_salary < current_threshold:
            st.error(
                f"IMPOSSIBLE: Most expensive 6 players (${max_6_salary:,.0f}) "
                f"can't reach {st.session_state.min_salary_pct}% minimum (${current_threshold:,.0f})"
            )
            suggested_pct = max(50, int((max_6_salary / DraftKingsRules.SALARY_CAP) * 100) - 5)
            
            # QUICK FIX BUTTON
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"Recommended: Set minimum to {suggested_pct}% or less")
            with col2:
                if st.button("Apply Fix", key="fix_min_salary"):
                    st.session_state.min_salary_pct = suggested_pct
                    st.success(f"Set to {suggested_pct}%")
                    st.rerun()
                    
        elif min_6_salary > current_threshold:
            st.warning("May be difficult: Even cheapest lineup exceeds minimum threshold")
            
            suggested_pct = max(50, int((min_6_salary / DraftKingsRules.SALARY_CAP) * 100) - 5)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"Try lowering to {suggested_pct}%")
            with col2:
                if st.button("Apply Fix", key="fix_min_salary_2"):
                    st.session_state.min_salary_pct = suggested_pct
                    st.success(f"Set to {suggested_pct}%")
                    st.rerun()
        else:
            st.success(f"Salary constraints are feasible (${min_6_salary:,.0f} - ${max_6_salary:,.0f})")
        
        # Sample valid lineup attempt
        st.write("**Sample Lineup Test:**")
        try:
            valid_found = False
            sample_count = 0
            max_samples = min(1000, len(list(combinations(range(len(df)), 6))))
            
            for combo_indices in combinations(range(len(df)), 6):
                if sample_count >= max_samples:
                    break
                sample_count += 1
                
                combo_players = df.iloc[list(combo_indices)]
                total_sal = combo_players['Salary'].sum()
                teams_in_combo = combo_players['Team'].nunique()
                max_from_team = combo_players['Team'].value_counts().max()
                
                if (current_threshold <= total_sal <= DraftKingsRules.SALARY_CAP 
                    and teams_in_combo >= DraftKingsRules.MIN_TEAMS_REQUIRED
                    and max_from_team <= DraftKingsRules.MAX_PLAYERS_PER_TEAM):
                    st.success(f"Found valid combination: ${total_sal:,.0f}, {teams_in_combo} teams")
                    st.write("Players:", ', '.join(combo_players['Player'].tolist()))
                    valid_found = True
                    break
            
            if not valid_found:
                st.error(f"No valid combinations found after checking {sample_count} possibilities")
                st.write("**Common issues:**")
                st.write("- Salaries too low (can't reach minimum threshold)")
                st.write("- All players from same team")
                st.write("- Only 1 team in player pool")
        except Exception as e:
            st.error(f"Error testing combinations: {e}")

# ============================================================================
# OPTIMIZATION SECTION
# ============================================================================

def render_optimization_section():
    """Render optimization section"""
    
    st.header("Optimization")
    
    if st.session_state.processed_df is None:
        st.warning("Please load and validate data first")
        return
    
    # Show current settings
    with st.expander("Current Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Game Settings:**")
            st.write(f"- Total: {st.session_state.game_total}")
            st.write(f"- Spread: {st.session_state.spread}")
            st.write(f"- Contest: {st.session_state.contest_type}")
        
        with col2:
            st.write("**Optimization Settings:**")
            st.write(f"- Lineups: {st.session_state.num_lineups}")
            st.write(f"- Mode: {st.session_state.optimization_mode}")
            st.write(f"- Min Salary: {st.session_state.min_salary_pct}%")
            st.write(f"- AI Enabled: {st.session_state.use_ai}")
    
    # Optimization button
    if st.button("Generate Lineups", type="primary", use_container_width=True):
        run_optimization()


def run_optimization():
    """
    CRITICAL: Thread-safe optimization with defensive DataFrame copy
    FIXED: Simplified error recovery without context manager
    """
    snapshot = StreamlitStateSnapshot()
    snapshot.capture()  # Capture state at the beginning
    
    logger = get_logger()
    profiler = get_profiler()
    
    try:
        # Validate prerequisites
        if st.session_state.use_ai and not st.session_state.api_key:
            st.error("AI enabled but no API key provided")
            return
        
        # CRITICAL: Capture all session state BEFORE threading
        df = st.session_state.processed_df
        num_lineups = st.session_state.num_lineups
        game_total = st.session_state.game_total
        spread = st.session_state.spread
        contest_type = st.session_state.contest_type
        api_key = st.session_state.api_key if st.session_state.use_ai else None
        use_ai = st.session_state.use_ai
        optimization_mode = st.session_state.optimization_mode
        ai_enforcement = st.session_state.ai_enforcement
        min_salary_pct = st.session_state.min_salary_pct
        
        # Pre-flight validation
        min_salary_threshold = int(DraftKingsRules.SALARY_CAP * (min_salary_pct / 100))
        max_6_salary = df.nlargest(6, 'Salary')['Salary'].sum()
        
        if max_6_salary < min_salary_threshold:
            st.error(
                f"IMPOSSIBLE: Most expensive 6 players (${max_6_salary:,.0f}) "
                f"cannot reach minimum salary (${min_salary_threshold:,.0f})"
            )
            suggested_pct = int((max_6_salary / DraftKingsRules.SALARY_CAP) * 100)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(
                    f"Lower 'Minimum Salary % of Cap' to {suggested_pct}% or less in Advanced Options"
                )
            with col2:
                if st.button("Quick Fix"):
                    st.session_state.min_salary_pct = max(50, suggested_pct - 5)
                    st.rerun()
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Setup
        status_text.text("Initializing optimizer...")
        progress_bar.progress(10)
        
        processor = OptimizedDataProcessor()
        game_info = processor.infer_game_info(df, game_total, spread)
        st.session_state.game_info = game_info
        
        # Optimization with timeout
        status_text.text(
            f"Running optimization (timeout: {StreamlitConstants.Timeouts.OPTIMIZATION_SEC}s)..."
        )
        progress_bar.progress(30)
        
        start_time = time.time()
        result = {'lineups': None, 'df': None, 'error': None}
        
        def optimization_thread():
            """Thread function with defensive DataFrame copy"""
            try:
                # CRITICAL: Make defensive deep copy
                df_copy = df.copy(deep=True)
                
                with profiler.profile('full_optimization'):
                    lineups, processed_df = optimize_showdown(
                        csv_path_or_df=df_copy,
                        num_lineups=num_lineups,
                        game_total=game_total,
                        spread=spread,
                        contest_type=contest_type,
                        api_key=api_key,
                        use_ai=use_ai,
                        optimization_mode=optimization_mode,
                        ai_enforcement=ai_enforcement
                    )
                
                result['lineups'] = lineups
                result['df'] = processed_df
                
            except Exception as e:
                result['error'] = e
        
        # Start thread
        thread = threading.Thread(target=optimization_thread)
        thread.daemon = True
        thread.start()
        
        # Wait with progress
        timeout_sec = StreamlitConstants.Timeouts.OPTIMIZATION_SEC
        elapsed = 0
        
        while thread.is_alive() and elapsed < timeout_sec:
            thread.join(timeout=StreamlitConstants.Timeouts.THREAD_JOIN_SEC)
            elapsed = time.time() - start_time
            
            # Update progress (30% to 90%)
            progress_pct = min(90, 30 + int((elapsed / timeout_sec) * 60))
            progress_bar.progress(progress_pct)
            
            # Status updates
            if (elapsed > StreamlitConstants.INITIAL_PROGRESS_THRESHOLD_SEC and 
                int(elapsed) % StreamlitConstants.PROGRESS_UPDATE_INTERVAL_SEC == 0):
                status_text.text(f"Still optimizing... ({int(elapsed)}s elapsed)")
        
        # Check timeout
        if thread.is_alive():
            progress_bar.empty()
            status_text.empty()
            st.error(f"Optimization timed out after {timeout_sec}s")
            st.info("Try these fixes:")
            st.write("1. Lower 'Minimum Salary % of Cap' to 80% or less")
            st.write("2. Reduce number of lineups to 5-10")
            st.write("3. Check Diagnostic Info for feasibility")
            return
        
        # Check errors
        if result['error']:
            progress_bar.empty()
            status_text.empty()
            raise result['error']
        
        lineups = result['lineups']
        elapsed_time = time.time() - start_time
        
        # Complete
        status_text.text("Optimization complete!")
        progress_bar.progress(100)
        time.sleep(0.5)
        
        progress_bar.empty()
        status_text.empty()
        
        # Validate results
        if not lineups:
            st.warning(
                "No lineups generated. Try lowering minimum salary percentage."
            )
            return
        
        # Store results
        st.session_state.lineups = lineups
        st.session_state.optimization_complete = True
        st.session_state.last_optimization_time = elapsed_time
        
        # Success
        st.success(
            f"Successfully generated {len(lineups)} lineups in {elapsed_time:.1f}s"
        )
        
        # Summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_proj = np.mean([l.get('Projected', 0) for l in lineups])
            st.metric("Avg Projection", f"{avg_proj:.2f}")
        
        with col2:
            avg_salary = np.mean([l.get('Total_Salary', 0) for l in lineups])
            st.metric("Avg Salary", f"${avg_salary:,.0f}")
        
        with col3:
            if 'Ceiling_90th' in lineups[0]:
                avg_ceiling = np.mean([l.get('Ceiling_90th', 0) for l in lineups])
                st.metric("Avg Ceiling", f"{avg_ceiling:.2f}")
            else:
                st.metric("Lineups", len(lineups))
        
        # Show profiling if enabled
        if st.session_state.show_profiling:
            with st.expander("Performance Profile", expanded=False):
                st.text(profiler.get_report())
        
        # Rerun to show results
        st.rerun()
        
    except Exception as e:
        snapshot.restore()  # Restore state on error
        logger.log_exception(e, "run_optimization", critical=True)
        st.error(f"Error: {str(e)}")
        
        # Get suggestions
        error_summary = logger.get_error_summary()
        if error_summary.get('recent_errors'):
            recent = error_summary['recent_errors'][-1]
            if recent.get('suggestions'):
                st.info("Suggestions:")
                for suggestion in recent['suggestions'][:3]:
                    st.write(f"- {suggestion}")
        
        if st.session_state.show_debug:
            with st.expander("Debug Info"):
                st.code(traceback.format_exc())
        
        st.info("Your settings have been preserved - adjust and try again")

# ============================================================================
# RESULTS SECTION
# ============================================================================

def render_results_section():
    """Render complete optimization results"""
    
    if not st.session_state.optimization_complete or not st.session_state.lineups:
        return
    
    st.header("Results")
    
    lineups = st.session_state.lineups
    df = st.session_state.processed_df
    
    # Results summary
    st.subheader("Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    projections = [l.get('Projected', 0) for l in lineups]
    salaries = [l.get('Total_Salary', 0) for l in lineups]
    
    with col1:
        st.metric("Total Lineups", len(lineups))
    
    with col2:
        st.metric("Best Projection", f"{max(projections):.2f}")
    
    with col3:
        st.metric("Avg Projection", f"{np.mean(projections):.2f}")
    
    with col4:
        st.metric("Avg Salary", f"${np.mean(salaries):,.0f}")
    
    # Lineup display options
    st.subheader("Lineup Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_count = st.selectbox(
            "Show lineups:",
            options=[5, 10, 20, 50, "All"],
            index=1
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort by:",
            options=['Projected', 'Ceiling_90th', 'Sharpe_Ratio', 'Total_Salary'],
            index=0
        )
    
    with col3:
        show_details = st.checkbox("Show simulation details", value=False)
    
    # Determine how many to show
    if show_count == "All":
        display_lineups = lineups
    else:
        display_lineups = lineups[:int(show_count)]
    
    # Sort lineups
    if sort_by in display_lineups[0]:
        display_lineups = sorted(
            display_lineups,
            key=lambda x: x.get(sort_by, 0),
            reverse=True
        )
    
    # Display lineups
    for i, lineup in enumerate(display_lineups, 1):
        with st.container():
            st.markdown(f"### Lineup {i}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Players
                st.write(f"**Captain:** {lineup.get('Captain', 'N/A')}")
                
                flex = lineup.get('FLEX', [])
                if isinstance(flex, str):
                    flex = [p.strip() for p in flex.split(',') if p.strip()]
                
                st.write("**FLEX:**")
                for player in flex:
                    # Get player info
                    player_info = df[df['Player'] == player]
                    if not player_info.empty:
                        pos = player_info.iloc[0]['Position']
                        salary = player_info.iloc[0]['Salary']
                        proj = player_info.iloc[0]['Projected_Points']
                        st.write(f"  - {player} ({pos}) - ${salary:,.0f} | {proj:.1f} pts")
                    else:
                        st.write(f"  - {player}")
            
            with col2:
                # Metrics
                st.metric("Projected", f"{lineup.get('Projected', 0):.2f}")
                st.metric("Salary", f"${lineup.get('Total_Salary', 0):,.0f}")
                st.metric("Ownership", f"{lineup.get('Total_Ownership', 0):.1f}%")
                
                if show_details and 'Ceiling_90th' in lineup:
                    st.metric("Ceiling (90th)", f"{lineup.get('Ceiling_90th', 0):.2f}")
                    st.metric("Floor (10th)", f"{lineup.get('Floor_10th', 0):.2f}")
                    st.metric("Sharpe Ratio", f"{lineup.get('Sharpe_Ratio', 0):.2f}")
                    st.metric("Win Prob", f"{lineup.get('Win_Probability', 0):.1%}")
            
            st.divider()

# ============================================================================
# EXPORT SECTION
# ============================================================================

def render_export_section():
    """Render complete export section"""
    
    if not st.session_state.optimization_complete or not st.session_state.lineups:
        return
    
    st.header("Export")
    
    lineups = st.session_state.lineups
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox(
            "Export Format",
            options=['Standard', 'DraftKings', 'Detailed'],
            help="Choose export format"
        )
    
    # Convert to enum
    format_map = {
        'Standard': ExportFormat.STANDARD,
        'DraftKings': ExportFormat.DRAFTKINGS,
        'Detailed': ExportFormat.DETAILED
    }
    selected_format = format_map[export_format]
    
    # Validate format
    is_valid, error_msg = validate_export_format(lineups, selected_format)
    
    if not is_valid:
        st.error(f"{error_msg}")
        return
    
    # Generate export
    try:
        export_df = format_lineup_for_export(lineups, selected_format)
        
        # Preview
        with st.expander("Export Preview", expanded=False):
            st.dataframe(export_df.head(10), use_container_width=True)
        
        # Download button
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label=f"Download {export_format} CSV",
            data=csv,
            file_name=f"optimized_lineups_{export_format.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.success(f"Ready to export {len(export_df)} lineups in {export_format} format")
        
    except Exception as e:
        st.error(f"Export failed: {str(e)}")
        
        if st.session_state.show_debug:
            st.code(traceback.format_exc())

# ============================================================================
# ANALYTICS SECTION
# ============================================================================

def render_analytics_section():
    """Render complete analytics section"""
    
    if not st.session_state.optimization_complete or not st.session_state.lineups:
        return
    
    st.header("Analytics")
    
    lineups = st.session_state.lineups
    df = st.session_state.processed_df
    
    # Player exposure
    st.subheader("Player Exposure")
    
    exposure = {}
    for lineup in lineups:
        captain = lineup.get('Captain', '')
        flex = lineup.get('FLEX', [])
        
        if isinstance(flex, str):
            flex = [p.strip() for p in flex.split(',') if p.strip()]
        
        all_players = [captain] + flex
        
        for player in all_players:
            exposure[player] = exposure.get(player, 0) + 1
    
    # Convert to percentage
    exposure_pct = {
        player: (count / len(lineups)) * 100
        for player, count in exposure.items()
    }
    
    # Sort and display top 15
    top_exposure = sorted(exposure_pct.items(), key=lambda x: x[1], reverse=True)[:15]
    
    exposure_df = pd.DataFrame(top_exposure, columns=['Player', 'Exposure %'])
    
    st.bar_chart(exposure_df.set_index('Player'))
    
    # Captain usage
    st.subheader("Captain Usage")
    
    captain_counts = {}
    for lineup in lineups:
        captain = lineup.get('Captain', '')
        captain_counts[captain] = captain_counts.get(captain, 0) + 1
    
    captain_pct = {
        captain: (count / len(lineups)) * 100
        for captain, count in captain_counts.items()
    }
    
    captain_df = pd.DataFrame(
        list(captain_pct.items()),
        columns=['Captain', 'Usage %']
    ).sort_values('Usage %', ascending=False)
    
    st.bar_chart(captain_df.set_index('Captain'))
    
    # Salary distribution
    st.subheader("Salary Distribution")
    
    salaries = [l.get('Total_Salary', 0) for l in lineups]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Min Salary", f"${min(salaries):,.0f}")
    
    with col2:
        st.metric("Max Salary", f"${max(salaries):,.0f}")
    
    with col3:
        st.metric("Avg Salary", f"${np.mean(salaries):,.0f}")
    
    # Histogram with graceful fallback
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.hist(salaries, bins=20, edgecolor='black')
        ax.set_xlabel('Total Salary')
        ax.set_ylabel('Frequency')
        ax.set_title('Salary Distribution')
        st.pyplot(fig)
    except ImportError:
        # Fallback to Streamlit native charting
        salary_bins = pd.cut(salaries, bins=20)
        salary_counts = salary_bins.value_counts().sort_index()
        st.bar_chart(salary_counts)
    
    # Team stacking
    st.subheader("Team Stacking Analysis")
    
    team_counts_dist = []
    for lineup in lineups:
        captain = lineup.get('Captain', '')
        flex = lineup.get('FLEX', [])
        
        if isinstance(flex, str):
            flex = [p.strip() for p in flex.split(',') if p.strip()]
        
        all_players = [captain] + flex
        
        teams = df[df['Player'].isin(all_players)]['Team'].value_counts()
        team_counts_dist.append(teams.max())
    
    stack_df = pd.DataFrame({
        'Max Players from Same Team': team_counts_dist
    })
    
    st.bar_chart(stack_df['Max Players from Same Team'].value_counts())

# ============================================================================
# FOOTER
# ============================================================================

def render_footer():
    """Render footer"""
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("NFL DFS Optimizer v3.2.1")
    
    with col2:
        st.caption("Built with Streamlit & Claude AI")
    
    with col3:
        if st.session_state.optimization_complete:
            elapsed = st.session_state.last_optimization_time
            if elapsed:
                st.caption(f"Last optimization: {elapsed:.1f}s")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point"""
    
    # Ensure state is initialized
    ensure_session_state()
    
    # Apply styling
    apply_custom_css()
    
    # Header
    st.title("NFL DFS AI-Driven Optimizer")
    st.markdown("*Optimize your DraftKings Showdown lineups with advanced AI and Monte Carlo simulation*")
    
    # Sidebar
    render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Overview",
        "Optimization",
        "Results",
        "Export & Analytics"
    ])
    
    with tab1:
        render_data_overview()
    
    with tab2:
        render_optimization_section()
    
    with tab3:
        render_results_section()
    
    with tab4:
        if st.session_state.optimization_complete:
            render_export_section()
            st.divider()
            render_analytics_section()
        else:
            st.info("Complete optimization to view export and analytics")
    
    # Footer
    render_footer()

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
