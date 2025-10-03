
"""
NFL Showdown Optimizer - Streamlit Interface
Enhanced with Dynamic Enforcement Integration & Robust Data Validation
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import io
import traceback
import numpy as np

# Import from your optimizer file
from Optimizer import (
    ShowdownOptimizer,
    AIEnforcementLevel,
    OptimizerConfig,
    get_logger,
    FieldSize
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="NFL Showdown Optimizer - AI Powered",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'optimizer' not in st.session_state:
    st.session_state.optimizer = None
if 'lineups' not in st.session_state:
    st.session_state.lineups = None
if 'optimization_complete' not in st.session_state:
    st.session_state.optimization_complete = False
if 'optimization_metadata' not in st.session_state:
    st.session_state.optimization_metadata = None
if 'show_advanced' not in st.session_state:
    st.session_state.show_advanced = False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _find_col_index(columns, search_terms):
    """Find best matching column index"""
    for i, col in enumerate(columns):
        if col and any(term.lower() in str(col).lower() for term in search_terms):
            return i
    return 0

def transform_salary_values(df, salary_col_name='Salary'):
    """
    Intelligently transform salary values to DraftKings format ($200-$12,000)

    Handles:
    - Thousands format (5.5 -> 5500)
    - Already correct format (5500 -> 5500)
    - Cents format (550000 -> 5500)
    """
    df = df.copy()

    salaries = df[salary_col_name]
    min_sal = salaries.min()
    max_sal = salaries.max()

    # Log original range
    st.info(f"📊 Original salary range: ${min_sal:,.2f} - ${max_sal:,.2f}")

    # Case 1: Values are in thousands (like 5.5, 7.2, 10.0)
    if max_sal <= 20:
        df[salary_col_name] = (salaries * 1000).astype(int)
        st.success(f"✅ Converted salaries from thousands format (x1000)")

    # Case 2: Values are in cents or too high (like 550000)
    elif max_sal > 50000:
        df[salary_col_name] = (salaries / 100).astype(int)
        st.warning(f"⚠️ Converted salaries by dividing by 100")

    # Case 3: Values look correct (between 200-15000)
    elif min_sal >= 200 and max_sal <= 15000:
        df[salary_col_name] = salaries.astype(int)
        st.success(f"✅ Salary format looks correct")

    # Case 4: Unclear format - try to infer
    else:
        st.error(f"❌ Unable to determine salary format. Min=${min_sal}, Max=${max_sal}")
        st.error("Expected DraftKings range: $200-$12,000")
        raise ValueError(
            f"Salary values unclear. Found: ${min_sal:,.0f} - ${max_sal:,.0f}. "
            f"Expected: $200-$12,000"
        )

    # Validate final range
    final_min = df[salary_col_name].min()
    final_max = df[salary_col_name].max()

    st.info(f"📊 Final salary range: ${final_min:,} - ${final_max:,}")

    if final_min < 200 or final_max > 15000:
        st.error(
            f"⚠️ WARNING: Salaries outside typical DK range. "
            f"Min=${final_min:,}, Max=${final_max:,}"
        )

    return df

def validate_player_pool(df):
    """
    Comprehensive player pool validation
    Returns: (critical_issues, warnings, info_messages)
    """
    critical = []
    warnings = []
    info = []

    player_count = len(df)
    team_count = df['Team'].nunique()

    # Critical issues (will block optimization)
    if player_count < 6:
        critical.append(f"Only {player_count} players - need at least 6")
    elif player_count < 12:
        critical.append(f"Only {player_count} players - optimization will use Advisory enforcement")

    if team_count < 2:
        critical.append("Only 1 team - DraftKings requires players from 2+ teams")

    # Salary validation
    min_possible = df.nsmallest(6, 'Salary')['Salary'].sum()
    max_possible = df.nlargest(6, 'Salary')['Salary'].sum()

    if min_possible > 50000:
        critical.append(f"Cheapest possible lineup (${min_possible:,}) exceeds salary cap")

    if (df['Salary'] > 15000).any():
        warnings.append("Salary values over $15,000 detected - verify data accuracy")
    if (df['Salary'] < 200).any():
        warnings.append("Salary values under $200 detected - verify data accuracy")

    # Warnings (optimization will proceed with adjustments)
    if player_count < 18:
        warnings.append(f"Small pool ({player_count} players) - enforcement will auto-adjust to Advisory")
    elif player_count < 25:
        warnings.append(f"Limited pool ({player_count} players) - enforcement capped at Moderate")

    team_counts = df['Team'].value_counts()
    for team, count in team_counts.items():
        if count < 3:
            warnings.append(f"Only {count} players from {team} - limits lineup diversity")

    # Informational
    info.append(f"Player pool: {player_count} players from {team_count} teams")
    info.append(f"Salary range: ${df['Salary'].min():,} - ${df['Salary'].max():,}")
    info.append(f"Projection range: {df['Projected_Points'].min():.1f} - {df['Projected_Points'].max():.1f} pts")

    return critical, warnings, info

def get_enforcement_recommendation(player_count):
    """Get recommended enforcement level based on pool size"""
    if player_count < 18:
        return "Advisory", "⚠️ Small pool"
    elif player_count < 25:
        return "Moderate", "✓ Limited pool"
    elif player_count < 35:
        return "Strong", "✓✓ Good pool size"
    else:
        return "Strong or Mandatory", "✓✓✓ Large pool"

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .stAlert > div {
        padding: 10px;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .enforcement-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.9em;
    }
    .enforcement-adjusted {
        background-color: #ffd700;
        color: #000;
    }
    .enforcement-normal {
        background-color: #90EE90;
        color: #000;
    }
    .data-issue {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.title("🏈 NFL Showdown Optimizer")
st.markdown("### AI-Powered DFS Lineup Generator with Dynamic Enforcement")

# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    # API Key Section
    st.subheader("🔑 Claude API Key")

    try:
        default_api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if default_api_key:
            st.success("✓ API key loaded from secrets")
            api_key = default_api_key
            use_secrets = True
        else:
            use_secrets = False
    except:
        use_secrets = False
        default_api_key = ""

    if not use_secrets:
        api_key = st.text_input(
            "Enter Claude API key",
            type="password",
            help="Get your API key from console.anthropic.com"
        )

    if api_key:
        st.success("✓ AI analysis enabled")
        with st.expander("Active AI Strategists"):
            st.markdown("- **Game Theory AI**: Ownership leverage")
            st.markdown("- **Correlation AI**: Stacking analysis")
            st.markdown("- **Contrarian AI**: Unique narratives")
    else:
        st.warning("No API key - using statistical mode")

    st.divider()

    # File Upload
    st.subheader("📁 Player Data")
    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=['csv'],
        help="CSV with player projections"
    )

    if uploaded_file:
        st.divider()

        # Game Information
        st.subheader("🏟️ Game Context")
        teams = st.text_input("Matchup", "Team1 vs Team2")

        col1, col2 = st.columns(2)
        with col1:
            game_total = st.number_input("Total", value=47.5, step=0.5)
        with col2:
            spread = st.number_input("Spread", value=-3.5, step=0.5)

        weather = st.selectbox(
            "Weather",
            ["Clear", "Dome", "Rain", "Snow", "Wind"]
        )

        st.divider()

        # Optimization Settings
        st.subheader("🎯 Settings")

        num_lineups = st.slider(
            "Number of Lineups",
            min_value=5,
            max_value=150,
            value=20,
            step=5
        )

        field_size = st.selectbox(
            "Field Size",
            options=[
                'small_field',
                'medium_field',
                'large_field',
                'large_field_aggressive',
                'milly_maker'
            ],
            index=2,
            format_func=lambda x: {
                'small_field': '3-Max / 5-Max',
                'medium_field': '20-Max',
                'large_field': 'Large GPP',
                'large_field_aggressive': 'Very Large GPP',
                'milly_maker': 'Milly Maker'
            }[x]
        )

        # AI Enforcement with recommendation
        st.markdown("**AI Enforcement**")

        # Show recommendation in expander
        with st.expander("ℹ️ Enforcement Guide"):
            st.markdown("""
            **Advisory**: AI recommendations as preferences only
            - Best for: < 18 players
            - AI influence: ~25%

            **Moderate**: High-confidence AI rules enforced
            - Best for: 18-25 players
            - AI influence: ~55%

            **Strong**: Most AI recommendations enforced
            - Best for: 25-35 players
            - AI influence: ~80%

            **Mandatory**: All AI rules strictly enforced
            - Best for: 35+ players
            - AI influence: ~95%

            *Note: System will auto-adjust if pool size is insufficient*
            """)

        ai_enforcement = st.select_slider(
            "Level",
            options=['Advisory', 'Moderate', 'Strong', 'Mandatory'],
            value='Moderate',
            help="Will auto-adjust based on player pool size"
        )

        st.divider()

        # Advanced Options
        if st.checkbox("⚡ Advanced Options"):
            st.session_state.show_advanced = True
        else:
            st.session_state.show_advanced = False

        if st.session_state.show_advanced:
            col1, col2 = st.columns(2)

            with col1:
                use_genetic = st.checkbox(
                    "Genetic Algorithm",
                    help="Better diversity, slower"
                )
            with col2:
                use_simulation = st.checkbox(
                    "Monte Carlo Sim",
                    value=True,
                    help="Calculate ceiling/floor"
                )

            randomness = st.slider(
                "Projection Variance",
                0.0, 0.30, 0.15, 0.05,
                help="Higher = more lineup diversity"
            )
        else:
            use_genetic = False
            use_simulation = True
            randomness = 0.15

        st.divider()

        # Generate Button
        optimize_button = st.button(
            "🚀 Generate Lineups",
            type="primary",
            use_container_width=True
        )

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

if uploaded_file is None:
    # Welcome Screen
    st.info("👆 Upload a CSV file in the sidebar to begin")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("📋 Required CSV Format"):
            st.markdown("""
            **Required Columns:**
            - Player/Name (or first_name + last_name)
            - Position (QB, RB, WR, TE, DST)
            - Team
            - Salary (DraftKings format)
            - Projected Points

            **Optional:**
            - Ownership % (will default to 10%)

            **Salary Formats Supported:**
            - Dollars: 5500, 7200, 9800
            - Thousands: 5.5, 7.2, 9.8
            - The app will auto-detect and convert
            """)

    with col2:
        with st.expander("🤖 How AI Optimization Works"):
            st.markdown("""
            **Triple-AI Analysis:**

            1. **Game Theory AI** analyzes ownership inefficiencies
            2. **Correlation AI** builds optimal stacks
            3. **Contrarian AI** finds unique winning angles

            The system automatically adjusts constraint enforcement based on your player pool size to ensure feasibility while maximizing AI strategic impact.
            """)

    st.stop()

# ============================================================================
# FILE UPLOADED - SHOW DATA MAPPING
# ============================================================================

try:
    df_raw = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"❌ Failed to read CSV: {str(e)}")
    st.stop()

st.subheader("📊 Data Preview & Mapping")

with st.expander("🔍 Raw Data (first 10 rows)", expanded=False):
    st.dataframe(df_raw.head(10), use_container_width=True)

# Column Mapping
st.markdown("**Map CSV columns to required fields:**")

col1, col2, col3 = st.columns(3)

csv_columns = [''] + list(df_raw.columns)

with col1:
    # Check for split name columns
    has_first_last = ('first_name' in df_raw.columns and 'last_name' in df_raw.columns)

    if has_first_last:
        st.info("📌 Detected first_name + last_name columns")
        player_col = 'first_name'  # Will be combined later
        combine_names = True
    else:
        player_col = st.selectbox(
            "Player Name",
            csv_columns,
            index=_find_col_index(csv_columns, ['player', 'name', 'full_name'])
        )
        combine_names = False

    position_col = st.selectbox(
        "Position",
        csv_columns,
        index=_find_col_index(csv_columns, ['position', 'pos'])
    )

with col2:
    team_col = st.selectbox(
        "Team",
        csv_columns,
        index=_find_col_index(csv_columns, ['team'])
    )

    salary_col = st.selectbox(
        "Salary",
        csv_columns,
        index=_find_col_index(csv_columns, ['salary', 'dk salary', 'draftkings'])
    )

with col3:
    proj_col = st.selectbox(
        "Projected Points",
        csv_columns,
        index=_find_col_index(csv_columns, ['fpts', 'projection', 'points', 'projected', 'point_projection'])
    )

    own_col = st.selectbox(
        "Ownership % (optional)",
        csv_columns,
        index=_find_col_index(csv_columns, ['ownership', 'own', 'projected_ownership'])
    )

# Validate mapping
if combine_names:
    required_mapped = all([position_col, team_col, salary_col, proj_col])
else:
    required_mapped = all([player_col, position_col, team_col, salary_col, proj_col])

if not required_mapped:
    st.warning("⚠️ Please map all required columns")
    st.stop()

# Create mapped dataframe
try:
    if combine_names:
        # Combine first and last name
        df_mapped = pd.DataFrame({
            'Player': df_raw['first_name'].astype(str) + ' ' + df_raw['last_name'].astype(str),
            'Position': df_raw[position_col],
            'Team': df_raw[team_col],
            'Salary': pd.to_numeric(df_raw[salary_col], errors='coerce'),
            'Projected_Points': pd.to_numeric(df_raw[proj_col], errors='coerce'),
            'Ownership': pd.to_numeric(df_raw[own_col], errors='coerce') if own_col else 10.0
        })
        st.success("✅ Combined first_name + last_name → Player")
    else:
        df_mapped = pd.DataFrame({
            'Player': df_raw[player_col],
            'Position': df_raw[position_col],
            'Team': df_raw[team_col],
            'Salary': pd.to_numeric(df_raw[salary_col], errors='coerce'),
            'Projected_Points': pd.to_numeric(df_raw[proj_col], errors='coerce'),
            'Ownership': pd.to_numeric(df_raw[own_col], errors='coerce') if own_col else 10.0
        })

    # Clean data
    df_mapped = df_mapped.dropna(subset=['Player', 'Position', 'Team', 'Salary', 'Projected_Points'])

    if df_mapped.empty:
        st.error("❌ No valid data after cleaning. Check your CSV for missing/invalid values.")
        st.stop()

    # Standardize position and team formatting
    df_mapped['Position'] = df_mapped['Position'].astype(str).str.upper()
    df_mapped['Team'] = df_mapped['Team'].astype(str).str.upper()

    # Fill missing ownership
    df_mapped['Ownership'] = df_mapped['Ownership'].fillna(10.0)

    st.success("✅ Data mapped successfully")

except Exception as e:
    st.error(f"❌ Error mapping columns: {str(e)}")
    st.exception(e)
    st.stop()

# Transform salary values
st.markdown("### 💰 Salary Validation & Transformation")
try:
    df_mapped = transform_salary_values(df_mapped, 'Salary')
except Exception as e:
    st.error(f"❌ Salary transformation failed: {str(e)}")
    with st.expander("🔧 Manual Fix Required"):
        st.markdown("""
        Your salary values couldn't be auto-converted. Please:
        1. Check your CSV salary column
        2. Ensure values are in one of these formats:
           - **Dollars**: 3000, 5500, 8200 (recommended)
           - **Thousands**: 3.0, 5.5, 8.2
        3. Re-upload the corrected CSV
        """)
    st.stop()

df = df_mapped

# Show processed data
st.markdown("### ✅ Processed Data")
st.dataframe(df.head(10), use_container_width=True)

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Players", len(df))
col2.metric("Teams", df['Team'].nunique())
col3.metric("Avg Salary", f"${df['Salary'].mean():,.0f}")
col4.metric("Avg Projection", f"{df['Projected_Points'].mean():.2f}")

st.divider()

# ============================================================================
# VALIDATION & RECOMMENDATIONS
# ============================================================================

critical, warnings, info = validate_player_pool(df)

# Show enforcement recommendation
recommended_level, rec_icon = get_enforcement_recommendation(len(df))
st.info(
    f"{rec_icon} **Recommended Enforcement:** {recommended_level} "
    f"(based on {len(df)} player pool)"
)

# Critical issues - block optimization
if critical:
    st.error("**⛔ Critical Issues - Cannot Optimize:**")
    for issue in critical:
        st.error(f"• {issue}")

    st.markdown("**Solutions:**")
    st.markdown("- Add more players to your CSV")
    st.markdown("- Ensure you have players from at least 2 teams")
    st.markdown("- Verify salary values are in DraftKings format ($200-$12,000)")
    st.stop()

# Warnings - show but allow optimization
if warnings:
    with st.expander("⚠️ Warnings (optimization will auto-adjust)", expanded=True):
        for warning in warnings:
            st.warning(f"• {warning}")

        st.info(
            "The optimizer will automatically adjust AI enforcement if needed. "
            "You'll see a notification if this occurs."
        )

# Info messages
if info:
    with st.expander("ℹ️ Pool Analysis"):
        for msg in info:
            st.caption(msg)

# ============================================================================
# RUN OPTIMIZATION
# ============================================================================

if optimize_button:
    st.divider()
    st.subheader("⚙️ Optimization in Progress")

    # Reset state
    st.session_state.optimization_complete = False
    st.session_state.lineups = None
    st.session_state.optimization_metadata = None

    # Create game info
    game_info = {
        'teams': teams,
        'total': game_total,
        'spread': spread,
        'weather': weather
    }

    # Map enforcement level
    enforcement_map = {
        'Advisory': AIEnforcementLevel.ADVISORY,
        'Moderate': AIEnforcementLevel.MODERATE,
        'Strong': AIEnforcementLevel.STRONG,
        'Mandatory': AIEnforcementLevel.MANDATORY
    }

    try:
        # Initialize optimizer
        with st.spinner("Initializing optimizer..."):
            optimizer = ShowdownOptimizer(api_key=api_key if api_key else None)
            st.session_state.optimizer = optimizer

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(pct, msg):
            progress_bar.progress(pct)
            status_text.text(f"[{int(pct*100)}%] {msg}")

        # Run optimization
        lineups = optimizer.optimize(
            df=df,
            game_info=game_info,
            num_lineups=num_lineups,
            field_size=field_size,
            ai_enforcement_level=enforcement_map[ai_enforcement],
            use_api=(api_key is not None and api_key != ""),
            randomness=randomness,
            use_genetic=use_genetic,
            use_simulation=use_simulation,
            progress_callback=update_progress
        )

        # Store results
        st.session_state.lineups = lineups
        st.session_state.optimization_metadata = optimizer.get_optimization_report()

        # Clear progress
        progress_bar.empty()
        status_text.empty()

        # Check for enforcement adjustment
        metadata = st.session_state.optimization_metadata.get('metadata', {})
        enforcement_adjusted = metadata.get('enforcement_was_adjusted', False)
        actual_enforcement = metadata.get('enforcement_level', ai_enforcement)

        if lineups is not None and not lineups.empty:
            st.session_state.optimization_complete = True

            # Success message with enforcement info
            if enforcement_adjusted:
                st.warning(
                    f"✅ Generated {len(lineups)} lineups | "
                    f"⚠️ Enforcement auto-adjusted: {ai_enforcement} → {actual_enforcement}"
                )
                st.info(
                    "Enforcement was automatically adjusted based on your player pool size "
                    "to ensure lineup generation feasibility while maintaining AI strategic influence."
                )
            else:
                st.success(f"✅ Successfully generated {len(lineups)} lineups!")
        else:
            st.error("❌ No lineups generated")

            # Show diagnostics
            logger = get_logger()
            error_summary = logger.get_error_summary()

            with st.expander("🔍 Diagnostic Information", expanded=True):
                col1, col2, col3 = st.columns(3)

                col1.metric("Total Errors", error_summary['total_errors'])
                col2.metric("Constraint Issues",
                           error_summary['error_categories'].get('constraint', 0))
                col3.metric("Validation Errors",
                           error_summary['error_categories'].get('validation', 0))

                st.markdown("**Suggested Solutions:**")
                st.info("1. Your player pool may be too small for the requested enforcement level")
                st.info("2. Try reducing enforcement to Advisory")
                st.info("3. Reduce the number of lineups requested")
                st.info("4. Add more players to your CSV if possible")

                if error_summary['recent_errors']:
                    st.markdown("**Recent Errors:**")
                    for err in error_summary['recent_errors'][-3:]:
                        st.code(err['message'])

    except Exception as e:
        st.error(f"❌ Optimization failed: {str(e)}")

        with st.expander("🔧 Error Details"):
            st.exception(e)
            st.code(traceback.format_exc())

        st.markdown("**Troubleshooting:**")
        st.markdown("1. Verify your CSV data is valid")
        st.markdown("2. Try with fewer lineups")
        st.markdown("3. Lower enforcement level to Advisory")
        st.markdown("4. Disable AI (remove API key) to test with statistical mode")

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

if st.session_state.optimization_complete and st.session_state.lineups is not None:
    lineups = st.session_state.lineups
    metadata = st.session_state.optimization_metadata

    st.divider()
    st.header("📈 Optimization Results")

    # Show enforcement badge
    if metadata:
        meta_data = metadata.get('metadata', {})
        enforcement_adjusted = meta_data.get('enforcement_was_adjusted', False)
        actual_enforcement = meta_data.get('enforcement_level', 'Unknown')
        original_enforcement = meta_data.get('original_enforcement_level', actual_enforcement)

        if enforcement_adjusted:
            st.markdown(
                f'<div class="enforcement-badge enforcement-adjusted">'
                f'Enforcement: {original_enforcement} → {actual_enforcement} (auto-adjusted)'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="enforcement-badge enforcement-normal">'
                f'Enforcement: {actual_enforcement}'
                f'</div>',
                unsafe_allow_html=True
            )

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Lineups", len(lineups))
    col2.metric("Avg Projection", f"{lineups['Projected'].mean():.2f}")
    col3.metric("Avg Ownership", f"{lineups['Total_Own'].mean():.1f}%")
    col4.metric("Unique Captains", lineups['CPT'].nunique())
    col5.metric("Avg Salary", f"${lineups['Total_Salary'].mean():,.0f}")

    # Simulation metrics
    if 'Sim_Ceiling_90th' in lineups.columns:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Ceiling (90th)", f"{lineups['Sim_Ceiling_90th'].mean():.2f}")
        col2.metric("Avg Sharpe", f"{lineups['Sim_Sharpe'].mean():.2f}")
        col3.metric("Avg Win Prob", f"{lineups['Sim_Win_Prob'].mean():.1%}")
        col4.metric("Top Ceiling", f"{lineups['Sim_Ceiling_90th'].max():.2f}")

    st.divider()

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 All Lineups",
        "🏆 Top Performers",
        "🤖 AI Analysis",
        "📊 Insights",
        "💾 Export"
    ])

    with tab1:
        st.dataframe(
            lineups,
            use_container_width=True,
            height=600
        )

    with tab2:
        st.subheader("Top 10 Lineups by Projection")

        sort_col = st.selectbox(
            "Sort by:",
            ['Projected', 'Sim_Ceiling_90th', 'Sim_Win_Prob'] if 'Sim_Ceiling_90th' in lineups.columns else ['Projected']
        )

        top_lineups = lineups.nlargest(10, sort_col)

        for idx, lineup in top_lineups.iterrows():
            with st.expander(
                f"#{lineup['Lineup']} - {lineup['Projected']:.2f} pts - "
                f"{lineup['Total_Own']:.1f}% own"
            ):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**Captain:** {lineup['CPT']}")
                    st.markdown("**FLEX:**")
                    for i in range(1, 6):
                        if f'FLEX{i}' in lineup:
                            st.markdown(f"- {lineup[f'FLEX{i}']}")

                with col2:
                    st.metric("Projection", f"{lineup['Projected']:.2f}")
                    st.metric("Ownership", f"{lineup['Total_Own']:.1f}%")
                    st.metric("Salary", f"${lineup['Total_Salary']:,}")

                    if 'Sim_Ceiling_90th' in lineup:
                        st.metric("Ceiling", f"{lineup['Sim_Ceiling_90th']:.2f}")
                        st.metric("Win Prob", f"{lineup['Sim_Win_Prob']:.1%}")

    with tab3:
        st.subheader("AI Strategy Analysis")

        if 'Strategy' in lineups.columns:
            strategy_counts = lineups['Strategy'].value_counts()

            col1, col2 = st.columns([1, 2])

            with col1:
                st.dataframe(
                    strategy_counts.reset_index().rename(
                        columns={'Strategy': 'Count', 'index': 'Strategy'}
                    ),
                    hide_index=True
                )

            with col2:
                st.bar_chart(strategy_counts)

            # Show examples from each strategy
            for strategy in strategy_counts.index:
                strategy_lineups = lineups[lineups['Strategy'] == strategy]
                if not strategy_lineups.empty:
                    example = strategy_lineups.iloc[0]

                    with st.expander(f"📌 {strategy} Example"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(f"**Captain:** {example['CPT']}")
                            st.markdown(f"**Projection:** {example['Projected']:.2f}")
                            st.markdown(f"**Ownership:** {example['Total_Own']:.1f}%")

                        with col2:
                            descriptions = {
                                'Game Theory': "Exploits ownership inefficiencies",
                                'Correlation': "Maximizes correlated scoring",
                                'Contrarian Narrative': "Unique tournament angle"
                            }
                            if strategy in descriptions:
                                st.info(descriptions[strategy])
        else:
            st.info("Strategy classification not available")

    with tab4:
        st.subheader("Portfolio Analysis")

        # Captain distribution
        st.markdown("**Captain Usage:**")
        captain_counts = lineups['CPT'].value_counts().head(10)
        st.bar_chart(captain_counts)

        # Ownership distribution
        st.markdown("**Ownership Tiers:**")
        ownership_bins = pd.cut(
            lineups['Total_Own'],
            bins=[0, 60, 80, 100, 200],
            labels=['<60%', '60-80%', '80-100%', '>100%']
        )
        st.bar_chart(ownership_bins.value_counts().sort_index())

        # Salary distribution
        st.markdown("**Salary Distribution:**")
        st.bar_chart(
            pd.cut(
                lineups['Total_Salary'],
                bins=5
            ).value_counts().sort_index()
        )

    with tab5:
        st.subheader("Download Options")

        col1, col2, col3 = st.columns(3)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with col1:
            csv = lineups.to_csv(index=False)
            st.download_button(
                label="📥 Full CSV",
                data=csv,
                file_name=f"showdown_lineups_{timestamp}.csv",
                mime='text/csv',
                use_container_width=True
            )

        with col2:
            dk_cols = ['CPT', 'FLEX1', 'FLEX2', 'FLEX3', 'FLEX4', 'FLEX5']
            if all(col in lineups.columns for col in dk_cols):
                dk_csv = lineups[dk_cols].to_csv(index=False)
                st.download_button(
                    label="📥 DK Upload Format",
                    data=dk_csv,
                    file_name=f"dk_upload_{timestamp}.csv",
                    mime='text/csv',
                    use_container_width=True
                )

        with col3:
            if metadata:
                report_text = f"""NFL SHOWDOWN OPTIMIZATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CONFIGURATION:
- Lineups Requested: {metadata['metadata']['num_lineups_requested']}
- Lineups Generated: {metadata['metadata']['num_lineups_generated']}
- Field Size: {metadata['metadata']['field_size']}
- Enforcement: {metadata['metadata']['original_enforcement_level']} → {metadata['metadata']['enforcement_level']}
- Adjusted: {metadata['metadata']['enforcement_was_adjusted']}
- Method: {metadata['metadata']['optimization_method']}

RESULTS:
- Average Projection: {lineups['Projected'].mean():.2f}
- Average Ownership: {lineups['Total_Own'].mean():.1f}%
- Unique Captains: {lineups['CPT'].nunique()}
- Salary Utilization: {lineups['Total_Salary'].mean() / 50000:.1%}

PERFORMANCE:
{metadata.get('performance', 'N/A')}
"""
                st.download_button(
                    label="📥 Optimization Report",
                    data=report_text,
                    file_name=f"report_{timestamp}.txt",
                    mime='text/plain',
                    use_container_width=True
                )

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption("NFL Showdown Optimizer v2.0 | Dynamic AI Enforcement | Powered by Claude")
