
import streamlit as st
import pandas as pd
from datetime import datetime
import io

# Import from your optimizer file
from Optimizer import (
    ShowdownOptimizer,
    AIEnforcementLevel,
    OptimizerConfig,
    get_logger
)

st.set_page_config(
    page_title="NFL Showdown Optimizer - AI Powered",
    page_icon="🏈",
    layout="wide"
)

# Initialize session state
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = None
if 'lineups' not in st.session_state:
    st.session_state.lineups = None
if 'optimization_complete' not in st.session_state:
    st.session_state.optimization_complete = False

# Helper Functions
def _find_col_index(columns, search_terms):
    """Find best matching column index"""
    for i, col in enumerate(columns):
        if col and any(term.lower() in col.lower() for term in search_terms):
            return i
    return 0

# Title
st.title("🏈 NFL Showdown Optimizer")
st.markdown("### AI-Powered DFS Lineup Generator with Triple-AI Analysis")

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Key - try secrets first, then manual input
    st.subheader("🔑 Claude API Key")

    # Try to get from secrets
    try:
        default_api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if default_api_key:
            st.success("API key loaded from secrets")
            api_key = default_api_key
            use_secrets = True
        else:
            use_secrets = False
    except:
        use_secrets = False
        default_api_key = ""

    # Manual input option
    if not use_secrets:
        api_key = st.text_input(
            "Enter your Anthropic API key",
            type="password",
            help="Get your API key from console.anthropic.com"
        )

    if api_key:
        st.success("API key provided - AI analysis enabled")
        st.info("**3 AI Strategists Active:**\n- Game Theory AI\n- Correlation AI\n- Contrarian Narrative AI")
    else:
        st.warning("No API key - using statistical fallback mode")

    st.divider()

    # File Upload
    st.subheader("📁 Upload Data")
    uploaded_file = st.file_uploader(
        "Upload Player Projections CSV",
        type=['csv'],
        help="CSV with player data"
    )

    if uploaded_file:
        st.divider()

        # Game Info
        st.subheader("🏟️ Game Information")
        teams = st.text_input("Teams", "Team1 vs Team2")

        col1, col2 = st.columns(2)
        with col1:
            game_total = st.number_input("Game Total", value=47.5, step=0.5)
        with col2:
            spread = st.number_input("Spread", value=-3.5, step=0.5)

        weather = st.selectbox("Weather", ["Clear", "Rain", "Snow", "Wind", "Dome"])

        st.divider()

        # Optimization Settings
        st.subheader("🎯 Optimization Settings")

        num_lineups = st.slider("Number of Lineups", 5, 150, 20)

        field_size = st.selectbox(
            "Field Size",
            ['small_field', 'medium_field', 'large_field',
             'large_field_aggressive', 'milly_maker'],
            index=2,
            help="Determines strategy aggressiveness"
        )

        ai_enforcement = st.select_slider(
            "AI Enforcement Level",
            options=['Advisory', 'Moderate', 'Strong', 'Mandatory'],
            value='Strong',
            help="How strictly to follow AI recommendations"
        )

        col1, col2 = st.columns(2)
        with col1:
            use_genetic = st.checkbox(
                "Use Genetic Algorithm",
                help="Slower but finds better lineup diversity"
            )
        with col2:
            use_simulation = st.checkbox(
                "Monte Carlo Simulation",
                value=True,
                help="Calculates ceiling/floor projections"
            )

        randomness = st.slider(
            "Randomness",
            0.0, 0.3, 0.15,
            help="Projection variance for lineup diversity"
        )

        st.divider()

        # Optimize Button
        optimize_button = st.button(
            "🚀 Generate Lineups",
            type="primary",
            use_container_width=True
        )

# Main Content Area
if uploaded_file is None:
    st.info("👆 Upload a CSV file in the sidebar to get started")

    with st.expander("📋 CSV Format Guide"):
        st.markdown("""
        Your CSV needs these columns (names can vary):

        **Required:**
        - **Player/Name**: Player name
        - **Position/Pos**: QB, RB, WR, TE, DST
        - **Team**: Team abbreviation
        - **Salary/DK Salary**: DraftKings salary
        - **Projected Points/FPTS**: Your projections

        **Optional:**
        - **Ownership/Own%**: Projected ownership %

        Example CSV:

Player,Position,Team,Salary,Projected_Points,Ownership
    Patrick Mahomes,QB,KC,11500,25.5,35.2
    Travis Kelce,TE,KC,9000,18.3,28.5

The app will help you map columns if they have different names.
        """)

    with st.expander("🤖 How the AI Analysis Works"):
        st.markdown("""
        **Game Theory AI**:
        - Analyzes ownership patterns
        - Identifies leverage opportunities
        - Finds optimal contrarian captains

        **Correlation AI**:
        - Builds QB-WR/TE stacks
        - Creates bring-back strategies
        - Maximizes correlated scoring

        **Contrarian Narrative AI**:
        - Finds unique winning angles
        - Identifies chalk to fade
        - Builds tournament-winning narratives

        All three work together to create optimal, differentiated lineups.
        """)

else:
    # Load and preview data
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview & Column Mapping")

    # Show raw data
    with st.expander("🔍 View Raw Data"):
        st.dataframe(df.head(10), use_container_width=True)

    # Column mapping interface
    st.caption("Map your CSV columns to required fields")

    col1, col2 = st.columns(2)

    csv_columns = [''] + list(df.columns)

    with col1:
        player_col = st.selectbox("Player Name →", csv_columns,
                                  index=_find_col_index(csv_columns, ['player', 'name']))
        position_col = st.selectbox("Position →", csv_columns,
                                   index=_find_col_index(csv_columns, ['position', 'pos']))
        team_col = st.selectbox("Team →", csv_columns,
                               index=_find_col_index(csv_columns, ['team']))

    with col2:
        salary_col = st.selectbox("Salary →", csv_columns,
                                 index=_find_col_index(csv_columns, ['salary', 'dk salary']))
        proj_col = st.selectbox("Projected Points →", csv_columns,
                               index=_find_col_index(csv_columns, ['fpts', 'projection', 'points', 'projected']))
        own_col = st.selectbox("Ownership % (optional) →", csv_columns,
                              index=_find_col_index(csv_columns, ['ownership', 'own']))

    # Validate mapping
    required_mapped = all([player_col, position_col, team_col, salary_col, proj_col])

    if not required_mapped:
        st.warning("⚠️ Please map all required columns")
        st.stop()

    # Create mapped dataframe
    df_mapped = pd.DataFrame({
        'Player': df[player_col],
        'Position': df[position_col],
        'Team': df[team_col],
        'Salary': pd.to_numeric(df[salary_col], errors='coerce'),
        'Projected_Points': pd.to_numeric(df[proj_col], errors='coerce'),
        'Ownership': pd.to_numeric(df[own_col], errors='coerce') if own_col else 10.0
    })

    # Drop rows with NaN values
    df_mapped = df_mapped.dropna()

    # Update df reference
    df = df_mapped

    st.success("✅ Columns mapped successfully!")

    # Show mapped data
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Players", len(df))
    col2.metric("Avg Salary", f"${df['Salary'].mean():,.0f}")
    col3.metric("Avg Projection", f"{df['Projected_Points'].mean():.2f}")
    col4.metric("Teams", df['Team'].nunique())

    # Run optimization
    if optimize_button:
        st.session_state.optimization_complete = False

        with st.spinner("Initializing optimizer..."):
            # Create game info dict
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
                optimizer = ShowdownOptimizer(api_key=api_key if api_key else None)
                st.session_state.optimizer = optimizer

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(pct, msg):
                    progress_bar.progress(pct)
                    status_text.text(msg)

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

                st.session_state.lineups = lineups
                st.session_state.optimization_complete = True

                # Clear progress
                progress_bar.empty()
                status_text.empty()

                st.success(f"✅ Successfully generated {len(lineups)} lineups!")
                st.balloons()

            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")
                with st.expander("Error Details"):
                    st.exception(e)

# Display Results
if st.session_state.optimization_complete and st.session_state.lineups is not None:
    lineups = st.session_state.lineups

    st.divider()
    st.header("📈 Results")

    # Summary Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Lineups Generated", len(lineups))
    col2.metric("Avg Projection", f"{lineups['Projected'].mean():.2f}")
    col3.metric("Avg Ownership", f"{lineups['Total_Own'].mean():.1f}%")
    col4.metric("Unique Captains", lineups['CPT'].nunique())
    col5.metric("Avg Salary", f"${lineups['Total_Salary'].mean():,.0f}")

    # Simulation metrics if available
    if 'Sim_Ceiling_90th' in lineups.columns:
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Ceiling (90th)", f"{lineups['Sim_Ceiling_90th'].mean():.2f}")
        col2.metric("Avg Sharpe Ratio", f"{lineups['Sim_Sharpe'].mean():.2f}")
        col3.metric("Avg Win Prob", f"{lineups['Sim_Win_Prob'].mean():.1%}")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 All Lineups",
        "🎯 Top Lineups",
        "🤖 AI Analysis",
        "📊 Insights"
    ])

    with tab1:
        st.dataframe(lineups, use_container_width=True)

    with tab2:
        st.subheader("Top 10 Lineups by Projection")
        top_lineups = lineups.nlargest(10, 'Projected')

        for idx, lineup in top_lineups.iterrows():
            with st.expander(f"Lineup #{lineup['Lineup']} - {lineup['Projected']:.2f} pts - {lineup['Total_Own']:.1f}% own"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**Captain:** {lineup['CPT']}")
                    st.markdown("**FLEX:**")
                    for i in range(1, 6):
                        st.markdown(f"- {lineup[f'FLEX{i}']}")

                with col2:
                    st.metric("Salary", f"${lineup['Total_Salary']:,.0f}")
                    st.metric("Remaining", f"${lineup['Remaining']:,.0f}")
                    st.metric("Ownership", f"{lineup['Total_Own']:.1f}%")
                    if 'Sim_Ceiling_90th' in lineup:
                        st.metric("Ceiling", f"{lineup['Sim_Ceiling_90th']:.2f}")

    with tab3:
        st.subheader("AI Strategy Distribution")

        if 'Strategy' in lineups.columns:
            strategy_counts = lineups['Strategy'].value_counts()

            col1, col2 = st.columns([1, 2])

            with col1:
                st.dataframe(strategy_counts.reset_index().rename(
                    columns={'index': 'Strategy', 'Strategy': 'Count'}
                ))

            with col2:
                st.bar_chart(strategy_counts)

            st.markdown("---")

            # Show example from each strategy
            for strategy in strategy_counts.index:
                strategy_lineups = lineups[lineups['Strategy'] == strategy]
                if not strategy_lineups.empty:
                    example = strategy_lineups.iloc[0]

                    with st.expander(f"📌 {strategy} Strategy Example"):
                        st.markdown(f"**Captain:** {example['CPT']}")
                        st.markdown(f"**Projection:** {example['Projected']:.2f} pts")
                        st.markdown(f"**Ownership:** {example['Total_Own']:.1f}%")

                        strategy_descriptions = {
                            'Game Theory': "Leverages ownership inefficiencies",
                            'Correlation': "Maximizes correlated scoring",
                            'Contrarian Narrative': "Unique tournament-winning angle"
                        }

                        if strategy in strategy_descriptions:
                            st.info(strategy_descriptions[strategy])
        else:
            st.info("Strategy information not available")

    with tab4:
        st.subheader("📊 Lineup Insights")

        # Captain distribution
        st.markdown("**Captain Usage:**")
        captain_counts = lineups['CPT'].value_counts().head(10)
        st.bar_chart(captain_counts)

        # Ownership distribution
        st.markdown("**Ownership Distribution:**")
        fig_data = pd.DataFrame({
            'Ownership Range': ['<60%', '60-80%', '80-100%', '>100%'],
            'Count': [
                len(lineups[lineups['Total_Own'] < 60]),
                len(lineups[(lineups['Total_Own'] >= 60) & (lineups['Total_Own'] < 80)]),
                len(lineups[(lineups['Total_Own'] >= 80) & (lineups['Total_Own'] < 100)]),
                len(lineups[lineups['Total_Own'] >= 100])
            ]
        })
        st.bar_chart(fig_data.set_index('Ownership Range'))

        # Salary utilization
        st.markdown("**Salary Utilization:**")
        avg_salary = lineups['Total_Salary'].mean()
        salary_pct = (avg_salary / 50000) * 100
        st.progress(salary_pct / 100)
        st.caption(f"Average: ${avg_salary:,.0f} ({salary_pct:.1f}% of cap)")

    # Download section
    st.divider()
    st.subheader("💾 Download Lineups")

    col1, col2, col3 = st.columns(3)

    with col1:
        # CSV download
        csv = lineups.to_csv(index=False)
        st.download_button(
            label="📥 Download Full CSV",
            data=csv,
            file_name=f"showdown_lineups_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv',
            use_container_width=True
        )

    with col2:
        # DK format download
        dk_cols = ['CPT', 'FLEX1', 'FLEX2', 'FLEX3', 'FLEX4', 'FLEX5']
        if all(col in lineups.columns for col in dk_cols):
            dk_csv = lineups[dk_cols].to_csv(index=False)
            st.download_button(
                label="📥 Download DK Format",
                data=dk_csv,
                file_name=f"dk_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv',
                use_container_width=True
            )

    with col3:
        # Report download
        if st.button("📊 Generate Report", use_container_width=True):
            report = st.session_state.optimizer.get_optimization_report()
            report_text = f"""
NFL Showdown Optimization Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY:
- Lineups Generated: {len(lineups)}
- Avg Projection: {lineups['Projected'].mean():.2f}
- Avg Ownership: {lineups['Total_Own'].mean():.1f}%
- Unique Captains: {lineups['CPT'].nunique()}

PERFORMANCE:
{report.get('performance', {})}

ENFORCEMENT:
{report.get('enforcement', {})}
"""
            st.download_button(
                label="📥 Download Report",
                data=report_text,
                file_name=f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime='text/plain',
                use_container_width=True
            )

# Footer
st.divider()
st.caption("NFL Showdown Optimizer v2.0 | Powered by Claude AI & Advanced ML")
