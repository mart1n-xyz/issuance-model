import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="Karma Issuance Modelling",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Main title
st.title("Karma Issuance Modelling")

# Welcome message
st.markdown("""
Welcome to the Karma Issuance Modelling application! This tool is designed to help you model 
and analyze karma token issuance scenarios.

""")

# Yield Parameters Section
st.header("Yield Parameters")

col1, col2 = st.columns(2)

with col1:
    usd_yield = st.number_input(
        "USD Yield Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=4.75,
        step=0.01,
        format="%.2f",
        help="Annual yield rate for USD assets"
    )

with col2:
    eth_yield = st.number_input(
        "ETH Yield Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=2.6,
        step=0.01,
        format="%.2f",
        help="Annual yield rate for ETH assets"
    )

# Display current settings
st.markdown("### Current Yield Settings")
col1, col2 = st.columns(2)
with col1:
    st.metric("USD Yield", f"{usd_yield}%")
with col2:
    st.metric("ETH Yield", f"{eth_yield}%")

st.divider()

# Data Selection and Visualization Section
st.header("L2 Network Data Analysis")

# Get list of available datasets
import os
data_folder = "l2_data"
if os.path.exists(data_folder):
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    dataset_options = [f.replace('.csv', '').replace('-', ' ').title() for f in csv_files]
    dataset_files = {opt: f for opt, f in zip(dataset_options, csv_files)}
    
    # Default to Linea
    default_index = 0
    for i, opt in enumerate(dataset_options):
        if 'linea' in opt.lower():
            default_index = i
            break
    
    # Dataset selection
    selected_dataset = st.selectbox(
        "Select L2 Network Dataset:",
        options=dataset_options,
        index=default_index,
        help="Choose which L2 network data to visualize"
    )
    
    # Load and display data
    if selected_dataset:
        try:
            file_path = os.path.join(data_folder, dataset_files[selected_dataset])
            df = pd.read_csv(file_path)
            
            # Convert timestamp to datetime if needed
            if 'Timestamp' in df.columns:
                df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Trim data to periods where both values are available
            numeric_columns = [col for col in df.columns if col not in ['Timestamp', 'Date']]
            if len(numeric_columns) >= 2:
                # Find first and last rows where all numeric values are not null
                df_clean = df.dropna(subset=numeric_columns)
                if not df_clean.empty:
                    df = df_clean
            
            # Display dataset info
            st.markdown(f"### {selected_dataset} Dataset")
            st.metric("Date Range (Based on data availability)", f"{df['Date'].min().strftime('%B %d, %Y')} - {df['Date'].max().strftime('%B %d, %Y')}")
            st.caption("Source: DeFiLlama")
            
            # Create visualization
            st.markdown("### Data Visualization")
            
            # Prepare data for plotting
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Timestamp' in numeric_columns:
                numeric_columns.remove('Timestamp')
            
            if len(numeric_columns) > 0:
                # Create multi-line chart
                fig = go.Figure()
                
                for col in numeric_columns:
                    if col != 'Timestamp':
                        fig.add_trace(go.Scatter(
                            x=df['Date'],
                            y=df[col],
                            mode='lines',
                            name=col,
                            line=dict(width=2)
                        ))
                
                fig.update_layout(
                    title=f"{selected_dataset} - Time Series Data",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    hovermode='x unified',
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns found for visualization.")
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            
else:
    st.warning("l2_data folder not found. Please ensure the data folder exists in the project directory.")

st.divider()

# Karma Issuance Modeling Section
st.header("⚡ Karma Issuance Modeling")

# SNT Staking Parameters
st.subheader("SNT Staking Parameters")
col1, col2 = st.columns(2)

with col1:
    snt_staked_pct = st.number_input(
        "% of SNT Staked",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.1,
        format="%.1f",
        help="Percentage of total SNT supply staked (Total supply: 6,804,870,174 SNT)"
    )

with col2:
    karma_multiplier = st.number_input(
        "Average Multiplier",
        min_value=0.0,
        value=2.0,
        step=0.1,
        format="%.1f",
        help="Average karma boost multiplier based on MPs (Multiplier Points) accrued by stakers"
    )

# Calculate actual SNT staked amount
snt_total_supply = 6_804_870_174
snt_staked_amount = (snt_staked_pct / 100) * snt_total_supply

# Calculate SNT Factor
stake_ratio = snt_staked_pct / 100
snt_factor = 1 + (stake_ratio * karma_multiplier)

# Display calculated values
st.markdown("### Current Staking Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("SNT Staked", f"{snt_staked_amount:,.0f}")
with col2:
    st.metric("Staking Ratio", f"{snt_staked_pct:.1f}% of supply")
with col3:
    st.metric("SNT Factor", f"{snt_factor:.2f}")

st.divider()

# App Yield Section
st.subheader("App Fee Parameters")

app_yield_pct = st.number_input(
    "Annual App Fees (% of TVL)",
    min_value=0.00,
    value=0.05,
    step=0.01,
    format="%.2f",
    help="Annual app fees as percentage of TVL - this varies over time with TVL changes (modeling simplification)"
)

col1, col2 = st.columns(2)
with col1:
    st.metric("App Fees", f"{app_yield_pct:.2f}% of TVL")
with col2:
    st.info("📌 **Note**: We assume app yield varies proportionally with TVL over time")

st.divider()

# Further Parameters
st.subheader("Further Parameters")

col1, col2 = st.columns(2)

with col1:
    ops_commission_pct = st.number_input(
        "Operations Commission - OpsCom (%)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=0.1,
        format="%.1f",
        help="L2 operations commission - can be modified through karma governance in the future"
    )

with col2:
    eth_tvl_pct = st.number_input(
        "% of TVL in ETH",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=0.1,
        format="%.1f",
        help="Percentage of total TVL that is denominated in ETH"
    )

# Formula explanation
st.markdown("### Yield Calculation Formula")
st.markdown("""
**L2Yield = NativeYield + AppFees**

**NetYield = (1 - OpsCom) × L2Yield**
""")

# Display current settings
col1, col2 = st.columns(2)
with col1:
    st.metric("Operations Commission", f"{ops_commission_pct:.1f}%")
with col2:
    st.metric("ETH Portion of TVL", f"{eth_tvl_pct:.1f}%")

net_yield_factor = (100 - ops_commission_pct) / 100

st.divider()

# Daily Yield Calculation and Visualization
st.header("Daily Yield Analysis")

# Only proceed if we have selected data from above
if 'df' in locals() and not df.empty:
    st.subheader(f"Yield Analysis for {selected_dataset}")
    
    # Calculate daily yields
    analysis_df = df.copy()
    
    # Calculate ETH portion of TVL using the % parameter
    analysis_df['ETH_TVL_USD'] = analysis_df['Bridged TVL'] * (eth_tvl_pct / 100)
    
    # Note: All yields calculated in USD values
    # ETH TVL is calculated as a percentage of total TVL
    analysis_df['Daily_ETH_Yield_USD'] = analysis_df['ETH_TVL_USD'] * (eth_yield / 100) / 365
    analysis_df['Daily_Stable_Yield_USD'] = analysis_df['Stablecoins Mcap'] * (usd_yield / 100) / 365
    
    # Calculate app fees based on total TVL (USD)
    analysis_df['Daily_App_Fees_USD'] = analysis_df['Bridged TVL'] * (app_yield_pct / 100) / 365
    
    # Calculate L2 Yield (NativeYield + AppFees) - all in USD
    analysis_df['Daily_Native_Yield_USD'] = analysis_df['Daily_ETH_Yield_USD'] + analysis_df['Daily_Stable_Yield_USD']
    analysis_df['Daily_L2_Yield_USD'] = analysis_df['Daily_Native_Yield_USD'] + analysis_df['Daily_App_Fees_USD']
    
    # Calculate Net Yield after operations commission (USD)
    analysis_df['Daily_Net_Yield_USD'] = analysis_df['Daily_L2_Yield_USD'] * net_yield_factor
    
    # Display key metrics (all in USD)

    st.info("📌 **Note**: For the time being, please ignore the absolute values of the yields. This dashboard aims to focus on the relative changes in yields driving the dynamics of karma issuance.")
    st.info("📌 **Note**: All yields shown in USD values. ETH TVL portion is denominated in ETH but yield is calculated on USD equivalent.")
    
  
    # Create visualization
    st.markdown("### Daily Yield Breakdown")
    
    fig = go.Figure()
    
    # Add yield components (all in USD)
    fig.add_trace(go.Scatter(
        x=analysis_df['Date'],
        y=analysis_df['Daily_ETH_Yield_USD'],
        name=f'ETH Yield ({eth_yield}%)',
        line=dict(color='blue', width=2),
        stackgroup='one'
    ))
    
    fig.add_trace(go.Scatter(
        x=analysis_df['Date'],
        y=analysis_df['Daily_Stable_Yield_USD'],
        name=f'Stablecoin Yield ({usd_yield}%)',
        line=dict(color='green', width=2),
        stackgroup='one'
    ))
    
    fig.add_trace(go.Scatter(
        x=analysis_df['Date'],
        y=analysis_df['Daily_App_Fees_USD'],
        name=f'App Fees ({app_yield_pct}%)',
        line=dict(color='orange', width=2),
        stackgroup='one'
    ))
    
    # Add net yield line
    fig.add_trace(go.Scatter(
        x=analysis_df['Date'],
        y=analysis_df['Daily_Net_Yield_USD'],
        name=f'Net Yield (after {ops_commission_pct}% OpsCom)',
        line=dict(color='red', width=3, dash='dash'),
        mode='lines'
    ))
    
    fig.update_layout(
        title=f"{selected_dataset} - Daily Yield Analysis (USD)",
        xaxis_title="Date",
        yaxis_title="Daily Yield (USD)",
        hovermode='x unified',
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Weekly Aggregated Net Yield
    st.markdown("### Weekly Net Yield Summary")
    
    # Aggregate by week
    analysis_df['Week'] = analysis_df['Date'].dt.to_period('W').dt.start_time
    weekly_yield = analysis_df.groupby('Week')['Daily_Net_Yield_USD'].sum().reset_index()
    weekly_yield['Week_Label'] = weekly_yield['Week'].dt.strftime('%Y-%m-%d')
    
    # Create bar chart
    fig_weekly = go.Figure()
    
    fig_weekly.add_trace(go.Bar(
        x=weekly_yield['Week'],
        y=weekly_yield['Daily_Net_Yield_USD'],
        name='Weekly Net Yield',
        marker_color='steelblue',
        hovertemplate='<b>Week of %{x}</b><br>Net Yield: $%{y:,.2f}<extra></extra>'
    ))
    
    fig_weekly.update_layout(
        title=f"{selected_dataset} - Weekly Net Yield Accumulation",
        xaxis_title="Week",
        yaxis_title="Weekly Net Yield (USD)",
        hovermode='x',
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig_weekly, use_container_width=True)
    
    # Display weekly summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Weekly Yield", f"${weekly_yield['Daily_Net_Yield_USD'].mean():,.2f}")
    with col2:
        st.metric("Highest Weekly Yield", f"${weekly_yield['Daily_Net_Yield_USD'].max():,.2f}")
    with col3:
        st.metric("Total Yield Period", f"${weekly_yield['Daily_Net_Yield_USD'].sum():,.2f}")
    
    st.divider()
    
    # KARMA Minting Section
    st.markdown("### KARMA Minting Calculation")
    
    # KARMA minting parameter
    k_parameter = st.number_input(
        "k: KARMA minted per 1 ETH of NetYield per epoch",
        min_value=0.00,
        value=1.00,
        step=0.01,
        format="%.2f",
        help="Multiplier for converting net yield to KARMA tokens"
    )
    
    st.markdown("""
    **Formula**: BaseKarma = k × NetYield (smoothed with 4-week EMA)
    
    Net yield is smoothed using a 4-week Exponential Moving Average to reduce epoch-to-epoch volatility.
    """)
    
    # Convert net yield to ETH equivalent (assuming 1 USD = 1/ETH_price)
    # For simplification, we'll use a representative ETH price or keep in USD terms
    # Load ETH price data for conversion
    try:
        eth_df = pd.read_csv("eth_price.csv")
        eth_df['Date'] = pd.to_datetime(eth_df['Timestamp'], unit='ms')
        eth_df = eth_df[['Date', 'Token Price']].rename(columns={'Token Price': 'ETH_Price'})
        
        # Merge with analysis data
        karma_df = analysis_df.merge(eth_df, on='Date', how='left')
        karma_df['ETH_Price'] = karma_df['ETH_Price'].fillna(method='ffill').fillna(method='bfill')
        
        # Calculate weekly average ETH price and add to daily data
        karma_df['Week'] = karma_df['Date'].dt.to_period('W').dt.start_time
        weekly_eth_avg = karma_df.groupby('Week')['ETH_Price'].mean().reset_index()
        weekly_eth_avg.rename(columns={'ETH_Price': 'Weekly_Avg_ETH_Price'}, inplace=True)
        karma_df = karma_df.merge(weekly_eth_avg, on='Week', how='left')
        
        # Convert daily net yield to ETH terms using weekly average ETH price
        karma_df['Daily_Net_Yield_ETH'] = karma_df['Daily_Net_Yield_USD'] / karma_df['Weekly_Avg_ETH_Price']
        
        # Create weekly aggregations for karma calculation
        weekly_karma_df = karma_df.groupby('Week').agg({
            'Daily_Net_Yield_USD': 'sum',  # Sum daily yields to get weekly total
            'Weekly_Avg_ETH_Price': 'first'  # Use the weekly average ETH price
        }).reset_index()
        
        # Convert weekly net yield to ETH terms
        weekly_karma_df['Weekly_Net_Yield_ETH'] = weekly_karma_df['Daily_Net_Yield_USD'] / weekly_karma_df['Weekly_Avg_ETH_Price']
        
        # Calculate 4-week EMA (literally 4 data points)
        weekly_karma_df['NetYield_EMA'] = weekly_karma_df['Weekly_Net_Yield_ETH'].ewm(span=4).mean()
        
        # Calculate Base Karma
        weekly_karma_df['Base_Karma'] = k_parameter * weekly_karma_df['NetYield_EMA']
        
        # Calculate SNT adjusted Karma using SNT_factor from earlier
        weekly_karma_df['Karma_SNT'] = weekly_karma_df['Base_Karma'] * snt_factor
        
        # Calculate daily minting rates (1/7 of weekly)
        weekly_karma_df['Daily_Base_Karma'] = weekly_karma_df['Base_Karma'] / 7
        weekly_karma_df['Daily_Karma_SNT'] = weekly_karma_df['Karma_SNT'] / 7
        
        # Display current metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current k Parameter", f"{k_parameter:.2f}")
        with col2:
            st.metric("SNT Factor", f"{snt_factor:.2f}")
        
        # Create daily KARMA minting visualization with flat weekly segments
        st.markdown("### Daily KARMA Minting Over Time")
        
        # Create daily data with actual daily yields and flat karma segments for each week
        daily_data = []
        for _, week_row in weekly_karma_df.iterrows():
            week_start = week_row['Week']
            daily_base_karma = week_row['Daily_Base_Karma']
            daily_snt_karma = week_row['Daily_Karma_SNT']
            
            # Create 7 days for this week
            for day_offset in range(7):
                daily_date = week_start + pd.Timedelta(days=day_offset)
                daily_data.append({
                    'Date': daily_date,
                    'Daily_Base_Karma': daily_base_karma,
                    'Daily_Karma_SNT': daily_snt_karma
                })
        
        daily_df = pd.DataFrame(daily_data)
        
        # Merge with original daily net yield data (in ETH terms)
        original_daily_df = karma_df[['Date', 'Daily_Net_Yield_ETH']].copy()
        daily_df = daily_df.merge(original_daily_df, on='Date', how='left')
        
        fig_karma = go.Figure()
        
        # Add daily net yield (ETH terms) - actual daily values
        fig_karma.add_trace(go.Scatter(
            x=daily_df['Date'],
            y=daily_df['Daily_Net_Yield_ETH'],
            name='Daily Net Yield (ETH)',
            line=dict(color='blue', width=2),  # smooth line for actual daily values
            mode='lines'
        ))
        
        # Add Base Karma (secondary y-axis)
        fig_karma.add_trace(go.Scatter(
            x=daily_df['Date'],
            y=daily_df['Daily_Base_Karma'],
            name=f'Daily Base Karma (k={k_parameter})',
            line=dict(color='red', width=2, shape='hv'),  # step-wise line
            yaxis='y2'
        ))
        
        # Add SNT adjusted Karma (secondary y-axis)
        fig_karma.add_trace(go.Scatter(
            x=daily_df['Date'],
            y=daily_df['Daily_Karma_SNT'],
            name=f'Daily SNT Karma (×{snt_factor:.2f})',
            line=dict(color='orange', width=3, shape='hv'),  # step-wise line
            yaxis='y2'
        ))
        
        # Update layout with secondary y-axis
        fig_karma.update_layout(
            title=f"{selected_dataset} - Daily KARMA Minting Analysis",
            xaxis_title="Date",
            yaxis=dict(title="Daily Net Yield (ETH/day)", side="left"),
            yaxis2=dict(title="Daily KARMA Minting (KARMA/day)", side="right", overlaying="y"),
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig_karma, use_container_width=True)
        
        st.divider()
        
        # Total Supply Section
        st.header("Total Supply")
        st.markdown("Analysis of cumulative Karma token supply based on weekly minting (SNT-adjusted values)")
        
        # Calculate cumulative total supply
        weekly_karma_df['Cumulative_Karma_Supply'] = weekly_karma_df['Karma_SNT'].cumsum()
        
        # Weekly minting statistics
        st.markdown("### Weekly Minting Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_weekly_mint = weekly_karma_df['Karma_SNT'].mean()
            st.metric("Average Weekly Mint", f"{avg_weekly_mint:,.2f} KARMA")
        
        with col2:
            max_weekly_mint = weekly_karma_df['Karma_SNT'].max()
            st.metric("Peak Weekly Mint", f"{max_weekly_mint:,.2f} KARMA")
        
        with col3:
            total_karma_minted = weekly_karma_df['Karma_SNT'].sum()
            st.metric("Total Karma Minted", f"{total_karma_minted:,.0f} KARMA")
        
        with col4:
            current_supply = weekly_karma_df['Cumulative_Karma_Supply'].iloc[-1]
            st.metric("Current Total Supply", f"{current_supply:,.0f} KARMA")
        
        # Cumulative supply visualization
        st.markdown("### Cumulative Karma Total Supply")
        
        fig_supply = go.Figure()
        
        # Add cumulative supply line
        fig_supply.add_trace(go.Scatter(
            x=weekly_karma_df['Week'],
            y=weekly_karma_df['Cumulative_Karma_Supply'],
            name='Total Karma Supply',
            line=dict(color='purple', width=3),
            fill='tonexty',
            mode='lines'
        ))
        
        # Add weekly minting as bar chart (secondary y-axis)
        fig_supply.add_trace(go.Bar(
            x=weekly_karma_df['Week'],
            y=weekly_karma_df['Karma_SNT'],
            name='Weekly Karma Mint',
            marker_color='lightblue',
            opacity=0.6,
            yaxis='y2'
        ))
        
        # Update layout with secondary y-axis
        fig_supply.update_layout(
            title=f"{selected_dataset} - Karma Total Supply Growth",
            xaxis_title="Week",
            yaxis=dict(title="Cumulative Karma Supply", side="left"),
            yaxis2=dict(title="Weekly Karma Mint", side="right", overlaying="y"),
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig_supply, use_container_width=True)
        
        # Growth rate analysis
        st.markdown("### Supply Growth Analysis")
        
        # Calculate week-over-week growth rate
        weekly_karma_df['Supply_Growth_Rate'] = weekly_karma_df['Cumulative_Karma_Supply'].pct_change() * 100
        
        st.divider()
        
        # Karma Value Analysis Section
        st.header("Karma Value Analysis")
        st.markdown("Analysis of Karma value based on weekly yield backing and supply dynamics")
        
        # Calculate weekly yield values in USD
        weekly_karma_df['Weekly_Yield_USD'] = weekly_karma_df['Daily_Net_Yield_USD']
        
        # Calculate value per Karma token
        weekly_karma_df['Yield_Per_Karma'] = weekly_karma_df['Weekly_Yield_USD'] / weekly_karma_df['Karma_SNT']
        
        # Calculate Karma as % of total supply
        weekly_karma_df['Weekly_Mint_Pct'] = (weekly_karma_df['Karma_SNT'] / weekly_karma_df['Cumulative_Karma_Supply']) * 100
        
        # Value metrics
        st.markdown("### Weekly Value Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            latest_weekly_yield = weekly_karma_df['Weekly_Yield_USD'].iloc[-1]
            st.metric("Latest Weekly Yield", f"${latest_weekly_yield:,.2f}")
        
        with col2:
            latest_karma_minted = weekly_karma_df['Karma_SNT'].iloc[-1]
            st.metric("Latest Karma Minted", f"{latest_karma_minted:,.2f} KARMA")
        
        with col3:
            latest_yield_per_karma = weekly_karma_df['Yield_Per_Karma'].iloc[-1]
            st.metric("Latest Yield per Karma", f"${latest_yield_per_karma:.4f}")
        
        with col4:
            avg_yield_per_karma = weekly_karma_df['Yield_Per_Karma'].mean()
            st.metric("Average Yield per Karma", f"${avg_yield_per_karma:.4f}")
        
        # Karma Value Visualization
        st.markdown("### Karma Value Backing Over Time")
        
        fig_value = go.Figure()
        
        # Add total weekly yield (left y-axis)
        fig_value.add_trace(go.Scatter(
            x=weekly_karma_df['Week'],
            y=weekly_karma_df['Weekly_Yield_USD'],
            name='Weekly Yield (USD)',
            line=dict(color='green', width=2),
            yaxis='y'
        ))
        
        # Add yield per karma (right y-axis)
        fig_value.add_trace(go.Scatter(
            x=weekly_karma_df['Week'],
            y=weekly_karma_df['Yield_Per_Karma'],
            name='Yield per Karma (USD)',
            line=dict(color='orange', width=3),
            yaxis='y2'
        ))
        
        # Update layout with dual y-axis
        fig_value.update_layout(
            title=f"{selected_dataset} - Karma Value Backing Analysis",
            xaxis_title="Week",
            yaxis=dict(title="Weekly Yield (USD)", side="left"),
            yaxis2=dict(title="Yield per Karma (USD)", side="right", overlaying="y"),
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig_value, use_container_width=True)
        
        # Inflation and TVL Analysis
        st.markdown("### Inflation Rate vs TVL Growth")
        st.info("📌 **Note**: Inflation rates shown are weekly percentage changes, not annualized rates.")
        
        # Get original TVL data for comparison
        tvl_weekly = analysis_df.groupby('Week')['Bridged TVL'].mean().reset_index()
        tvl_weekly['TVL_Growth_Rate'] = tvl_weekly['Bridged TVL'].pct_change() * 100
        
        # Merge with karma data
        inflation_df = weekly_karma_df.merge(tvl_weekly, on='Week', how='left')
        
        fig_inflation = go.Figure()
        
        # Add Karma inflation rate (left y-axis)
        fig_inflation.add_trace(go.Scatter(
            x=inflation_df['Week'],
            y=inflation_df['Supply_Growth_Rate'],
            name='Karma Inflation Rate (%)',
            line=dict(color='red', width=2),
            yaxis='y'
        ))
        
        # Add TVL (right y-axis, area chart)
        fig_inflation.add_trace(go.Scatter(
            x=inflation_df['Week'],
            y=inflation_df['Bridged TVL'],
            name='Bridged TVL',
            line=dict(color='blue', width=2),
            fill='tonexty',
            opacity=0.3,
            yaxis='y2'
        ))
        
        # Update layout with dual y-axis
        fig_inflation.update_layout(
            title=f"{selected_dataset} - Karma Inflation vs TVL Growth",
            xaxis_title="Week",
            yaxis=dict(title="Karma Inflation Rate (%)", side="left"),
            yaxis2=dict(title="Bridged TVL", side="right", overlaying="y"),
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig_inflation, use_container_width=True)
        
        # Summary statistics
        st.markdown("### Value & Inflation Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_inflation = weekly_karma_df['Supply_Growth_Rate'].mean()
            st.metric("Average Karma Inflation", f"{avg_inflation:.2f}%/week")
        
        with col2:
            avg_tvl_growth = tvl_weekly['TVL_Growth_Rate'].mean()
            st.metric("Average TVL Growth", f"{avg_tvl_growth:.2f}%/week")
        
        with col3:
            value_stability = weekly_karma_df['Yield_Per_Karma'].std() / weekly_karma_df['Yield_Per_Karma'].mean() * 100
            st.metric("Yield/Karma Volatility", f"{value_stability:.1f}%")

    except Exception as e:
        st.error(f"Error loading ETH price data for KARMA calculation: {str(e)}")
        st.info("Please ensure eth_price.csv is available for ETH price conversion.")
    
else:
    st.info("Please select a dataset in the L2 Network Data Analysis section above to enable yield calculations.")

