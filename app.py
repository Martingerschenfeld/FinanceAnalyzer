import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.title('Stock Price Trend Analyzer')
st.write("Analyze multi-period price directions and patterns")

# User inputs
ticker = st.text_input('Enter stock ticker:', 'AAPL')
years_back = st.number_input('Amount of years to look back on:', min_value=1, max_value=10, value=3)
show_last = st.number_input('Number of recent days to display:', min_value=5, max_value=100, value=20)

if st.button('Run Analysis'):
    if ticker:
        with st.spinner('Fetching and analyzing data...'):
            try:
                # Calculate start date based on user input for years back
                start_date = datetime.now() - timedelta(days=365 * years_back)

                # Get data
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=datetime.now(),
                    progress=False
                )

                # Need at least 4 data points for t-3 to t, and 5 for t+1 analysis
                if len(data) >= 5:
                    # Create analysis dataframe
                    df = pd.DataFrame(data['Close'])
                    df.columns = ['Price']

                    # Calculate directions between periods
                    df['t vs t-1'] = np.where(df['Price'] > df['Price'].shift(1), '↑', '↓')
                    df['t-1 vs t-2'] = np.where(df['Price'].shift(1) > df['Price'].shift(2), '↑', '↓')
                    df['t-2 vs t-3'] = np.where(df['Price'].shift(2) > df['Price'].shift(3), '↑', '↓')

                    # Create pattern indicator (e.g., 'udd', 'duu', etc.)
                    df['Pattern'] = df['t-2 vs t-3'] + df['t-1 vs t-2'] + df['t vs t-1']

                    # Add human-readable pattern description
                    pattern_names = {
                        '↑↑↑': 'Strong uptrend',
                        '↑↑↓': 'Pullback after 2 gains',
                        '↑↓↑': 'Recovery after dip',
                        '↑↓↓': 'Start of downtrend',
                        '↓↑↑': 'Start of uptrend',
                        '↓↑↓': 'Volatile, no trend',
                        '↓↓↑': 'Rebound after 2 drops',
                        '↓↓↓': 'Strong downtrend'
                    }
                    df['Pattern Name'] = df['Pattern'].map(pattern_names)

                    # Calculate t+1 vs t direction
                    df['t+1 vs t'] = np.where(df['Price'].shift(-1) > df['Price'], '↑', '↓')

                    # Add date info
                    df['Date'] = df.index
                    df['Day'] = df.index.day_name()

                    # Format display columns
                    display_cols = ['Date', 'Day', 'Price', 't vs t-1', 't-1 vs t-2',
                                    't-2 vs t-3', 'Pattern', 'Pattern Name', 't+1 vs t']
                    display_df = df[display_cols].copy().dropna() # dropna will remove rows with NaNs from shifts

                    st.success(f"Analysis complete for {ticker} over the last {years_back} year(s).")

                    #---
                    st.subheader(f"Recent Price Directions")
                    #---
                    recent_df = display_df.tail(show_last)

                    # Color formatting function
                    def color_direction(val):
                        if val == '↑': return 'color: green'
                        elif val == '↓': return 'color: red'
                        return ''

                    st.dataframe(
                        recent_df.style
                        .format({'Price': '${:.2f}'})
                        .applymap(color_direction, subset=['t vs t-1', 't-1 vs t-2', 't-2 vs t-3', 't+1 vs t'])
                    )

                    #---
                    st.subheader("Direction Change Frequencies")
                    #---
                    cols = st.columns(3)
                    # Use df.dropna for percentages to ensure accurate base for calculations
                    df_t_vs_t_1_cleaned = df.dropna(subset=['t vs t-1'])
                    df_t_1_vs_t_2_cleaned = df.dropna(subset=['t-1 vs t-2'])
                    df_t_2_vs_t_3_cleaned = df.dropna(subset=['t-2 vs t-3'])

                    cols[0].metric("t vs t-1 Up",
                                   f"{len(df_t_vs_t_1_cleaned[df_t_vs_t_1_cleaned['t vs t-1'] == '↑'])} days",
                                   f"{len(df_t_vs_t_1_cleaned[df_t_vs_t_1_cleaned['t vs t-1'] == '↑'])/len(df_t_vs_t_1_cleaned)*100:.1f}%")
                    cols[1].metric("t-1 vs t-2 Up",
                                   f"{len(df_t_1_vs_t_2_cleaned[df_t_1_vs_t_2_cleaned['t-1 vs t-2'] == '↑'])} days",
                                   f"{len(df_t_1_vs_t_2_cleaned[df_t_1_vs_t_2_cleaned['t-1 vs t-2'] == '↑'])/len(df_t_1_vs_t_2_cleaned)*100:.1f}%")
                    cols[2].metric("t-2 vs t-3 Up",
                                   f"{len(df_t_2_vs_t_3_cleaned[df_t_2_vs_t_3_cleaned['t-2 vs t-3'] == '↑'])} days",
                                   f"{len(df_t_2_vs_t_3_cleaned[df_t_2_vs_t_3_cleaned['t-2 vs t-3'] == '↑'])/len(df_t_2_vs_t_3_cleaned)*100:.1f}%")

                    #---
                    st.subheader("Overall Pattern Probabilities")
                    #---
                    # Calculate overall pattern probabilities
                    pattern_counts_overall = df['Pattern Name'].value_counts(normalize=True) * 100
                    pattern_prob_df_overall = pattern_counts_overall.reset_index()
                    pattern_prob_df_overall.columns = ['Pattern Name', 'Probability (%)']

                    # Ensure all 8 patterns are present, even if their count is 0
                    all_patterns_df_base = pd.DataFrame(pattern_names.items(), columns=['Pattern', 'Pattern Name'])

                    # Merge with calculated probabilities
                    merged_patterns_overall = pd.merge(all_patterns_df_base, pattern_prob_df_overall, on='Pattern Name', how='left').fillna(0)
                    merged_patterns_overall = merged_patterns_overall[['Pattern Name', 'Pattern', 'Probability (%)']]
                    merged_patterns_overall = merged_patterns_overall.sort_values(by='Pattern')
                    st.dataframe(merged_patterns_overall.style.format({'Probability (%)': '{:.2f}%'}))

                    #---
                    st.subheader("Next Period (t+1) Direction Probabilities by Pattern")
                    #---

                    # Drop NA values for pattern and t+1 vs t calculation
                    df_for_conditional_prob = df.dropna(subset=['Pattern Name', 't+1 vs t'])

                    # Group by pattern and count occurrences of '↑' and '↓' for 't+1 vs t'
                    conditional_counts = df_for_conditional_prob.groupby('Pattern Name')['t+1 vs t'].value_counts().unstack(fill_value=0)

                    # Calculate probabilities
                    conditional_probabilities = conditional_counts.apply(lambda x: x / x.sum() * 100, axis=1)

                    # Rename columns for clarity
                    conditional_probabilities.columns = [f"t+1 is {col} (%)" for col in conditional_probabilities.columns]

                    # Merge with all_patterns_df_base to ensure all 8 patterns are listed
                    # Fill NaN for patterns that might not have occurred
                    final_conditional_prob_df = pd.merge(
                        all_patterns_df_base,
                        conditional_probabilities,
                        on='Pattern Name',
                        how='left'
                    ).fillna(0) # Fill with 0 for patterns that never occurred or had no t+1 data

                    # Reorder and format for display
                    final_conditional_prob_df = final_conditional_prob_df[['Pattern Name', 'Pattern', 't+1 is ↑ (%)', 't+1 is ↓ (%)']]
                    final_conditional_prob_df = final_conditional_prob_df.sort_values(by='Pattern')

                    st.dataframe(final_conditional_prob_df.style.format({
                        't+1 is ↑ (%)': '{:.2f}%',
                        't+1 is ↓ (%)': '{:.2f}%'
                    }))

                    #---
                    st.subheader("Pattern Legend")
                    #---
                    pattern_df = pd.DataFrame.from_dict(pattern_names, orient='index', columns=['Description'])
                    st.table(pattern_df)

                else:
                    st.warning(f"Not enough historical data available for {ticker} over the last {years_back} year(s) to perform full analysis. Please try a different ticker or a longer look-back period (requires at least 5 data points).")

            except Exception as e:
                st.error(f"Error: {str(e)}. Please check the ticker symbol and try again.")
    else:
        st.warning("Please enter a ticker symbol")