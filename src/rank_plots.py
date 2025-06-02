import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def plot_keyword_ranks_over_time(df, token_col, method_name, top_n=8, 
                                period_freq='M', date_col='created_at', 
                                figsize=(24, 16)):
    """
    Generate keyword rank plots with improved spacing
    """
    print(f"\nGenerating keyword rank plots for: {method_name}")
    
    # Prepare data
    df_plot = df.copy()
    df_plot[date_col] = pd.to_datetime(df_plot[date_col], errors='coerce')
    df_plot = df_plot.dropna(subset=[date_col])
    df_plot = df_plot.set_index(date_col).sort_index()
    
    # Get top keywords
    all_tokens = []
    for tokens_list in df_plot[token_col].dropna():
        if isinstance(tokens_list, list):
            all_tokens.extend(tokens_list)
    
    top_keywords = [token for token, count in Counter(all_tokens).most_common(top_n)]
    print(f"Tracking top {top_n} keywords: {top_keywords}")
    
    # Calculate ranks for each period
    ranks_data = []
    period_stats = []
    
    for period_start, group in df_plot.groupby(pd.Grouper(freq=period_freq)):
        if not group.empty:
            period_label = period_start.strftime('%Y-%m')
            
            # Count keywords
            period_tokens = []
            for tokens_list in group[token_col].dropna():
                if isinstance(tokens_list, list):
                    period_tokens.extend(tokens_list)
            
            period_counter = Counter(period_tokens)
            total_mentions = sum(period_counter.values())
            
            # Get rankings
            sorted_keywords = sorted(period_counter.items(), key=lambda x: x[1], reverse=True)
            rank_dict = {kw: rank+1 for rank, (kw, _) in enumerate(sorted_keywords)}
            
            # Store data
            for keyword in top_keywords:
                count = period_counter.get(keyword, 0)
                rank = rank_dict.get(keyword, top_n + 5)
                market_share = (count / total_mentions * 100) if total_mentions > 0 else 0
                
                ranks_data.append({
                    'Period': period_label,
                    'Period_Date': period_start,
                    'Keyword': keyword,
                    'Rank': min(rank, top_n + 1),
                    'Count': count,
                    'Market_Share': market_share
                })
            
            period_stats.append({
                'Period': period_label,
                'Total_Keywords': len(period_counter),
                'Total_Mentions': total_mentions
            })
    
    ranks_df = pd.DataFrame(ranks_data)
    stats_df = pd.DataFrame(period_stats)
    
    # Create figure with more spacing
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.5, 1, 1], width_ratios=[3, 1], 
                         hspace=0.6, wspace=0.4, left=0.05, right=0.95, top=0.95, bottom=0.1)
    
    # 1. Main rank plot
    ax1 = fig.add_subplot(gs[0, 0])
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_keywords)))
    
    for i, keyword in enumerate(top_keywords):
        keyword_data = ranks_df[ranks_df['Keyword'] == keyword]
        if len(keyword_data) > 1:
            ax1.plot(keyword_data['Period'], keyword_data['Rank'], 
                    marker='o', linewidth=2.5, markersize=6,
                    label=keyword, color=colors[i], alpha=0.8)
    
    ax1.set_ylim(top_n + 1.5, 0.5)
    ax1.set_title(f'Keyword Rankings Over Time - {method_name}', fontsize=18, pad=25)
    ax1.set_ylabel('Rank (1 = Most Frequent)', fontsize=14)
    ax1.set_xlabel('Period', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Move legend outside with more padding
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=11)
    
    # Rotate x-axis labels and show every 3rd label
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=11)
    x_ticks = ax1.get_xticks()
    x_labels = ax1.get_xticklabels()
    for i, label in enumerate(x_labels):
        if i % 3 != 0:
            label.set_visible(False)
    
    # 2. Keyword stability
    ax2 = fig.add_subplot(gs[0, 1])
    volatility_data = []
    for keyword in top_keywords:
        kw_data = ranks_df[ranks_df['Keyword'] == keyword]
        volatility = kw_data['Rank'].std()
        volatility_data.append({'Keyword': keyword, 'Volatility': volatility})
    
    vol_df = pd.DataFrame(volatility_data).sort_values('Volatility')
    bars = ax2.barh(range(len(vol_df)), vol_df['Volatility'], 
                   color='coral', height=0.7, alpha=0.7)
    ax2.set_yticks(range(len(vol_df)))
    ax2.set_yticklabels(vol_df['Keyword'], fontsize=11)
    ax2.set_xlabel('Rank Volatility', fontsize=12)
    ax2.set_title('Keyword Stability', fontsize=14, pad=15)
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Market share
    ax3 = fig.add_subplot(gs[1, 0])
    for i, keyword in enumerate(top_keywords[:5]):
        keyword_data = ranks_df[ranks_df['Keyword'] == keyword]
        ax3.plot(keyword_data['Period'], keyword_data['Market_Share'],
                marker='s', linewidth=2, markersize=4, label=keyword, 
                color=colors[i], alpha=0.7)
    
    ax3.set_ylabel('Market Share (%)', fontsize=12)
    ax3.set_xlabel('Period', fontsize=12)
    ax3.set_title('Keyword Market Share', fontsize=14, pad=15)
    ax3.legend(loc='upper right', fontsize=10, ncol=2)
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)
    
    # Show every 3rd label
    x_labels = ax3.get_xticklabels()
    for i, label in enumerate(x_labels):
        if i % 3 != 0:
            label.set_visible(False)
    
    # 4. Trends
    ax4 = fig.add_subplot(gs[1, 1])
    trend_data = []
    mid_point = len(ranks_df['Period'].unique()) // 2
    periods = sorted(ranks_df['Period'].unique())
    
    for keyword in top_keywords:
        kw_data = ranks_df[ranks_df['Keyword'] == keyword]
        first_half = kw_data[kw_data['Period'].isin(periods[:mid_point])]['Rank'].mean()
        second_half = kw_data[kw_data['Period'].isin(periods[mid_point:])]['Rank'].mean()
        trend = first_half - second_half
        trend_data.append({'Keyword': keyword, 'Trend': trend})
    
    trend_df = pd.DataFrame(trend_data).sort_values('Trend', ascending=False)
    colors_trend = ['green' if x > 0 else 'red' for x in trend_df['Trend']]
    bars = ax4.barh(range(len(trend_df)), trend_df['Trend'], 
                   color=colors_trend, alpha=0.7, height=0.7)
    ax4.set_yticks(range(len(trend_df)))
    ax4.set_yticklabels(trend_df['Keyword'], fontsize=11)
    ax4.set_xlabel('Rank Change', fontsize=12)
    ax4.set_title('Keyword Trends', fontsize=14, pad=15)
    ax4.axvline(0, color='black', linewidth=0.5)
    ax4.grid(axis='x', alpha=0.3)
    
    # 5. Diversity
    ax5 = fig.add_subplot(gs[2, :])
    ax5.bar(range(len(stats_df)), stats_df['Total_Keywords'], 
           color='lightblue', alpha=0.7, width=0.8)
    ax5.set_ylabel('Unique Keywords', fontsize=12)
    ax5.set_xlabel('Period', fontsize=12)
    ax5.set_title('Keyword Diversity Over Time', fontsize=14, pad=15)
    ax5.grid(axis='y', alpha=0.3)
    
    # Show every 3rd label
    ax5.set_xticks(range(0, len(stats_df), 3))
    ax5.set_xticklabels([stats_df.iloc[i]['Period'] for i in range(0, len(stats_df), 3)], 
                       rotation=45, ha='right', fontsize=10)
    
    plt.tight_layout(pad=3.0)
    return fig, ranks_df


def plot_party_ranks_over_time(df, flag_col, party_col='party', method_name='Economic Tweets',
                              period_freq='M', min_tweets=5, date_col='created_at', 
                              figsize=(24, 16)):
    """
    Generate party rank plots with improved spacing
    """
    print(f"\nGenerating party rank plots for: {method_name}")
    
    # Party colors
    party_colors = {
        'Konfederacja': '#DC143C',
        'NL': '#4169E1',
        'PL2050': '#32CD32',
        'PO': '#9370DB',
        'PSL': '#FF8C00',
        'PIS': '#FFD700'
    }
    
    # Prepare data
    df_plot = df.copy()
    df_plot[date_col] = pd.to_datetime(df_plot[date_col], errors='coerce')
    df_plot = df_plot.dropna(subset=[date_col])
    df_plot = df_plot.set_index(date_col).sort_index()
    
    # Filter for economic tweets
    df_econ = df_plot[df_plot[flag_col] == True].copy()
    
    # Get party rankings by period
    ranks_data = []
    for period_start, period_df in df_econ.groupby(pd.Grouper(freq=period_freq)):
        if not period_df.empty:
            period_label = period_start.strftime('%Y-%m')
            party_counts = period_df[party_col].value_counts()
            total_tweets = party_counts.sum()
            party_counts = party_counts[party_counts >= min_tweets]
            
            for rank, (party, count) in enumerate(party_counts.items(), 1):
                market_share = (count / total_tweets * 100) if total_tweets > 0 else 0
                ranks_data.append({
                    'Period': period_label,
                    'Period_Date': period_start,
                    'Party': party,
                    'Rank': rank,
                    'Count': count,
                    'Market_Share': market_share
                })
    
    ranks_df = pd.DataFrame(ranks_data)
    top_parties = ranks_df.groupby('Party')['Count'].sum().nlargest(6).index.tolist()
    
    # Create figure with more spacing
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.5, 1, 1], width_ratios=[2.5, 1], 
                         hspace=0.7, wspace=0.5, left=0.05, right=0.95, top=0.95, bottom=0.1)
    
    # 1. Main rank plot
    ax1 = fig.add_subplot(gs[0, :])
    for party in top_parties:
        party_data = ranks_df[ranks_df['Party'] == party]
        if len(party_data) > 1:
            color = party_colors.get(party, 'gray')
            ax1.plot(party_data['Period'], party_data['Rank'],
                    marker='o', linewidth=3, markersize=6,
                    label=party, color=color, alpha=0.8)
    
    ax1.set_ylim(len(top_parties) + 0.5, 0.5)
    ax1.set_title(f'Party Rankings by Economic Tweet Volume - {method_name}', fontsize=18, pad=25)
    ax1.set_ylabel('Rank (1 = Most Active)', fontsize=14)
    ax1.set_xlabel('Period', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Improve x-axis labels
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=11)
    x_labels = ax1.get_xticklabels()
    for i, label in enumerate(x_labels):
        if i % 3 != 0:
            label.set_visible(False)
    
    # 2. Market share stacked area
    ax2 = fig.add_subplot(gs[1, :])
    market_pivot = ranks_df[ranks_df['Party'].isin(top_parties)].pivot_table(
        index='Period', columns='Party', values='Market_Share', fill_value=0
    )
    
    ax2.stackplot(market_pivot.index,
                  [market_pivot[party] if party in market_pivot.columns else [0]*len(market_pivot) 
                   for party in top_parties],
                  labels=top_parties,
                  colors=[party_colors.get(p, 'gray') for p in top_parties],
                  alpha=0.7)
    
    ax2.set_ylabel('Market Share (%)', fontsize=13)
    ax2.set_xlabel('Period', fontsize=13)
    ax2.set_title('Party Market Share Over Time', fontsize=15, pad=15)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)
    x_labels = ax2.get_xticklabels()
    for i, label in enumerate(x_labels):
        if i % 4 != 0:
            label.set_visible(False)
    
    # 3. Competitive positioning
    ax3 = fig.add_subplot(gs[2, 0])
    party_summary = ranks_df[ranks_df['Party'].isin(top_parties)].groupby('Party').agg({
        'Rank': 'mean',
        'Count': 'sum',
        'Market_Share': 'mean'
    })
    
    for party in party_summary.index:
        x = party_summary.loc[party, 'Rank']
        y = party_summary.loc[party, 'Market_Share']
        size = party_summary.loc[party, 'Count']
        ax3.scatter(x, y, s=size/3, color=party_colors.get(party, 'gray'),
                   alpha=0.7, edgecolors='black', linewidth=1.5)
        ax3.annotate(party, (x, y), xytext=(8, 8), textcoords='offset points', 
                    fontsize=11, weight='bold')
    
    ax3.set_xlabel('Average Rank (lower better)', fontsize=12)
    ax3.set_ylabel('Average Market Share (%)', fontsize=12)
    ax3.set_title('Party Positioning', fontsize=15, pad=15)
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()
    
    # 4. Growth trends with improved label positioning
    ax4 = fig.add_subplot(gs[2, 1])
    periods = sorted(ranks_df['Period'].unique())
    available_years = set(p.split('-')[0] for p in periods)
    
    growth_data = []
    if '2024' in available_years and '2023' in available_years:
        year_2023_periods = [p for p in periods if p.startswith('2023')]
        year_2024_periods = [p for p in periods if p.startswith('2024') and int(p.split('-')[1]) <= 10]
        year_2023_comparable = [p for p in year_2023_periods if int(p.split('-')[1]) <= 10]
        
        for party in top_parties:
            party_data = ranks_df[ranks_df['Party'] == party]
            activity_2023 = party_data[party_data['Period'].isin(year_2023_comparable)]['Count'].sum()
            activity_2024 = party_data[party_data['Period'].isin(year_2024_periods)]['Count'].sum()
            
            if activity_2023 > 0:
                growth = ((activity_2024 - activity_2023) / activity_2023) * 100
            else:
                growth = 100 if activity_2024 > 0 else 0
            growth_data.append({'Party': party, 'Growth': growth})
    
    growth_df = pd.DataFrame(growth_data).sort_values('Growth')
    colors_growth = [party_colors.get(p, 'gray') for p in growth_df['Party']]
    bars = ax4.barh(range(len(growth_df)), growth_df['Growth'], 
                   color=colors_growth, alpha=0.7, height=0.7)
    
    ax4.set_yticks(range(len(growth_df)))
    ax4.set_yticklabels(growth_df['Party'], fontsize=11)
    ax4.set_xlabel('Growth Rate (%)', fontsize=12)
    ax4.set_title('Activity Growth\n(2024 vs 2023)', fontsize=14, pad=15)
    ax4.axvline(0, color='black', linewidth=0.8)
    ax4.grid(axis='x', alpha=0.3)
    
    # Position percentage labels at the end of bars with proper spacing
    for i, (bar, growth) in enumerate(zip(bars, growth_df['Growth'])):
        bar_width = bar.get_width()
        if bar_width >= 0:
            label_x = bar_width + 1
            ha = 'left'
        else:
            label_x = bar_width - 1
            ha = 'right'
        ax4.text(label_x, bar.get_y() + bar.get_height()/2,
                f'{growth:.0f}%', ha=ha, va='center', fontsize=11, weight='bold')
    
    # Adjust x-axis limits to accommodate labels
    x_min, x_max = ax4.get_xlim()
    ax4.set_xlim(x_min - 5, x_max + 5)
    
    plt.tight_layout(pad=3.0)
    return fig, ranks_df


def plot_top_contributors_ranks(df, flag_col, user_col='username', party_col='party',
                               method_name='Economic Discourse', top_n=10,
                               period_freq='2M', min_tweets=3, date_col='created_at',
                               figsize=(24, 16)):
    """
    Generate contributor rank plots with improved spacing
    """
    print(f"\nGenerating contributor rank plots for: {method_name}")
    
    df_plot = df.copy()
    party_colors = {
        'Konfederacja': '#DC143C', 'NL': '#4169E1', 'PL2050': '#32CD32',
        'PO': '#9370DB', 'PSL': '#FF8C00', 'PIS': '#FFD700'
    }
    
    # Prepare data
    df_plot[date_col] = pd.to_datetime(df_plot[date_col], errors='coerce')
    df_plot = df_plot.dropna(subset=[date_col])
    df_plot = df_plot.set_index(date_col).sort_index()
    df_econ = df_plot[df_plot[flag_col] == True].copy()
    
    # Get top contributors
    overall_contributors = df_econ[user_col].value_counts().head(top_n).index.tolist()
    
    # Get rankings by 2-month periods
    ranks_data = []
    for period_start, period_df in df_econ.groupby(pd.Grouper(freq=period_freq)):
        if not period_df.empty:
            period_label = period_start.strftime('%Y-%m')
            period_counts = period_df[user_col].value_counts()
            period_counts = period_counts[period_counts >= min_tweets]
            
            for rank, (user, count) in enumerate(period_counts.items(), 1):
                if user in overall_contributors:
                    user_parties = df_econ[df_econ[user_col] == user][party_col].dropna()
                    user_party = user_parties.mode()[0] if len(user_parties) > 0 else 'Unknown'
                    ranks_data.append({
                        'Period': period_label, 'User': user, 'Party': user_party,
                        'Rank': rank, 'Count': count
                    })
    
    if not ranks_data:
        print("No sufficient data")
        return None, None
    
    ranks_df = pd.DataFrame(ranks_data)
    
    # Create plots with more spacing
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    plt.subplots_adjust(hspace=0.5, wspace=0.5, left=0.05, right=0.95, top=0.95, bottom=0.1)
    
    # 1. Top 5 rank evolution
    top_5_users = overall_contributors[:5]
    for user in top_5_users:
        user_data = ranks_df[ranks_df['User'] == user]
        if not user_data.empty:
            user_party = user_data['Party'].iloc[0]
            color = party_colors.get(user_party, 'gray')
            clean_name = user.replace('@', '').split('_')[0][:8]
            ax1.plot(user_data['Period'], user_data['Rank'], 
                    marker='o', linewidth=2.5, markersize=6,
                    label=f"{clean_name} ({user_party})", color=color, alpha=0.8)
    
    ax1.set_ylim(16, 0)
    ax1.set_yticks(range(1, 16, 2))
    ax1.set_title('Top 5 Contributors (2-Month Periods)', fontsize=16, pad=20)
    ax1.set_ylabel('Rank', fontsize=13)
    ax1.set_xlabel('Period', fontsize=13)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=11)
    
    # 2. Consistency vs performance
    consistency_data = []
    for user in overall_contributors:
        user_data = ranks_df[ranks_df['User'] == user]
        periods_active = len(user_data)
        total_periods = len(ranks_df['Period'].unique())
        consistency = (periods_active / total_periods) * 100 if total_periods > 0 else 0
        avg_rank = user_data['Rank'].mean() if len(user_data) > 0 else 0
        total_tweets = user_data['Count'].sum()
        party = user_data['Party'].iloc[0] if not user_data.empty else 'Unknown'
        consistency_data.append({
            'User': user, 'Party': party, 'Consistency': consistency,
            'Avg_Rank': avg_rank, 'Total_Tweets': total_tweets
        })
    
    consist_df = pd.DataFrame(consistency_data).sort_values('Total_Tweets', ascending=False)
    
    for idx, row in consist_df.head(8).iterrows():
        color = party_colors.get(row['Party'], 'gray')
        ax2.scatter(row['Consistency'], row['Avg_Rank'], 
                   s=row['Total_Tweets']*1.5, color=color, alpha=0.6,
                   edgecolors='black', linewidth=1)
        if idx < 4:
            username_short = row['User'].replace('@', '').split('_')[0][:8]
            ax2.annotate(username_short, (row['Consistency'], row['Avg_Rank']),
                       xytext=(8, 8), textcoords='offset points', fontsize=10)
    
    ax2.set_xlabel('Activity Consistency (%)', fontsize=13)
    ax2.set_ylabel('Average Rank', fontsize=13)
    ax2.set_title('Consistency vs Performance', fontsize=16, pad=20)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)
    
    # 3. Party activity
    party_contrib = ranks_df.groupby(['Period', 'Party'])['Count'].sum().reset_index()
    party_pivot = party_contrib.pivot(index='Period', columns='Party', values='Count').fillna(0)
    
    if len(party_pivot.columns) > 0:
        top_parties_by_contrib = party_pivot.sum().nlargest(4).index
        for party in top_parties_by_contrib:
            if party in party_pivot.columns:
                color = party_colors.get(party, 'gray')
                ax3.plot(party_pivot.index, party_pivot[party],
                        marker='s', linewidth=2.5, markersize=5, 
                        label=party, color=color, alpha=0.8)
    
    ax3.set_title('Party Activity (2-Month Periods)', fontsize=16, pad=20)
    ax3.set_ylabel('Total Economic Tweets', fontsize=13)
    ax3.set_xlabel('Period', fontsize=13)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=11)
    
    # 4. Summary
    ax4.axis('off')
    summary_text = "TOP CONTRIBUTORS SUMMARY\n" + "="*30 + "\n\n"
    
    if len(consist_df) > 0:
        most_active = consist_df.iloc[0]
        active_name = most_active['User'].replace('@', '').split('_')[0][:12]
        summary_text += f"Most Active:\n{active_name}\nTweets: {most_active['Total_Tweets']}\n\n"
        
        most_consistent = consist_df.nlargest(1, 'Consistency').iloc[0]
        consistent_name = most_consistent['User'].replace('@', '').split('_')[0][:12]
        summary_text += f"Most Consistent:\n{consistent_name}\nActive: {most_consistent['Consistency']:.0f}%\n\n"
        
        party_totals = consist_df.groupby('Party')['Total_Tweets'].sum().nlargest(3)
        summary_text += "Top Parties:\n"
        for party, total in party_totals.items():
            summary_text += f"{party}: {total}\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))
    
    plt.tight_layout(pad=3.0)
    return fig, ranks_df


def plot_rank_momentum_analysis(df, token_col, flag_col, method_name='Momentum Analysis',
                               period_freq='M', date_col='created_at', figsize=(20, 10)):
    """
    Generate momentum analysis showing rising and falling keywords
    Note: The numbers (#17, #19, etc.) show the CURRENT ranking of each keyword
    """

    # Filter economic tweets
    df_econ = df[df[flag_col] == True].copy()
    df_econ[date_col] = pd.to_datetime(df_econ[date_col], errors='coerce')
    df_econ = df_econ.dropna(subset=[date_col])
    
    # Calculate keyword rankings over time
    keyword_momentum = []
    periods = []
    
    for period_start, period_df in df_econ.groupby(pd.Grouper(key=date_col, freq=period_freq)):
        if not period_df.empty:
            period_label = period_start.strftime('%Y-%m')
            periods.append(period_label)
            
            period_keywords = []
            for tokens in period_df[token_col].dropna():
                if isinstance(tokens, list):
                    period_keywords.extend(tokens)
            
            keyword_counts = Counter(period_keywords)
            
            for rank, (keyword, count) in enumerate(keyword_counts.most_common(50), 1):
                keyword_momentum.append({
                    'Period': period_label,
                    'Keyword': keyword,
                    'Rank': rank,
                    'Count': count
                })
    
    momentum_df = pd.DataFrame(keyword_momentum)
    
    if len(periods) >= 4:
        recent_periods = periods[-4:]
        momentum_scores = []
        
        for keyword in momentum_df['Keyword'].unique():
            kw_data = momentum_df[momentum_df['Keyword'] == keyword]
            recent_data = kw_data[kw_data['Period'].isin(recent_periods)]
            
            if len(recent_data) >= 2:
                early_rank = recent_data.iloc[0]['Rank']
                late_rank = recent_data.iloc[-1]['Rank']
                momentum = early_rank - late_rank
                
                momentum_scores.append({
                    'Keyword': keyword,
                    'Momentum': momentum,
                    'Current_Rank': late_rank,
                    'Previous_Rank': early_rank
                })
        
        momentum_results = pd.DataFrame(momentum_scores)
        
        # Create visualization with more spacing
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        plt.subplots_adjust(wspace=0.4)
        
        # Rising keywords
        top_rising = momentum_results.nlargest(10, 'Momentum')
        bars1 = ax1.barh(top_rising['Keyword'], top_rising['Momentum'], color='green', alpha=0.7)
        ax1.set_xlabel('Rank Improvement (positions gained)', fontsize=13)
        ax1.set_title('Fastest Rising Keywords', fontsize=16, pad=20)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add current rank labels with more spacing
        for bar, (idx, row) in zip(bars1, top_rising.iterrows()):
            width = bar.get_width()
            ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'#{int(row["Current_Rank"])}', va='center', fontsize=11, weight='bold') 
        
        # Adjust x-axis to accommodate labels
        x_max = ax1.get_xlim()[1]
        ax1.set_xlim(0, x_max + 2)
        
        # Falling keywords
        top_falling = momentum_results.nsmallest(10, 'Momentum')
        top_falling['Momentum_Abs'] = top_falling['Momentum'].abs()
        bars2 = ax2.barh(top_falling['Keyword'], top_falling['Momentum_Abs'], color='red', alpha=0.7)
        ax2.set_xlabel('Rank Decline (positions lost)', fontsize=13)
        ax2.set_title('Fastest Falling Keywords', fontsize=16, pad=20)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add current rank labels with more spacing
        for bar, (idx, row) in zip(bars2, top_falling.iterrows()):
            width = bar.get_width()
            ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                    f'#{int(row["Current_Rank"])}', va='center', fontsize=11, weight='bold')
        
        # Adjust x-axis to accommodate labels
        x_max = ax2.get_xlim()[1]
        ax2.set_xlim(0, x_max + 2)
        
        plt.tight_layout(pad=3.0)
        return fig, momentum_results
    
    else:
        print("Not enough time periods for momentum analysis")
        return None, None


def generate_all_rank_plots(df, method='Final_Refined', period_freq='M', date_col='created_at',
                           save_pdf=True, pdf_filename='rank_analysis_report.pdf'):
    """
    Generate all rank plots and optionally save to PDF
    """
    print(f"\n{'='*60}")
    print(f"GENERATING RANK ANALYSIS FOR {method.upper()}")
    print(f"{'='*60}")
    
    # Determine columns
    if method == 'Final_Refined':
        flag_col = 'has_econ_term_FILTERED'
        token_col = 'matched_keywords_FILTERED'
    elif method == 'Union':
        flag_col = 'union_flag'
        token_col = 'union_tokens'
    elif method == 'Gateway':
        flag_col = 'gateway_flag'
        token_col = 'gateway_tokens'
    else:
        print(f"Unknown method: {method}")
        return {}
    
    results = {}
    figures = []
    
    # 1. Keyword rankings
    if token_col in df.columns:
        print(f"\n[1/4] Generating keyword rankings...")
        fig, data = plot_keyword_ranks_over_time(
            df, token_col, method, period_freq=period_freq, date_col=date_col)
        results['keywords'] = data
        figures.append(('Keyword Rankings', fig))
    
    # 2. Party rankings
    if flag_col in df.columns and 'party' in df.columns:
        print(f"\n[2/4] Generating party rankings...")
        fig, data = plot_party_ranks_over_time(
            df, flag_col, method_name=method, period_freq=period_freq, date_col=date_col)
        results['parties'] = data
        figures.append(('Party Rankings', fig))
    
    # 3. Contributors
    if flag_col in df.columns and 'username' in df.columns:
        print(f"\n[3/4] Generating contributor rankings...")
        fig, data = plot_top_contributors_ranks(
            df, flag_col, method_name=method, period_freq='2M', date_col=date_col)
        if fig is not None:
            results['contributors'] = data
            figures.append(('Contributor Rankings', fig))
    
    # 4. Momentum
    if token_col in df.columns and flag_col in df.columns:
        print(f"\n[4/4] Generating momentum analysis...")
        fig, data = plot_rank_momentum_analysis(
            df, token_col, flag_col, method_name=method,
            period_freq=period_freq, date_col=date_col)
        if fig is not None:
            results['momentum'] = data
            figures.append(('Momentum Analysis', fig))
    
    # Save to PDF if requested
    if save_pdf and figures:
        print(f"\nSaving all plots to PDF: {pdf_filename}")
        with PdfPages(pdf_filename) as pdf:
            for title, fig in figures:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Add metadata
            d = pdf.infodict()
            d['Title'] = f'Rank Analysis Report - {method}'
            d['Author'] = 'Rank Analysis System'
            d['Subject'] = 'Economic Discourse Analysis'
            d['Keywords'] = f'Rankings, {method}, Economic Analysis'
            d['CreationDate'] = pd.Timestamp.now()
        
        print(f"PDF saved successfully: {pdf_filename}")
    
    print(f"\n{'='*60}")
    print("RANK ANALYSIS COMPLETE")
    print(f"{'='*60}")
    return results


def print_module_info():
    """
    Display available functions
    """
    print("="*60)
    print("ENHANCED RANK PLOTS MODULE - Available Functions:")
    print("="*60)
    print("plot_keyword_ranks_over_time() - Keyword rankings with trends")
    print("plot_party_ranks_over_time() - Party rankings with market share")
    print("plot_top_contributors_ranks() - Contributor analysis (2M periods)")
    print("plot_rank_momentum_analysis() - Rising/falling keywords")
    print("generate_all_rank_plots() - Run all analyses with PDF export")
    print("="*60)