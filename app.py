import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import PIL
from PIL import Image, ImageEnhance
import seaborn as sns
import plotly.graph_objs as go


document_id = '1CWOu-S9NtRn49wXP_7as2sknb82Uvj5YUWy-Ot4g3wE'
def fetch_data(sheet_name):
    url = f'https://docs.google.com/spreadsheets/d/{document_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    return pd.read_csv(url, index_col=0)

data = fetch_data('matches')


def main():
    st.title("FIFA U20 Women's World Cup ")

    # Step 5: Select two teams for analysis
    unique_home_teams = data['home_team_name'].unique()
    unique_away_teams = data['away_team_name'].unique()

    # Remove home teams already present in the away teams list
    unique_away_teams = [team for team in unique_away_teams if team not in unique_home_teams]

    # Concatenate unique home and away team names
    all_unique_teams = list(unique_home_teams) + list(unique_away_teams)

    # Use the multiselect widget with the combined unique team names
    selected_teams = st.multiselect("Select Teams", all_unique_teams)

    if len(selected_teams) != 2:
        st.warning("Please select exactly two teams.")
        return

    team1, team2 = selected_teams

    # Step 6: Filter the data based on the selected teams
    team_data = data[(data['home_team_name'].isin(selected_teams)) & (data['away_team_name'].isin(selected_teams))]

    # Step 7: Display the raw data for the selected teams
    if st.checkbox("Show Raw Data"):
        st.dataframe(team_data[['home_team_name', 'away_team_name', 'home_team_score', 'away_team_score']])

    # Step 8: Display the head-to-head comparison
    st.subheader("U20-WWC Head-to-Head Comparison")
    head_to_head_plot(team_data, team1, team2)

    # Step 9: Display total goals scored by each team
    st.subheader("Total Goals Scored ")
    total_goals_plot(data, team1, team2)

    st.subheader(f"Head-to-Head Stats for {team1} and {team2}")
    (average_goals_scored_team1, average_goals_conceded_team1), (average_goals_scored_team2, average_goals_conceded_team2) = calculate_head_to_head_totals(data, team1, team2)

    st.header("Average Goals Analysis")
    st.write(f"**{team1}**:")
    st.write(f"Average Goals Scored: {average_goals_scored_team1:.2f}")
    st.write(f"Average Goals Conceded: {average_goals_conceded_team1:.2f}")

    st.write(f"**{team2}**:")
    st.write(f"Average Goals Scored: {average_goals_scored_team2:.2f}")
    st.write(f"Average Goals Conceded: {average_goals_conceded_team2:.2f}")


  
    st.title("Goals Distribution by stage")

    goals_distribution = team_data.groupby(['home_team_name', 'stage'])[['home_team_score', 'away_team_score']].sum().reset_index()

    # Sum the total goals (home_goal + away_goal)
    goals_distribution['total_goals'] = goals_distribution['home_team_score'].astype(int) + goals_distribution['away_team_score'].astype(int)
    # goals_distribution['leg'] = goals_distribution.groupby('stage').cumcount() + 1
    # goals_distribution['leg'] = goals_distribution['leg'].replace({1: 'First Leg', 2: 'Second Leg'})

    st.write("Number of goals per stage:")
    # Group by stage, sum the total_goals, convert to int, and sort by total_goals
    sorted_goals_distribution = goals_distribution[['stage', 'total_goals']].groupby('stage').sum().astype(int).sort_values(by='total_goals', ascending=False)

    # Display the sorted table
    st.table(sorted_goals_distribution)

    # Filter to include only the last 10 stages
    last_10_stages = goals_distribution['stage'].unique()[-11:]

    # Plot horizontal bar chart using top 10 sorted_goals_distribution
    top_10_goals_distribution = sorted_goals_distribution.head(10)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_10_goals_distribution['total_goals'], y=top_10_goals_distribution.index, palette=sns.color_palette("Reds")[::-2])
    plt.title("Stages by Total Goals")
    plt.xlabel("Total Goals")
    plt.ylabel("Season")
    plt.yticks(rotation=0, fontsize=10)  # Ensure y-axis labels are readable

    # Add the first logo
    logo1 = Image.open("logos/logo.png")
    logo1 = logo1.resize((400, 400), Image.Resampling.LANCZOS)
    opacity = 0.2
    enhancer = ImageEnhance.Brightness(logo1)
    logo1 = enhancer.enhance(opacity)


    fig = plt.gcf()
    ax = plt.gca()

    # Calculate position for the first logo (top center)
    center_x1 = (fig.bbox.x0 + fig.bbox.width / 2) + (logo1.size[0] / 2)
    center_y1 = fig.bbox.y0 - (logo1.size[1] / 2) + (logo1.size[0] * 1.85 )

    

    # Place the logos
    fig.figimage(logo1, xo=center_x1, yo=center_y1, origin='upper')
    # fig.figimage(logo2, xo=center_x2, yo=center_y2, origin='upper')

    # Render the chart
    st.pyplot(fig)


        


    st.title("Goals Distribution by Stage per Team")
    goals_distribution_per_team = team_data.groupby(['stage', 'home_team_name'])[['home_team_score']].sum().reset_index()
    goals_distribution_per_team = goals_distribution_per_team.rename(columns={'home_team_name': 'team', 'home_team_score': 'goals_scored'})
    
    # Sum away goals per team and merge with home goals
    away_goals_per_team = team_data.groupby(['stage', 'away_team_name'])[['away_team_score']].sum().reset_index()
    away_goals_per_team = away_goals_per_team.rename(columns={'away_team_name': 'team', 'away_team_score': 'goals_scored'})
    
    goals_distribution_per_team = pd.concat([goals_distribution_per_team, away_goals_per_team], axis=0)
    goals_distribution_per_team = goals_distribution_per_team.groupby(['stage', 'team'])[['goals_scored']].sum().reset_index()

    st.write("Number of goals per stage per team:")
    st.table(goals_distribution_per_team.pivot(index='stage', columns='team', values='goals_scored').fillna(0).astype(int))

    team_palette = ["red", "#0078D4"]

    last_10_stages = goals_distribution_per_team['stage'].unique()[-10:]
    goals_distribution_per_team_last_10 = goals_distribution_per_team[goals_distribution_per_team['stage'].isin(last_10_stages)]

# Check the length of the filtered DataFrame
    if len(goals_distribution_per_team_last_10) >= 10:
        goals_distribution_per_team_last_10 = goals_distribution_per_team_last_10[1:]
    else:
        goals_distribution_per_team_last_10 = goals_distribution_per_team[goals_distribution_per_team['stage'].isin(last_10_stages)]
   
    plt.figure(figsize=(10, 6))
    sns.barplot(x='stage', y='goals_scored', hue='team', data=goals_distribution_per_team_last_10, ci=None, palette=team_palette)
    plt.title("Goals Distribution by Stage per Team ")
    plt.xlabel("Season")
    plt.ylabel("Total Goals")
    plt.xticks(rotation=45, ha='right', fontsize=10)

    logo1 = Image.open("logos/logo.png")
    logo1 = logo1.resize((400, 400), Image.Resampling.LANCZOS)
    opacity = 0.2
    enhancer = ImageEnhance.Brightness(logo1)
    logo1 = enhancer.enhance(opacity)


    fig2 = plt.gcf()
    ax2 = plt.gca()

    # Calculate position for the first logo (top center)
    center_x1 = (fig2.bbox.x0 + fig2.bbox.width / 2) + (logo1.size[0] / 2)
    center_y1 = fig2.bbox.y0 - (logo1.size[1] / 2) + (logo1.size[0] * 1.85 )

    

    # Place the logos
    fig2.figimage(logo1, xo=center_x1, yo=center_y1, origin='upper')
    # fig2.figimage(logo2, xo=center_x2, yo=center_y2, origin='upper')

    # Render the chart
    st.pyplot(fig2)

  

    team1_games = filter_team_games(data, team1)
    team2_games = filter_team_games(data, team2)

    last_5_team1_games = get_last_n_games(team1_games)
    last_5_team2_games = get_last_n_games(team2_games)

    last_5_team1_games['result'] = last_5_team1_games.apply(determine_result, axis=1, team=team1)
    last_5_team2_games['result'] = last_5_team2_games.apply(determine_result, axis=1, team=team2)

    team1_results_str = generate_result_string(last_5_team1_games)
    team2_results_str = generate_result_string(last_5_team2_games)


    st.subheader(f"Last 5 U20-WWC games involving {team1}: {team1_results_str}")
    st.write(display_last_n_games(last_5_team1_games))

    st.subheader(f"Last 5 U20-WWC games involving {team2}: {team2_results_str}")
    st.write(display_last_n_games(last_5_team2_games))


def generate_result_string(last_n_team_games):
    return ''.join(last_n_team_games['result'].values)
# Step 10: Data Visualization functions
def filter_team_games(data, team):
    return data[(data['home_team_name'] == team) | (data['away_team_name'] == team)]

def get_last_n_games(team_games, n=5):
    # Filter out games that haven't been played yet (missing goals)
    completed_games = team_games.dropna(subset=['home_team_score', 'away_team_score'])
    return completed_games.tail(n)

def determine_result(row, team):
    if team == row['home_team_name']:
        if row['home_team_score'] > row['away_team_score']:
            return 'W'
        elif row['home_team_score'] < row['away_team_score']:
            return 'L'
        else:
            return 'D'
    elif team == row['away_team_name']:
        if row['away_team_score'] > row['home_team_score']:
            return 'W'
        elif row['away_team_score'] < row['home_team_score']:
            return 'L'
        else:
            return 'D'
    else:
        return None

def display_last_n_games(last_n_team_games):
    columns_to_display = ['home_team_name', 'away_team_name', 'home_team_score', 'away_team_score', 'result']
    return last_n_team_games[columns_to_display]

def head_to_head_plot(data, team1, team2):
    home_wins_team1 = data[(data['home_team_name'] == team1) & (data['home_team_score'] > data['away_team_score'])]
    away_wins_team1 = data[(data['away_team_name'] == team1) & (data['away_team_score'] > data['home_team_score'])]
    draws_team1 = data[((data['home_team_name'] == team1) | (data['away_team_name'] == team1)) & (data['home_team_score'] == data['away_team_score'])]

    home_wins_team2 = data[(data['home_team_name'] == team2) & (data['home_team_score'] > data['away_team_score'])]
    away_wins_team2 = data[(data['away_team_name'] == team2) & (data['away_team_score'] > data['home_team_score'])]
    draws_team2 = data[((data['home_team_name'] == team2) | (data['away_team_name'] == team2)) & (data['home_team_score'] == data['away_team_score'])]

    x_labels = ['Wins', 'Draws']
    team1_values = [len(home_wins_team1) + len(away_wins_team1), len(draws_team1)]
    team2_values = [len(home_wins_team2) + len(away_wins_team2), len(draws_team2)]

    fig, ax = plt.subplots()
    bar_width = 0.15
    bar_positions = list(range(len(x_labels)))

    ax.bar(bar_positions, team1_values, bar_width, label=team1, color=['red'])
    ax.bar([pos + bar_width for pos in bar_positions], team2_values, bar_width, label=team2)

    for i, value in enumerate(team1_values):
        ax.annotate(str(value), xy=(bar_positions[i], value), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom')

    for i, value in enumerate(team2_values):
        ax.annotate(str(value), xy=(bar_positions[i] + bar_width, value), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom')

    team1_matches = data[(data['home_team_name'] == team1) | (data['away_team_name'] == team1)]
    team2_matches = data[(data['home_team_name'] == team2) | (data['away_team_name'] == team2)]
    #total_matches = round((int(len(team1_matches)) + int(len(team2_matches))) / 2)

    total_matches = len(home_wins_team1) + len(away_wins_team1) + len(home_wins_team2) + len(away_wins_team2) + len(draws_team1)

  

    logo1 = Image.open("logos/logo.png")
    logo1 = logo1.resize((400, 400), Image.Resampling.LANCZOS)
    opacity = 0.2
    enhancer = ImageEnhance.Brightness(logo1)
    logo1 = enhancer.enhance(opacity)

    fig = plt.gcf()
    ax = plt.gca()

    # Calculate position for the first logo (top center)
    center_x1 = (fig.bbox.x0 + fig.bbox.width / 2) + (logo1.size[0] / 3)
    center_y1 = fig.bbox.y0 - (logo1.size[1] / 2) + (logo1.size[0]  )

    # Place the logos
    fig.figimage(logo1, xo=center_x1, yo=center_y1, origin='upper')
    # fig.figimage(logo2, xo=center_x2, yo=center_y2, origin='upper')


    ax.set_xticks([pos + bar_width / 2 for pos in bar_positions])
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Matches')
    ax.set_title(f'{team1} vs. {team2} U20-WWC Comparison')
    ax.legend()
    st.write(f"Total U20-WWC matches played by {team1} and {team2}: {total_matches}")
    st.pyplot(fig)

def total_goals_plot(data, team1, team2):
    team1_goals = data[((data['home_team_name'] == team1) & (data['away_team_name'] == team2)) | ((data['home_team_name'] == team2) & (data['away_team_name'] == team1))]
    team2_goals = data[((data['home_team_name'] == team2) & (data['away_team_name'] == team1)) | ((data['home_team_name'] == team1) & (data['away_team_name'] == team2))]

    team1_home_data = team1_goals[team1_goals['home_team_name'] == team1]
    team1_away_data = team1_goals[team1_goals['away_team_name'] == team1]
    team1_score = round(team1_home_data['home_team_score'].sum() + team1_away_data['away_team_score'].sum())

    team2_home_data = team2_goals[team2_goals['home_team_name'] == team2]
    team2_away_data = team2_goals[team2_goals['away_team_name'] == team2]
    team2_score = round(team2_home_data['home_team_score'].sum() + team2_away_data['away_team_score'].sum())

    st.write(team1 + " total goals against " + team2 + ": " + str(team1_score))
    st.write(team2 + " total goals against " + team1 + ": " + str(team2_score))

    x_labels = [team1, team2]
    y_values = [team1_score, team2_score]

    bar_width = 0.15
    bar_positions = list(range(len(x_labels)))

    fig, ax = plt.subplots(figsize=(4, 3))

    logo1 = Image.open("logos/logo.png")
    logo1 = logo1.resize((300, 300), Image.Resampling.LANCZOS)
    opacity = 0.2
    enhancer = ImageEnhance.Brightness(logo1)
    logo1 = enhancer.enhance(opacity)



    fig = plt.gcf()
    ax = plt.gca()

    # Calculate position for the first logo (top center)
    center_x1 = (fig.bbox.x0 + fig.bbox.width / 2) + (logo1.size[0] / 3.5)
    center_y1 = fig.bbox.y0 - (logo1.size[1] / 2) + (logo1.size[0] / 1.2)


    # Place the logos
    fig.figimage(logo1, xo=center_x1, yo=center_y1, origin='upper')
    # fig.figimage(logo2, xo=center_x2, yo=center_y2, origin='upper')

    # Place the logo in the middle
    # ax.figure.figimage(logo, xo=center_x, yo=center_y, origin='upper')
    ax.bar(x_labels, y_values, width=bar_width, color=['#0078D4', 'red'])
    # for i, value in enumerate(y_values):
    #     # ax.text(i, value + 1, value, ha='center')
    #     pass

    ax.set_ylabel('Total Goals Scored')
    st.pyplot(fig)

def calculate_head_to_head_totals(team_data, team1, team2):
    # Filter data for matches between team1 and team2
    head_to_head_data = team_data[((team_data['home_team_name'] == team1) & (team_data['away_team_name'] == team2)) |
                                  ((team_data['home_team_name'] == team2) & (team_data['away_team_name'] == team1))]

    # Calculate total goals and matches for team1
    total_goals_team1 = head_to_head_data[(head_to_head_data['home_team_name'] == team1)]['home_team_score'].sum() + \
                        head_to_head_data[(head_to_head_data['away_team_name'] == team1)]['away_team_score'].sum()
    total_goals_conceded_team1 = head_to_head_data[(head_to_head_data['home_team_name'] == team1)]['away_team_score'].sum() + \
                                 head_to_head_data[(head_to_head_data['away_team_name'] == team1)]['home_team_score'].sum()
    total_matches_team1 = len(head_to_head_data[(head_to_head_data['home_team_name'] == team1) | (head_to_head_data['away_team_name'] == team1)])

    # Calculate total goals and matches for team2
    total_goals_team2 = head_to_head_data[(head_to_head_data['home_team_name'] == team2)]['home_team_score'].sum() + \
                        head_to_head_data[(head_to_head_data['away_team_name'] == team2)]['away_team_score'].sum()
    total_goals_conceded_team2 = head_to_head_data[(head_to_head_data['home_team_name'] == team2)]['away_team_score'].sum() + \
                                 head_to_head_data[(head_to_head_data['away_team_name'] == team2)]['home_team_score'].sum()
    total_matches_team2 = len(head_to_head_data[(head_to_head_data['home_team_name'] == team2) | (head_to_head_data['away_team_name'] == team2)])

    # Create a DataFrame to display the totals
    totals_df = pd.DataFrame({
        'Metric': ['Total Goals', 'Total Goals Conceded', 'Total Matches'],
        team1: [int(total_goals_team1), int(total_goals_conceded_team1), total_matches_team1],
        team2: [int(total_goals_team2), int(total_goals_conceded_team2), total_matches_team2]
    })

    st.table(totals_df)

    # Calculate average goals scored and conceded
    if total_matches_team1 > 0:
        average_goals_scored_team1 = total_goals_team1 / total_matches_team1
        average_goals_conceded_team1 = total_goals_conceded_team1 / total_matches_team1
    else:
        average_goals_scored_team1 = 0
        average_goals_conceded_team1 = 0

    if total_matches_team2 > 0:
        average_goals_scored_team2 = total_goals_team2 / total_matches_team2
        average_goals_conceded_team2 = total_goals_conceded_team2 / total_matches_team2
    else:
        average_goals_scored_team2 = 0
        average_goals_conceded_team2 = 0

    return (average_goals_scored_team1, average_goals_conceded_team1), (average_goals_scored_team2, average_goals_conceded_team2)

def create_plot(df, title):
    stats_column = 'stats'
    team1_column = df.iloc[0, 1]  # Home team
    team2_column = df.iloc[1, 1]  # Away team

    # Create a new DataFrame for plotting
    plot_df = pd.DataFrame({
        stats_column: df.columns[2:],  # Extract stats from column names starting from the 3rd column
        team1_column: df.iloc[0, 2:],  # Extract data for Home Team
        team2_column: df.iloc[1, 2:]   # Extract data for Away Team
    })

    
    trace1 = go.Bar(
        y=plot_df[stats_column],
        x=-plot_df[team1_column],  # Make x-values negative for one team
        name=team1_column,
        orientation='h',
        marker=dict(color='blue'),
        text=[f"<b>{x}</b>" for x in plot_df[team1_column]],  # Make text bold
        textposition='outside',  # Position labels outside the bars
        textfont=dict(color='black'),  # Set text color to black
        hoverinfo='x+text'  # Hover information
    )

    trace2 = go.Bar(
        y=plot_df[stats_column],
        x=plot_df[team2_column],
        name=team2_column,
        orientation='h',
        marker=dict(color='red'),
        text=[f"<b>{x}</b>" for x in plot_df[team2_column]],  # Make text bold
        textposition='outside',  # Position labels outside the bars
        textfont=dict(color='black'),  # Set text color to black
        hoverinfo='x+text'  # Hover information 
    )


    

    layout = go.Layout(
        title=title,
        barmode='overlay',
        bargap=0.1,
        bargroupgap=0,
        xaxis=dict(
            title='Values',
            showgrid=False,  # Hide x-axis grid lines
            zeroline=True,
            showline=True,
            showticklabels=False  # Hide x-axis ticks
        ),
        yaxis=dict(
            title='Stats',
            showgrid=False,  # Hide y-axis grid lines
            showline=True,
            showticklabels=True,
            tickfont=dict(color='black'),  # Set stats values color to black
            categoryorder='array',  # Order by the values in the DataFrame
            categoryarray=list(plot_df[stats_column])[::-1]  # Use the order from the DataFrame and reverse it
        )
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # Adding central line for visual reference
    fig.add_shape(type="line",
                  x0=0, y0=-0.5, x1=0, y1=len(plot_df[stats_column])-0.5,
                  line=dict(color="black", width=2))

    return fig


df2 = fetch_data('TEAM_STATS')
# df_team = fetch_data('TEAM_STATS')

df2 = df2[['match_id_new','team_name','team_score','goal_attempt','shot_ON','shot_OFF', 'fouls', 'Corners', 'Offsides','yellow card', 'red card','goalkeeper_saves']]
# 'shot_BLOCK', 'shot_accuracy', 'fouls_committed','fouls_drawn','pass_attempt','pass_complete','possession'


match_id = df2['match_id_new'].unique()
item = st.sidebar.selectbox("Select match ID", match_id)

match_id_filtered = df2[df2['match_id_new'] == item]


# st.write(match_id_filtered)

fig1 = create_plot(match_id_filtered, f'{match_id_filtered.columns[1]} vs {match_id_filtered.columns[2]} ')


df = fetch_data('PLAYER_STATS')
df = df.iloc[:, 5:22]



st.sidebar.header("Select Teams for rader chat")
team_options = df['Team'].unique()
team1 = st.sidebar.selectbox("Select Team 1", team_options)
team2 = st.sidebar.selectbox("Select Team 2", team_options)

# Filter the DataFrame based on selected teams
df_team1_filtered = df[df['Team'] == team1]
df_team2_filtered = df[df['Team'] == team2]

# Sidebar for Position Selection
st.sidebar.header("Select Position")
position_options_team1 = df_team1_filtered['POS'].unique()
position_options_team2 = df_team2_filtered['POS'].unique()

# Ensure positions are only selectable if they exist in both teams
common_positions = list(set(position_options_team1) & set(position_options_team2))
selected_position = st.sidebar.selectbox("Select Position", common_positions)

# Further filter the DataFrame based on the selected position
df_position_team1 = df_team1_filtered[df_team1_filtered['POS'] == selected_position]
df_position_team2 = df_team2_filtered[df_team2_filtered['POS'] == selected_position]

# Ensure players are uniquely filtered
players_team1 = df_position_team1['Player'].unique().tolist()
players_team2 = df_position_team2['Player'].unique().tolist()

# Select players from the filtered teams
player1 = st.sidebar.selectbox(f"Select Player from {team1}", players_team1, index=0)
player2 = st.sidebar.selectbox(f"Select Player from {team2}", players_team2, index=1)



if selected_position == 'GK':
    columns_to_use = ['Min','GC', 'Y', 'R', 'FD']
elif selected_position == 'FW':
    columns_to_use = ['Min','GS', 'AS', 'SH', 'OT', 'FD']
elif selected_position == 'MF':
    columns_to_use = ['Min','GS', 'AS', 'SH', 'OT', 'FD', 'FC']
elif selected_position == 'DF':
    columns_to_use = ['Min','GS', 'AS', 'OT', 'FD', 'FC']

# # Filter the DataFrame to include only the relevant columns
df_position_team1 = df_position_team1[columns_to_use]
df_position_team2 = df_position_team2[columns_to_use]

def create_radar_chart(df1, df2, player1, player2):
    metrics = df1.columns[1:].tolist()  # Exclude 'Player' column
    player1_data = df[df['Player'] == player1].iloc[:, 4:10].sum()
    player2_data = df[df['Player'] == player2].iloc[:, 4:10].sum()

    # col1, col2 = st.columns(2)
    
    # with col1:
    #     st.header(f"{player1} Stats")
    #     st.write(player1_data)
    
    # with col2:
    #     st.header(f"{player2} Stats")
    #     st.write(player2_data) 

    player1_data = player1_data.iloc[:].tolist()
    player2_data = player2_data.iloc[:].tolist()





    trace1 = go.Scatterpolar(
        r=player1_data,
        theta=metrics,
        fill='toself',
        name=player1,
        marker=dict(color='blue')
    )

    trace2 = go.Scatterpolar(
        r=player2_data,
        theta=metrics,
        fill='toself',
        name=player2,
        marker=dict(color='red')
    )

    layout = go.Layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]  # Fixed range set to 10
            )
        ),
        showlegend=True
    )


    fig = go.Figure(data=[trace1, trace2], layout=layout)
    return fig






df_stats = df.iloc[:,3:22]
df_stats = df_stats.groupby('Player').sum(numeric_only=True).reset_index()


stats = ['GS', 'GC', 'AS', 'SH', 'OT', 'PK', 'FC', 'FD', 'Y', '2Y', 'R']



# Sidebar for selecting stats
selected_stat = st.sidebar.selectbox("Select Stat", stats)

# Calculate the top 5 players for the selected stat
df_top5 = df_stats.sort_values(by=selected_stat, ascending=False).head(5)

# Plotting the bar chart
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(df_top5['Player'], df_top5[selected_stat], color=['blue', 'green', 'red', 'orange', 'purple'])
ax.set_title(f'Top 5 Players by {selected_stat}')
ax.set_xlabel('Player')
ax.set_ylabel(selected_stat)
ax.set_xticklabels(df_top5['Player'], rotation=45)

# Display the plot in Streamlit



# # Step 11: Run the app
if __name__ == "__main__":
    main()
    st.title("Match Day Team Stats")
    st.plotly_chart(fig1)
    st.title("Player Stats Visualization")
    st.pyplot(fig)
    if team1 != team2:
        if player1 and player2:
            # Create and display radar chart
            # merged_df = pd.concat([df_position_team1, df_position_team2], ignore_index=True)
            # st.dataframe(merged_df, use_container_width=True)
            fig = create_radar_chart(df_position_team1, df_position_team2, player1, player2)
            st.plotly_chart(fig)
    else:
        st.sidebar.write("Please select two different teams.")
