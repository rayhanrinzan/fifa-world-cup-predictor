from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="2026 World Cup Knockout Predictor",
    page_icon="⚽",
    layout="wide",
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ASSETS_DIR = PROJECT_ROOT / "assets"


ACTUAL_ROUND_OF_32 = [
    ("Germany", "Paraguay"),
    ("France", "Sweden"),
    ("South Africa", "Canada"),
    ("Netherlands", "Morocco"),

    ("Portugal", "Croatia"),
    ("Spain", "Austria"),
    ("United States", "Bosnia and Herzegovina"),
    ("Belgium", "Senegal"),

    ("Brazil", "Japan"),
    ("Ivory Coast", "Norway"),
    ("Mexico", "Ecuador"),
    ("England", "DR Congo"),

    ("Argentina", "Cape Verde"),
    ("Australia", "Egypt"),
    ("Switzerland", "Algeria"),
    ("Colombia", "Ghana"),
]


BRACKET_TEAMS = sorted(
    set(team for matchup in ACTUAL_ROUND_OF_32 for team in matchup)
)


@st.cache_data
def load_csv(path):
    if not path.exists():
        return None

    return pd.read_csv(path)


def convert_probability_columns_to_percent(df, probability_cols):
    display_df = df.copy()

    existing_cols = [col for col in probability_cols if col in display_df.columns]

    if len(existing_cols) == 0:
        return display_df

    if display_df[existing_cols].max().max() <= 1:
        display_df[existing_cols] = display_df[existing_cols] * 100

    return display_df


def get_saved_match_prediction(prediction_matrix, team_a, team_b):
    row = prediction_matrix[
        (prediction_matrix["team_a"] == team_a)
        & (prediction_matrix["team_b"] == team_b)
    ]

    if len(row) == 0:
        return None

    return row.iloc[[0]].copy()


st.title("⚽ FIFA World Cup 2026 Knockout Match Predictor")

st.markdown(
    """
    This app predicts 2026 World Cup knockout matchups using a machine learning model trained on historical
    international match results, recent team form, and Elo rating features.

    For stability, the web app loads saved model outputs from CSV files instead of rerunning the model every
    time the app updates.
    """
)


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Match Predictor",
        "Deterministic Bracket",
        "Monte Carlo Results",
        "About the Model",
    ]
)


with tab1:
    st.header("Single Knockout Match Predictor")

    prediction_matrix_path = DATA_PROCESSED / "match_prediction_matrix.csv"
    prediction_matrix = load_csv(prediction_matrix_path)

    if prediction_matrix is None:
        st.warning(
            "Match prediction matrix not found. "
            "Run `python scripts/create_match_prediction_matrix.py` first."
        )
    else:
        col1, col2 = st.columns(2)

        with col1:
            default_team_a = BRACKET_TEAMS.index("France") if "France" in BRACKET_TEAMS else 0
            team_a = st.selectbox("Team A", BRACKET_TEAMS, index=default_team_a)

        with col2:
            default_team_b = BRACKET_TEAMS.index("Brazil") if "Brazil" in BRACKET_TEAMS else 1
            team_b = st.selectbox("Team B", BRACKET_TEAMS, index=default_team_b)

        if team_a == team_b:
            st.warning("Please choose two different teams.")
        else:
            prediction = get_saved_match_prediction(
                prediction_matrix,
                team_a,
                team_b,
            )

            if prediction is None:
                st.error("Prediction not found for this matchup.")
            else:
                row = prediction.iloc[0]

                st.subheader(f"{team_a} vs {team_b}")

                metric_col1, metric_col2, metric_col3 = st.columns(3)

                metric_col1.metric(
                    f"{team_a} Win",
                    f"{row['team_a_win_probability'] * 100:.1f}%",
                )

                metric_col2.metric(
                    "Draw",
                    f"{row['draw_probability'] * 100:.1f}%",
                )

                metric_col3.metric(
                    f"{team_b} Win",
                    f"{row['team_b_win_probability'] * 100:.1f}%",
                )

                st.divider()

                adv_col1, adv_col2, adv_col3 = st.columns(3)

                adv_col1.metric(
                    f"{team_a} Advances",
                    f"{row['team_a_advancement_probability'] * 100:.1f}%",
                )

                adv_col2.metric(
                    f"{team_b} Advances",
                    f"{row['team_b_advancement_probability'] * 100:.1f}%",
                )

                adv_col3.metric(
                    "Predicted Advancing Team",
                    row["predicted_advancing_team"],
                )

                st.subheader("Full Prediction Output")

                probability_cols = [
                    "team_a_win_probability",
                    "draw_probability",
                    "team_b_win_probability",
                    "team_a_advancement_probability",
                    "team_b_advancement_probability",
                ]

                display_prediction = convert_probability_columns_to_percent(
                    prediction,
                    probability_cols,
                )

                st.dataframe(
                    display_prediction,
                    use_container_width=True,
                )


with tab2:
    st.header("Deterministic Knockout Bracket")

    st.markdown(
        """
        The deterministic bracket always advances the team with the higher advancement probability.
        This produces one fixed predicted bracket.
        """
    )

    deterministic_path = DATA_PROCESSED / "actual_deterministic_bracket_results.csv"
    bracket_results = load_csv(deterministic_path)

    if bracket_results is None:
        st.warning(
            "Deterministic bracket results file not found. "
            "Run `python scripts/run_knockout_simulation.py` first."
        )
    else:
        final_rows = bracket_results[bracket_results["round"] == "Final"]

        if len(final_rows) > 0:
            champion = final_rows["predicted_advancing_team"].iloc[0]
            st.success(f"Predicted Champion: {champion}")
        else:
            st.warning("Could not find the final row in the deterministic bracket results.")

        summary_cols = [
            "round",
            "match_number",
            "team_a",
            "team_b",
            "team_a_advancement_probability",
            "team_b_advancement_probability",
            "predicted_advancing_team",
        ]

        existing_summary_cols = [
            col for col in summary_cols
            if col in bracket_results.columns
        ]

        display_df = bracket_results[existing_summary_cols].copy()

        probability_cols = [
            "team_a_advancement_probability",
            "team_b_advancement_probability",
        ]

        display_df = convert_probability_columns_to_percent(
            display_df,
            probability_cols,
        )

        st.dataframe(display_df, use_container_width=True)


with tab3:
    st.header("Monte Carlo Tournament Results")

    st.markdown(
        """
        The Monte Carlo simulation randomly samples winners based on advancement probabilities.
        This estimates each team's chance of reaching each stage of the tournament.
        """
    )

    monte_carlo_path = DATA_PROCESSED / "monte_carlo_knockout_simulation_results.csv"
    monte_carlo_results = load_csv(monte_carlo_path)

    if monte_carlo_results is None:
        st.warning(
            "Monte Carlo results file not found. "
            "Run `python scripts/run_knockout_simulation.py` first."
        )
    else:
        probability_cols = [
            "round_of_16",
            "quarterfinals",
            "semifinals",
            "final",
            "champion",
        ]

        monte_carlo_percent = convert_probability_columns_to_percent(
            monte_carlo_results,
            probability_cols,
        )

        sorted_results = monte_carlo_percent.sort_values(
            "champion",
            ascending=False,
        )

        top_10 = sorted_results.head(10)

        st.subheader("Top 10 Championship Probabilities")

        st.bar_chart(
            top_10.set_index("team")["champion"],
        )

        st.subheader("Full Monte Carlo Results")

        st.dataframe(
            sorted_results,
            use_container_width=True,
        )

        st.subheader("Saved Visualizations")

        championship_chart = ASSETS_DIR / "championship_probabilities.png"
        final_chart = ASSETS_DIR / "final_probabilities.png"
        semifinal_chart = ASSETS_DIR / "semifinal_probabilities.png"

        if championship_chart.exists():
            st.image(
                str(championship_chart),
                caption="Top 10 Championship Probabilities",
            )

        if final_chart.exists():
            st.image(
                str(final_chart),
                caption="Top 10 Final Appearance Probabilities",
            )

        if semifinal_chart.exists():
            st.image(
                str(semifinal_chart),
                caption="Top 10 Semifinal Appearance Probabilities",
            )

        if not championship_chart.exists() and not final_chart.exists() and not semifinal_chart.exists():
            st.info(
                "Saved chart images were not found. "
                "Run `python scripts/create_visualizations.py` to create them."
            )


with tab4:
    st.header("About the Model")

    st.markdown(
        """
        ### Model Inputs

        The current model uses:

        - Recent form difference features
        - Goals scored and conceded over recent matches
        - Recent win/draw/loss counts
        - Elo rating difference

        ### Prediction Target

        The model predicts the match result from Team A's perspective:

        - `win`
        - `draw`
        - `loss`

        ### Knockout Advancement Rule

        Since knockout matches require a team to advance, draw probability is split evenly:

        ```text
        Team A advances = Team A win probability + 0.5 × draw probability
        Team B advances = Team B win probability + 0.5 × draw probability
        ```

        ### App Design

        The model predictions are generated ahead of time and saved to CSV files.
        The Streamlit app reads those saved outputs, which makes the web app faster and more stable.

        ### Current Limitations

        This is a portfolio MVP, not a final betting-grade model. It does not yet include:

        - Player availability
        - Injuries
        - Squad strength
        - FIFA ranking features
        - Penalty shootout modeling
        - Probability calibration
        - Actual 2026 group-stage form
        """
    )