from pathlib import Path
import csv
import html

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
def load_csv_rows(path_str):
    path = Path(path_str)

    if not path.exists():
        return None

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_percent(value):
    value = safe_float(value)

    if value <= 1:
        value *= 100

    return value


def format_percent(value):
    return f"{as_percent(value):.2f}%"


def get_team_options(prediction_rows, bracket_rows):
    teams = set()

    if prediction_rows:
        for row in prediction_rows:
            if row.get("team_a"):
                teams.add(row["team_a"])
            if row.get("team_b"):
                teams.add(row["team_b"])

    elif bracket_rows:
        for row in bracket_rows:
            if row.get("team_a"):
                teams.add(row["team_a"])
            if row.get("team_b"):
                teams.add(row["team_b"])

    if not teams:
        teams = set(BRACKET_TEAMS)

    return sorted(teams)


def invert_match_outcome(outcome):
    if outcome == "win":
        return "loss"
    if outcome == "loss":
        return "win"
    return outcome


def normalize_prediction_order(row, requested_team_a, requested_team_b, was_reversed):
    prediction = dict(row)

    if not was_reversed:
        return prediction

    prediction["team_a"] = requested_team_a
    prediction["team_b"] = requested_team_b

    if "team_a_win_probability" in prediction and "team_b_win_probability" in prediction:
        original_team_a_win = prediction.get("team_a_win_probability")
        original_team_b_win = prediction.get("team_b_win_probability")
        prediction["team_a_win_probability"] = original_team_b_win
        prediction["team_b_win_probability"] = original_team_a_win

    if "team_a_advancement_probability" in prediction and "team_b_advancement_probability" in prediction:
        original_team_a_adv = prediction.get("team_a_advancement_probability")
        original_team_b_adv = prediction.get("team_b_advancement_probability")
        prediction["team_a_advancement_probability"] = original_team_b_adv
        prediction["team_b_advancement_probability"] = original_team_a_adv

    if "predicted_outcome_from_team_a_perspective" in prediction:
        prediction["predicted_outcome_from_team_a_perspective"] = invert_match_outcome(
            prediction.get("predicted_outcome_from_team_a_perspective")
        )

    return prediction


def get_saved_match_prediction(prediction_rows, team_a, team_b):
    if not prediction_rows:
        return None

    for row in prediction_rows:
        if row.get("team_a") == team_a and row.get("team_b") == team_b:
            return normalize_prediction_order(row, team_a, team_b, was_reversed=False)

    for row in prediction_rows:
        if row.get("team_a") == team_b and row.get("team_b") == team_a:
            return normalize_prediction_order(row, team_a, team_b, was_reversed=True)

    return None


def make_html_table(rows, columns, probability_columns=None, max_rows=None):
    if probability_columns is None:
        probability_columns = set()
    else:
        probability_columns = set(probability_columns)

    if max_rows is not None:
        rows = rows[:max_rows]

    if not rows:
        return "<p>No rows to display.</p>"

    header_html = "".join(
        f"<th>{html.escape(col)}</th>"
        for col in columns
    )

    body_html = ""

    for row in rows:
        body_html += "<tr>"

        for col in columns:
            value = row.get(col, "")

            if col in probability_columns:
                value = format_percent(value)

            body_html += f"<td>{html.escape(str(value))}</td>"

        body_html += "</tr>"

    return f"""
    <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr>{header_html}</tr>
            </thead>
            <tbody>
                {body_html}
            </tbody>
        </table>
    </div>

    <style>
        table, th, td {{
            border: 1px solid rgba(128, 128, 128, 0.35);
        }}
        th, td {{
            padding: 8px;
            text-align: left;
            white-space: nowrap;
        }}
        th {{
            background-color: rgba(128, 128, 128, 0.15);
        }}
    </style>
    """


def sort_rows_by_probability(rows, column):
    return sorted(
        rows,
        key=lambda row: as_percent(row.get(column, 0)),
        reverse=True,
    )


st.title("⚽ FIFA World Cup 2026 Knockout Match Predictor")

st.markdown(
    """
    This app predicts 2026 World Cup knockout outcomes using saved CSV outputs from a machine learning
    pipeline trained on historical international match results, recent team form, and Elo rating features.

    For stability, the web app does not load the model and does not import pandas, NumPy, scikit-learn,
    or joblib. It only reads precomputed CSV files.
    """
)


prediction_matrix_path = DATA_PROCESSED / "match_prediction_matrix.csv"
deterministic_path = DATA_PROCESSED / "actual_deterministic_bracket_results.csv"
monte_carlo_path = DATA_PROCESSED / "monte_carlo_knockout_simulation_results.csv"

prediction_rows = load_csv_rows(str(prediction_matrix_path))
bracket_rows = load_csv_rows(str(deterministic_path))
monte_carlo_rows = load_csv_rows(str(monte_carlo_path))


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

    if prediction_rows:
        rows_to_search = prediction_rows
        st.caption("Using saved pairwise predictions from `match_prediction_matrix.csv`.")
    else:
        rows_to_search = bracket_rows
        st.info(
            "`match_prediction_matrix.csv` was not found, so this tab is using saved deterministic "
            "bracket matchups only. Some team combinations may not be available until the matrix is generated."
        )

    if not rows_to_search:
        st.warning(
            "No saved prediction CSV was found. Add either "
            "`data/processed/match_prediction_matrix.csv` or "
            "`data/processed/actual_deterministic_bracket_results.csv`."
        )
    else:
        team_options = get_team_options(prediction_rows, bracket_rows)

        col1, col2 = st.columns(2)

        with col1:
            default_team_a = team_options.index("France") if "France" in team_options else 0
            team_a = st.selectbox("Team A", team_options, index=default_team_a)

        with col2:
            default_team_b = team_options.index("Brazil") if "Brazil" in team_options else min(1, len(team_options) - 1)
            team_b = st.selectbox("Team B", team_options, index=default_team_b)

        if team_a == team_b:
            st.warning("Please choose two different teams.")
        else:
            prediction = get_saved_match_prediction(
                rows_to_search,
                team_a,
                team_b,
            )

            if prediction is None:
                st.error(
                    f"No saved prediction was found for {team_a} vs {team_b}. "
                    "Generate `data/processed/match_prediction_matrix.csv` to support every pairwise matchup."
                )
            else:
                st.subheader(f"{team_a} vs {team_b}")

                if "team_a_win_probability" in prediction:
                    metric_col1, metric_col2, metric_col3 = st.columns(3)

                    metric_col1.metric(
                        f"{team_a} Win",
                        format_percent(prediction.get("team_a_win_probability")),
                    )

                    metric_col2.metric(
                        "Draw",
                        format_percent(prediction.get("draw_probability")),
                    )

                    metric_col3.metric(
                        f"{team_b} Win",
                        format_percent(prediction.get("team_b_win_probability")),
                    )

                    st.divider()

                adv_col1, adv_col2, adv_col3 = st.columns(3)

                adv_col1.metric(
                    f"{team_a} Advances",
                    format_percent(prediction.get("team_a_advancement_probability")),
                )

                adv_col2.metric(
                    f"{team_b} Advances",
                    format_percent(prediction.get("team_b_advancement_probability")),
                )

                adv_col3.metric(
                    "Predicted Advancing Team",
                    prediction.get("predicted_advancing_team", ""),
                )

                st.subheader("Saved Prediction Output")

                prediction_columns = [
                    "round",
                    "match_number",
                    "team_a",
                    "team_b",
                    "predicted_outcome_from_team_a_perspective",
                    "team_a_win_probability",
                    "draw_probability",
                    "team_b_win_probability",
                    "team_a_advancement_probability",
                    "team_b_advancement_probability",
                    "predicted_advancing_team",
                ]

                prediction_columns = [col for col in prediction_columns if col in prediction]

                probability_columns = [
                    "team_a_win_probability",
                    "draw_probability",
                    "team_b_win_probability",
                    "team_a_advancement_probability",
                    "team_b_advancement_probability",
                ]

                st.markdown(
                    make_html_table(
                        [prediction],
                        prediction_columns,
                        probability_columns=probability_columns,
                    ),
                    unsafe_allow_html=True,
                )


with tab2:
    st.header("Deterministic Knockout Bracket")

    st.markdown(
        """
        The deterministic bracket always advances the team with the higher advancement probability.
        This produces one fixed predicted bracket.
        """
    )

    if bracket_rows is None:
        st.warning(
            "Deterministic bracket results file not found: "
            "`data/processed/actual_deterministic_bracket_results.csv`."
        )
    else:
        final_rows = [
            row for row in bracket_rows
            if row.get("round") == "Final"
        ]

        if final_rows:
            champion = final_rows[0].get("predicted_advancing_team", "")
            st.success(f"Predicted Champion: {champion}")
        else:
            st.warning("Could not find the final row in the deterministic bracket results.")

        bracket_columns = [
            "round",
            "match_number",
            "team_a",
            "team_b",
            "team_a_advancement_probability",
            "team_b_advancement_probability",
            "predicted_advancing_team",
        ]

        bracket_probability_columns = [
            "team_a_advancement_probability",
            "team_b_advancement_probability",
        ]

        st.markdown(
            make_html_table(
                bracket_rows,
                bracket_columns,
                probability_columns=bracket_probability_columns,
            ),
            unsafe_allow_html=True,
        )


with tab3:
    st.header("Monte Carlo Tournament Results")

    st.markdown(
        """
        The Monte Carlo simulation randomly samples winners based on advancement probabilities.
        This estimates each team's chance of reaching each stage of the tournament.
        """
    )

    if monte_carlo_rows is None:
        st.warning(
            "Monte Carlo results file not found: "
            "`data/processed/monte_carlo_knockout_simulation_results.csv`."
        )
    else:
        sorted_rows = sort_rows_by_probability(
            monte_carlo_rows,
            "champion",
        )

        top_10 = sorted_rows[:10]

        st.subheader("Top 10 Championship Probabilities")

        monte_carlo_columns = [
            "team",
            "round_of_16",
            "quarterfinals",
            "semifinals",
            "final",
            "champion",
        ]

        monte_carlo_probability_columns = [
            "round_of_16",
            "quarterfinals",
            "semifinals",
            "final",
            "champion",
        ]

        st.markdown(
            make_html_table(
                top_10,
                monte_carlo_columns,
                probability_columns=monte_carlo_probability_columns,
            ),
            unsafe_allow_html=True,
        )

        st.subheader("Full Monte Carlo Results")

        st.markdown(
            make_html_table(
                sorted_rows,
                monte_carlo_columns,
                probability_columns=monte_carlo_probability_columns,
            ),
            unsafe_allow_html=True,
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
        The Streamlit app reads those saved outputs instead of loading the model directly.
        This makes the app faster and more stable.

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
