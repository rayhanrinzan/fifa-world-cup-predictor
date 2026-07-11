import numpy as np
import pandas as pd


class BracketSimulator:
    def __init__(self, predictor):
        self.predictor = predictor

    def simulate_round(self, matchups, round_name):
        winners = []
        round_results = []

        for match_number, (team_a, team_b) in enumerate(matchups, start=1):
            prediction = self.predictor.predict_knockout_match(team_a, team_b)
            row = prediction.iloc[0].to_dict()

            winner = row["predicted_advancing_team"]
            winners.append(winner)

            row["round"] = round_name
            row["match_number"] = match_number

            round_results.append(row)

        round_results_df = pd.DataFrame(round_results)

        return winners, round_results_df

    def make_next_round_matchups(self, winners):
        if len(winners) % 2 != 0:
            raise ValueError("Number of winners must be even.")

        return [
            (winners[i], winners[i + 1])
            for i in range(0, len(winners), 2)
        ]

    def simulate_deterministic_bracket(self, round_of_32_matchups):
        all_round_results = []

        r32_winners, r32_results = self.simulate_round(
            round_of_32_matchups,
            "Round of 32"
        )
        all_round_results.append(r32_results)

        r16_matchups = self.make_next_round_matchups(r32_winners)
        r16_winners, r16_results = self.simulate_round(
            r16_matchups,
            "Round of 16"
        )
        all_round_results.append(r16_results)

        qf_matchups = self.make_next_round_matchups(r16_winners)
        qf_winners, qf_results = self.simulate_round(
            qf_matchups,
            "Quarterfinals"
        )
        all_round_results.append(qf_results)

        sf_matchups = self.make_next_round_matchups(qf_winners)
        sf_winners, sf_results = self.simulate_round(
            sf_matchups,
            "Semifinals"
        )
        all_round_results.append(sf_results)

        final_matchup = self.make_next_round_matchups(sf_winners)
        final_winner, final_results = self.simulate_round(
            final_matchup,
            "Final"
        )
        all_round_results.append(final_results)

        champion = final_winner[0]
        bracket_results_df = pd.concat(all_round_results, ignore_index=True)

        return champion, bracket_results_df

    def sample_knockout_winner(self, team_a, team_b):
        prediction = self.predictor.predict_knockout_match(team_a, team_b)
        row = prediction.iloc[0]

        team_a_adv_prob = row["team_a_advancement_probability"]
        team_b_adv_prob = row["team_b_advancement_probability"]

        total_prob = team_a_adv_prob + team_b_adv_prob

        team_a_adv_prob = team_a_adv_prob / total_prob
        team_b_adv_prob = team_b_adv_prob / total_prob

        winner = np.random.choice(
            [row["team_a"], row["team_b"]],
            p=[team_a_adv_prob, team_b_adv_prob]
        )

        return winner

    def simulate_bracket_once(self, round_of_32_matchups):
        round_results = {
            "Round of 16": [],
            "Quarterfinals": [],
            "Semifinals": [],
            "Final": [],
            "Champion": [],
        }

        r32_winners = [
            self.sample_knockout_winner(team_a, team_b)
            for team_a, team_b in round_of_32_matchups
        ]
        round_results["Round of 16"] = r32_winners

        r16_matchups = self.make_next_round_matchups(r32_winners)
        r16_winners = [
            self.sample_knockout_winner(team_a, team_b)
            for team_a, team_b in r16_matchups
        ]
        round_results["Quarterfinals"] = r16_winners

        qf_matchups = self.make_next_round_matchups(r16_winners)
        qf_winners = [
            self.sample_knockout_winner(team_a, team_b)
            for team_a, team_b in qf_matchups
        ]
        round_results["Semifinals"] = qf_winners

        sf_matchups = self.make_next_round_matchups(qf_winners)
        sf_winners = [
            self.sample_knockout_winner(team_a, team_b)
            for team_a, team_b in sf_matchups
        ]
        round_results["Final"] = sf_winners

        final_matchup = self.make_next_round_matchups(sf_winners)
        champion = self.sample_knockout_winner(
            final_matchup[0][0],
            final_matchup[0][1]
        )
        round_results["Champion"] = [champion]

        return round_results

    def run_monte_carlo_simulation(self, round_of_32_matchups, n_simulations=10000):
        teams = sorted(
            set(team for matchup in round_of_32_matchups for team in matchup)
        )

        results = pd.DataFrame({
            "team": teams,
            "round_of_16": 0,
            "quarterfinals": 0,
            "semifinals": 0,
            "final": 0,
            "champion": 0,
        }).set_index("team")

        for _ in range(n_simulations):
            simulation = self.simulate_bracket_once(round_of_32_matchups)

            for team in simulation["Round of 16"]:
                results.loc[team, "round_of_16"] += 1

            for team in simulation["Quarterfinals"]:
                results.loc[team, "quarterfinals"] += 1

            for team in simulation["Semifinals"]:
                results.loc[team, "semifinals"] += 1

            for team in simulation["Final"]:
                results.loc[team, "final"] += 1

            for team in simulation["Champion"]:
                results.loc[team, "champion"] += 1

        results = results / n_simulations
        results = results.reset_index()

        return results