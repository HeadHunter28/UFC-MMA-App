"""
Exhaustive Fight Simulation Testing Script.

This script:
1. Carefully selects diverse past fights from the database
2. Runs simulations against known outcomes
3. Stores predictions for model improvement analysis
4. Generates a comprehensive testing report
"""

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATABASE_PATH
from services.simulation_service import FightSimulationService


class ExhaustiveSimulationTester:
    """
    Comprehensive test runner for the fight simulation system.
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the tester."""
        self.db_path = db_path or str(DATABASE_PATH)
        self.sim_service = FightSimulationService()
        self.predictions = []
        self.test_fights = []

    def get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def select_diverse_test_fights(self, total_fights: int = 200) -> List[Dict[str, Any]]:
        """
        Select a diverse set of past fights for testing.

        Categories:
        - Different weight classes (proportional representation)
        - Different methods (KO/TKO, Submission, Decision)
        - Upsets (underdog wins)
        - Title fights
        - Rematches
        - Recent fights (2020+) and classic fights
        """
        conn = self.get_connection()
        fights = []

        # 1. Get fights by weight class (ensure all weight classes represented)
        print("\n[1/6] Selecting fights by weight class...")
        weight_class_query = """
            SELECT DISTINCT weight_class, COUNT(*) as fight_count
            FROM fights
            WHERE winner_id IS NOT NULL AND method IN ('KO/TKO', 'Submission', 'Decision')
            GROUP BY weight_class
            ORDER BY fight_count DESC
        """
        cursor = conn.execute(weight_class_query)
        weight_classes = [(row['weight_class'], row['fight_count']) for row in cursor.fetchall()]

        # Select ~5 fights per weight class for good coverage
        for wc, count in weight_classes[:14]:  # Top 14 weight classes
            wc_fights = conn.execute("""
                SELECT f.*, e.name as event_name, e.date as event_date,
                       fr.name as fighter_red_name, fb.name as fighter_blue_name,
                       fr.wins as red_wins, fr.losses as red_losses,
                       fb.wins as blue_wins, fb.losses as blue_losses
                FROM fights f
                JOIN events e ON f.event_id = e.event_id
                JOIN fighters fr ON f.fighter_red_id = fr.fighter_id
                JOIN fighters fb ON f.fighter_blue_id = fb.fighter_id
                WHERE f.winner_id IS NOT NULL
                  AND f.weight_class = ?
                  AND f.method IN ('KO/TKO', 'Submission', 'Decision')
                ORDER BY RANDOM()
                LIMIT 5
            """, (wc,)).fetchall()
            for fight in wc_fights:
                fights.append({**dict(fight), 'selection_reason': f'weight_class_{wc}'})

        # 2. Get KO/TKO finishes
        print("[2/6] Selecting KO/TKO finishes...")
        ko_fights = conn.execute("""
            SELECT f.*, e.name as event_name, e.date as event_date,
                   fr.name as fighter_red_name, fb.name as fighter_blue_name,
                   fr.wins as red_wins, fr.losses as red_losses,
                   fb.wins as blue_wins, fb.losses as blue_losses
            FROM fights f
            JOIN events e ON f.event_id = e.event_id
            JOIN fighters fr ON f.fighter_red_id = fr.fighter_id
            JOIN fighters fb ON f.fighter_blue_id = fb.fighter_id
            WHERE f.winner_id IS NOT NULL
              AND f.method = 'KO/TKO'
            ORDER BY RANDOM()
            LIMIT 30
        """).fetchall()
        for fight in ko_fights:
            fights.append({**dict(fight), 'selection_reason': 'ko_tko_finish'})

        # 3. Get Submission finishes
        print("[3/6] Selecting Submission finishes...")
        sub_fights = conn.execute("""
            SELECT f.*, e.name as event_name, e.date as event_date,
                   fr.name as fighter_red_name, fb.name as fighter_blue_name,
                   fr.wins as red_wins, fr.losses as red_losses,
                   fb.wins as blue_wins, fb.losses as blue_losses
            FROM fights f
            JOIN events e ON f.event_id = e.event_id
            JOIN fighters fr ON f.fighter_red_id = fr.fighter_id
            JOIN fighters fb ON f.fighter_blue_id = fb.fighter_id
            WHERE f.winner_id IS NOT NULL
              AND f.method = 'Submission'
            ORDER BY RANDOM()
            LIMIT 25
        """).fetchall()
        for fight in sub_fights:
            fights.append({**dict(fight), 'selection_reason': 'submission_finish'})

        # 4. Get Title fights
        print("[4/6] Selecting Title fights...")
        title_fights = conn.execute("""
            SELECT f.*, e.name as event_name, e.date as event_date,
                   fr.name as fighter_red_name, fb.name as fighter_blue_name,
                   fr.wins as red_wins, fr.losses as red_losses,
                   fb.wins as blue_wins, fb.losses as blue_losses
            FROM fights f
            JOIN events e ON f.event_id = e.event_id
            JOIN fighters fr ON f.fighter_red_id = fr.fighter_id
            JOIN fighters fb ON f.fighter_blue_id = fb.fighter_id
            WHERE f.winner_id IS NOT NULL
              AND f.is_title_fight = 1
              AND f.method IN ('KO/TKO', 'Submission', 'Decision')
            ORDER BY RANDOM()
            LIMIT 30
        """).fetchall()
        for fight in title_fights:
            fights.append({**dict(fight), 'selection_reason': 'title_fight'})

        # 5. Get Rematches (fighters who have fought before)
        print("[5/6] Selecting Rematches...")
        rematch_fights = conn.execute("""
            WITH fighter_pairs AS (
                SELECT
                    MIN(fighter_red_id, fighter_blue_id) as f1,
                    MAX(fighter_red_id, fighter_blue_id) as f2,
                    COUNT(*) as fight_count
                FROM fights
                WHERE winner_id IS NOT NULL
                GROUP BY MIN(fighter_red_id, fighter_blue_id), MAX(fighter_red_id, fighter_blue_id)
                HAVING COUNT(*) >= 2
            )
            SELECT f.*, e.name as event_name, e.date as event_date,
                   fr.name as fighter_red_name, fb.name as fighter_blue_name,
                   fr.wins as red_wins, fr.losses as red_losses,
                   fb.wins as blue_wins, fb.losses as blue_losses,
                   fp.fight_count as total_meetings
            FROM fights f
            JOIN events e ON f.event_id = e.event_id
            JOIN fighters fr ON f.fighter_red_id = fr.fighter_id
            JOIN fighters fb ON f.fighter_blue_id = fb.fighter_id
            JOIN fighter_pairs fp ON (
                MIN(f.fighter_red_id, f.fighter_blue_id) = fp.f1
                AND MAX(f.fighter_red_id, f.fighter_blue_id) = fp.f2
            )
            WHERE f.winner_id IS NOT NULL
              AND f.method IN ('KO/TKO', 'Submission', 'Decision')
            ORDER BY RANDOM()
            LIMIT 25
        """).fetchall()
        for fight in rematch_fights:
            fights.append({**dict(fight), 'selection_reason': 'rematch'})

        # 6. Get recent high-profile fights (2020+)
        print("[6/6] Selecting recent fights (2020+)...")
        recent_fights = conn.execute("""
            SELECT f.*, e.name as event_name, e.date as event_date,
                   fr.name as fighter_red_name, fb.name as fighter_blue_name,
                   fr.wins as red_wins, fr.losses as red_losses,
                   fb.wins as blue_wins, fb.losses as blue_losses
            FROM fights f
            JOIN events e ON f.event_id = e.event_id
            JOIN fighters fr ON f.fighter_red_id = fr.fighter_id
            JOIN fighters fb ON f.fighter_blue_id = fb.fighter_id
            WHERE f.winner_id IS NOT NULL
              AND e.date >= '2020-01-01'
              AND f.method IN ('KO/TKO', 'Submission', 'Decision')
              AND (f.is_main_event = 1 OR f.is_title_fight = 1)
            ORDER BY RANDOM()
            LIMIT 30
        """).fetchall()
        for fight in recent_fights:
            fights.append({**dict(fight), 'selection_reason': 'recent_high_profile'})

        conn.close()

        # Deduplicate by fight_id while preserving selection reasons
        seen = {}
        for fight in fights:
            fid = fight['fight_id']
            if fid not in seen:
                seen[fid] = fight
            else:
                # Append selection reasons
                existing_reason = seen[fid].get('selection_reason', '')
                new_reason = fight.get('selection_reason', '')
                if new_reason and new_reason not in existing_reason:
                    seen[fid]['selection_reason'] = f"{existing_reason}, {new_reason}"

        unique_fights = list(seen.values())[:total_fights]

        print(f"\nSelected {len(unique_fights)} unique fights for testing")
        self.test_fights = unique_fights
        return unique_fights

    def run_simulation_test(self, fight: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run simulation on a single fight and compare to actual outcome.
        """
        fighter_red_id = fight['fighter_red_id']
        fighter_blue_id = fight['fighter_blue_id']
        actual_winner_id = fight['winner_id']
        actual_method = fight['method']

        # Run simulation
        try:
            simulations = self.sim_service.simulate_fight(fighter_red_id, fighter_blue_id)
        except Exception as e:
            return {
                'fight_id': fight['fight_id'],
                'error': str(e),
                'prediction_correct': False,
                'method_correct': False,
            }

        if not simulations:
            return {
                'fight_id': fight['fight_id'],
                'error': 'No simulations returned',
                'prediction_correct': False,
                'method_correct': False,
            }

        # Get most realistic simulation (top result)
        best_sim = simulations[0]

        # Determine predicted winner
        predicted_winner_id = best_sim.winner_id
        predicted_method = best_sim.method
        confidence = best_sim.confidence

        # Compare to actual outcome
        prediction_correct = predicted_winner_id == actual_winner_id
        method_correct = self._method_matches(predicted_method, actual_method)

        # Check if this was a rematch (head-to-head history)
        is_rematch = bool(fight.get('total_meetings', 0) > 1)

        return {
            'fight_id': fight['fight_id'],
            'event_name': fight.get('event_name', 'Unknown'),
            'event_date': fight.get('event_date', 'Unknown'),
            'fighter_red_name': fight.get('fighter_red_name', 'Red'),
            'fighter_blue_name': fight.get('fighter_blue_name', 'Blue'),
            'fighter_red_id': fighter_red_id,
            'fighter_blue_id': fighter_blue_id,
            'weight_class': fight.get('weight_class', 'Unknown'),
            'is_title_fight': fight.get('is_title_fight', False),
            'selection_reason': fight.get('selection_reason', ''),
            'is_rematch': is_rematch,
            # Actual outcome
            'actual_winner_id': actual_winner_id,
            'actual_winner_name': fight['fighter_red_name'] if actual_winner_id == fighter_red_id else fight['fighter_blue_name'],
            'actual_method': actual_method,
            'actual_round': fight.get('round'),
            # Predicted outcome
            'predicted_winner_id': predicted_winner_id,
            'predicted_winner_name': best_sim.winner_name,
            'predicted_method': predicted_method,
            'predicted_round': best_sim.finish_round,
            'confidence': confidence,
            # Analysis
            'prediction_correct': prediction_correct,
            'method_correct': method_correct,
            'both_correct': prediction_correct and method_correct,
            'model_name': best_sim.model_name,
            'realism_score': best_sim.realism_score,
            'key_factors': best_sim.key_factors,
            # All model results for ensemble analysis
            'all_models': [
                {
                    'model': sim.model_name,
                    'winner_id': sim.winner_id,
                    'method': sim.method,
                    'confidence': sim.confidence,
                    'realism_score': sim.realism_score,
                }
                for sim in simulations
            ],
        }

    def _method_matches(self, predicted: str, actual: str) -> bool:
        """Check if predicted method matches actual (allowing for grouping)."""
        predicted = predicted.lower()
        actual = actual.lower()

        if predicted == actual:
            return True

        # Group KO and TKO together
        ko_group = ['ko', 'tko', 'ko/tko']
        if any(p in predicted for p in ko_group) and any(a in actual for a in ko_group):
            return True

        return False

    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run simulations on all selected test fights."""
        if not self.test_fights:
            print("No test fights selected. Run select_diverse_test_fights() first.")
            return []

        print(f"\nRunning simulations on {len(self.test_fights)} fights...")
        print("=" * 60)

        for i, fight in enumerate(self.test_fights):
            result = self.run_simulation_test(fight)
            self.predictions.append(result)

            # Progress indicator
            if (i + 1) % 10 == 0 or (i + 1) == len(self.test_fights):
                correct = sum(1 for p in self.predictions if p.get('prediction_correct', False))
                print(f"[{i+1}/{len(self.test_fights)}] Accuracy so far: {correct}/{len(self.predictions)} ({100*correct/len(self.predictions):.1f}%)")

        return self.predictions

    def generate_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report from predictions."""
        if not self.predictions:
            return {'error': 'No predictions to analyze'}

        # Filter out errors
        valid_predictions = [p for p in self.predictions if 'error' not in p]
        error_count = len(self.predictions) - len(valid_predictions)

        # Basic accuracy metrics
        total = len(valid_predictions)
        winner_correct = sum(1 for p in valid_predictions if p['prediction_correct'])
        method_correct = sum(1 for p in valid_predictions if p['method_correct'])
        both_correct = sum(1 for p in valid_predictions if p['both_correct'])

        # Accuracy by actual method
        method_breakdown = defaultdict(lambda: {'total': 0, 'winner_correct': 0, 'method_correct': 0})
        for p in valid_predictions:
            method = p['actual_method']
            method_breakdown[method]['total'] += 1
            if p['prediction_correct']:
                method_breakdown[method]['winner_correct'] += 1
            if p['method_correct']:
                method_breakdown[method]['method_correct'] += 1

        # Accuracy by weight class
        weight_class_breakdown = defaultdict(lambda: {'total': 0, 'correct': 0})
        for p in valid_predictions:
            wc = p['weight_class']
            weight_class_breakdown[wc]['total'] += 1
            if p['prediction_correct']:
                weight_class_breakdown[wc]['correct'] += 1

        # Accuracy by confidence level
        confidence_brackets = {
            'low (50-60%)': (0.50, 0.60),
            'medium (60-70%)': (0.60, 0.70),
            'high (70-80%)': (0.70, 0.80),
            'very_high (80%+)': (0.80, 1.01),
        }
        confidence_breakdown = {}
        for bracket_name, (low, high) in confidence_brackets.items():
            bracket_preds = [p for p in valid_predictions if low <= p['confidence'] < high]
            if bracket_preds:
                correct = sum(1 for p in bracket_preds if p['prediction_correct'])
                confidence_breakdown[bracket_name] = {
                    'total': len(bracket_preds),
                    'correct': correct,
                    'accuracy': correct / len(bracket_preds),
                }

        # Rematch analysis
        rematch_preds = [p for p in valid_predictions if p.get('is_rematch')]
        rematch_stats = {
            'total': len(rematch_preds),
            'correct': sum(1 for p in rematch_preds if p['prediction_correct']),
        }
        if rematch_stats['total'] > 0:
            rematch_stats['accuracy'] = rematch_stats['correct'] / rematch_stats['total']

        # Title fight analysis
        title_preds = [p for p in valid_predictions if p.get('is_title_fight')]
        title_stats = {
            'total': len(title_preds),
            'correct': sum(1 for p in title_preds if p['prediction_correct']),
        }
        if title_stats['total'] > 0:
            title_stats['accuracy'] = title_stats['correct'] / title_stats['total']

        # Model performance comparison
        model_performance = defaultdict(lambda: {'correct': 0, 'total': 0})
        for p in valid_predictions:
            for model_result in p.get('all_models', []):
                model_name = model_result['model']
                model_performance[model_name]['total'] += 1
                if model_result['winner_id'] == p['actual_winner_id']:
                    model_performance[model_name]['correct'] += 1

        # Upset detection (when prediction was wrong)
        upsets = [p for p in valid_predictions if not p['prediction_correct']]

        # Identify patterns in wrong predictions
        wrong_by_method = defaultdict(list)
        for p in upsets:
            wrong_by_method[p['actual_method']].append(p)

        report = {
            'summary': {
                'total_fights_tested': total,
                'errors': error_count,
                'winner_accuracy': winner_correct / total if total > 0 else 0,
                'method_accuracy': method_correct / total if total > 0 else 0,
                'both_correct_accuracy': both_correct / total if total > 0 else 0,
                'winner_correct_count': winner_correct,
                'method_correct_count': method_correct,
            },
            'method_breakdown': {
                method: {
                    'total': data['total'],
                    'winner_accuracy': data['winner_correct'] / data['total'] if data['total'] > 0 else 0,
                    'method_accuracy': data['method_correct'] / data['total'] if data['total'] > 0 else 0,
                }
                for method, data in method_breakdown.items()
            },
            'weight_class_breakdown': {
                wc: {
                    'total': data['total'],
                    'accuracy': data['correct'] / data['total'] if data['total'] > 0 else 0,
                }
                for wc, data in weight_class_breakdown.items()
            },
            'confidence_calibration': confidence_breakdown,
            'rematch_analysis': rematch_stats,
            'title_fight_analysis': title_stats,
            'model_performance': {
                model: {
                    'total': data['total'],
                    'accuracy': data['correct'] / data['total'] if data['total'] > 0 else 0,
                }
                for model, data in model_performance.items()
            },
            'upset_analysis': {
                'total_upsets': len(upsets),
                'upset_rate': len(upsets) / total if total > 0 else 0,
                'upsets_by_method': {
                    method: len(fights)
                    for method, fights in wrong_by_method.items()
                },
            },
            'timestamp': datetime.now().isoformat(),
        }

        return report

    def generate_improvement_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate specific improvement recommendations based on test results."""
        recommendations = []

        # Check overall accuracy
        winner_acc = report['summary']['winner_accuracy']
        method_acc = report['summary']['method_accuracy']

        if winner_acc < 0.60:
            recommendations.append(
                f"CRITICAL: Overall winner prediction accuracy ({winner_acc:.1%}) is below 60%. "
                "Consider reviewing the base probability calculations and stat weighting."
            )
        elif winner_acc < 0.65:
            recommendations.append(
                f"Winner prediction accuracy ({winner_acc:.1%}) has room for improvement. "
                "Consider adding more factors or refining existing weights."
            )

        # Check method prediction
        method_breakdown = report.get('method_breakdown', {})

        ko_data = method_breakdown.get('KO/TKO', {})
        if ko_data.get('method_accuracy', 0) < 0.50:
            recommendations.append(
                f"KO/TKO prediction accuracy ({ko_data.get('method_accuracy', 0):.1%}) is low. "
                "Consider increasing the weight of power_puncher tendency and chin_issues vulnerability."
            )

        sub_data = method_breakdown.get('Submission', {})
        if sub_data.get('method_accuracy', 0) < 0.30:
            recommendations.append(
                f"Submission prediction accuracy ({sub_data.get('method_accuracy', 0):.1%}) is low. "
                "Submissions are inherently unpredictable, but consider boosting submission_hunter tendency weight."
            )

        dec_data = method_breakdown.get('Decision', {})
        if dec_data.get('winner_accuracy', 0) < winner_acc:
            recommendations.append(
                "Decision fights have lower prediction accuracy than average. "
                "Consider improving the round-by-round scoring simulation."
            )

        # Check confidence calibration
        confidence_cal = report.get('confidence_calibration', {})
        high_conf = confidence_cal.get('high (70-80%)', {})
        if high_conf.get('accuracy', 0) < 0.70:
            recommendations.append(
                f"High confidence predictions ({high_conf.get('accuracy', 0):.1%} actual) are overconfident. "
                "Consider reducing the confidence calculation or adding uncertainty factors."
            )

        low_conf = confidence_cal.get('low (50-60%)', {})
        if low_conf.get('accuracy', 0) > 0.60:
            recommendations.append(
                f"Low confidence predictions ({low_conf.get('accuracy', 0):.1%} actual) are underconfident. "
                "The model identifies good picks but doesn't express enough confidence."
            )

        # Check rematch performance
        rematch = report.get('rematch_analysis', {})
        if rematch.get('total', 0) >= 10:
            rematch_acc = rematch.get('accuracy', 0)
            if rematch_acc < winner_acc:
                recommendations.append(
                    f"Rematch predictions ({rematch_acc:.1%}) underperform overall accuracy. "
                    "Head-to-head history factor may need adjustment."
                )
            elif rematch_acc > winner_acc + 0.05:
                recommendations.append(
                    f"Rematch predictions ({rematch_acc:.1%}) outperform overall accuracy. "
                    "Head-to-head history integration is working well."
                )

        # Check weight class performance
        wc_breakdown = report.get('weight_class_breakdown', {})
        underperforming_wc = [
            (wc, data['accuracy'])
            for wc, data in wc_breakdown.items()
            if data['total'] >= 5 and data['accuracy'] < winner_acc - 0.10
        ]
        if underperforming_wc:
            wc_list = ", ".join([f"{wc} ({acc:.1%})" for wc, acc in underperforming_wc])
            recommendations.append(
                f"Underperforming weight classes: {wc_list}. "
                "Consider weight-class-specific adjustments."
            )

        # Model comparison
        model_perf = report.get('model_performance', {})
        best_model = max(model_perf.items(), key=lambda x: x[1]['accuracy'], default=(None, {}))
        if best_model[0]:
            recommendations.append(
                f"Best performing model: {best_model[0]} ({best_model[1]['accuracy']:.1%}). "
                "Consider weighting this model more heavily in ensemble."
            )

        if not recommendations:
            recommendations.append(
                "Model is performing well overall. Continue monitoring with more test data."
            )

        return recommendations

    def save_results(self, output_dir: Optional[str] = None):
        """Save all results to files."""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "data" / "test_results"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw predictions
        predictions_file = output_dir / f"predictions_{timestamp}.json"
        with open(predictions_file, 'w') as f:
            json.dump(self.predictions, f, indent=2, default=str)
        print(f"\nSaved predictions to: {predictions_file}")

        # Generate and save report
        report = self.generate_analysis_report()
        report_file = output_dir / f"analysis_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Saved analysis report to: {report_file}")

        # Generate and save recommendations
        recommendations = self.generate_improvement_recommendations(report)
        recommendations_file = output_dir / f"recommendations_{timestamp}.txt"
        with open(recommendations_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("FIGHT SIMULATION MODEL - IMPROVEMENT RECOMMENDATIONS\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")

            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total fights tested: {report['summary']['total_fights_tested']}\n")
            f.write(f"Winner accuracy: {report['summary']['winner_accuracy']:.1%}\n")
            f.write(f"Method accuracy: {report['summary']['method_accuracy']:.1%}\n")
            f.write(f"Both correct: {report['summary']['both_correct_accuracy']:.1%}\n\n")

            f.write("RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"\n{i}. {rec}\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("DETAILED BREAKDOWNS\n")
            f.write("=" * 70 + "\n\n")

            f.write("BY METHOD:\n")
            for method, data in report.get('method_breakdown', {}).items():
                f.write(f"  {method}: {data['total']} fights, "
                       f"winner acc: {data['winner_accuracy']:.1%}, "
                       f"method acc: {data['method_accuracy']:.1%}\n")

            f.write("\nBY CONFIDENCE LEVEL:\n")
            for level, data in report.get('confidence_calibration', {}).items():
                f.write(f"  {level}: {data['total']} fights, actual acc: {data['accuracy']:.1%}\n")

            f.write("\nBY MODEL:\n")
            for model, data in report.get('model_performance', {}).items():
                f.write(f"  {model}: {data['accuracy']:.1%} ({data['total']} predictions)\n")

        print(f"Saved recommendations to: {recommendations_file}")

        # Generate detailed markdown report
        md_report = self._generate_markdown_report(report, recommendations)
        md_file = output_dir / f"EXHAUSTIVE_TEST_REPORT_{timestamp}.md"
        with open(md_file, 'w') as f:
            f.write(md_report)
        print(f"Saved markdown report to: {md_file}")

        return {
            'predictions_file': str(predictions_file),
            'report_file': str(report_file),
            'recommendations_file': str(recommendations_file),
            'markdown_file': str(md_file),
        }

    def _generate_markdown_report(self, report: Dict, recommendations: List[str]) -> str:
        """Generate a comprehensive markdown report."""
        md = []
        md.append("# UFC Fight Simulation - Exhaustive Test Report")
        md.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Summary
        md.append("## Executive Summary\n")
        summary = report['summary']
        md.append(f"| Metric | Value |")
        md.append(f"|--------|-------|")
        md.append(f"| Total Fights Tested | {summary['total_fights_tested']} |")
        md.append(f"| Winner Prediction Accuracy | **{summary['winner_accuracy']:.1%}** |")
        md.append(f"| Method Prediction Accuracy | **{summary['method_accuracy']:.1%}** |")
        md.append(f"| Both Correct (Winner + Method) | **{summary['both_correct_accuracy']:.1%}** |")
        md.append(f"| Errors/Failures | {summary['errors']} |")
        md.append("")

        # Recommendations
        md.append("## Key Recommendations\n")
        for i, rec in enumerate(recommendations, 1):
            md.append(f"{i}. {rec}\n")

        # Method Breakdown
        md.append("## Performance by Method\n")
        md.append("| Method | Fights | Winner Accuracy | Method Accuracy |")
        md.append("|--------|--------|-----------------|-----------------|")
        for method, data in sorted(report.get('method_breakdown', {}).items(),
                                   key=lambda x: x[1]['total'], reverse=True):
            md.append(f"| {method} | {data['total']} | {data['winner_accuracy']:.1%} | {data['method_accuracy']:.1%} |")
        md.append("")

        # Confidence Calibration
        md.append("## Confidence Calibration\n")
        md.append("| Confidence Level | Fights | Actual Accuracy | Calibration |")
        md.append("|------------------|--------|-----------------|-------------|")
        for level, data in report.get('confidence_calibration', {}).items():
            # Parse the confidence range from the level name
            if '50-60' in level:
                expected = 0.55
            elif '60-70' in level:
                expected = 0.65
            elif '70-80' in level:
                expected = 0.75
            else:
                expected = 0.85

            actual = data['accuracy']
            if actual > expected + 0.05:
                calibration = "Underconfident"
            elif actual < expected - 0.05:
                calibration = "Overconfident"
            else:
                calibration = "Well calibrated"

            md.append(f"| {level} | {data['total']} | {actual:.1%} | {calibration} |")
        md.append("")

        # Weight Class Breakdown
        md.append("## Performance by Weight Class\n")
        md.append("| Weight Class | Fights | Accuracy |")
        md.append("|--------------|--------|----------|")
        for wc, data in sorted(report.get('weight_class_breakdown', {}).items(),
                               key=lambda x: x[1]['total'], reverse=True):
            if data['total'] >= 3:  # Only show classes with sufficient data
                md.append(f"| {wc} | {data['total']} | {data['accuracy']:.1%} |")
        md.append("")

        # Model Comparison
        md.append("## Model Performance Comparison\n")
        md.append("| Model | Accuracy | Notes |")
        md.append("|-------|----------|-------|")
        for model, data in sorted(report.get('model_performance', {}).items(),
                                  key=lambda x: x[1]['accuracy'], reverse=True):
            notes = ""
            if model == "ensemble":
                notes = "Primary recommendation model"
            elif model == "statistical":
                notes = "Pure stats-based"
            elif model == "momentum":
                notes = "Recent form weighted"
            elif model == "stylistic":
                notes = "Style matchup focused"
            elif model == "historical":
                notes = "Experience weighted"
            md.append(f"| {model} | {data['accuracy']:.1%} | {notes} |")
        md.append("")

        # Special Categories
        md.append("## Special Categories\n")

        rematch = report.get('rematch_analysis', {})
        if rematch.get('total', 0) > 0:
            md.append(f"### Rematches")
            md.append(f"- Total: {rematch['total']} fights")
            md.append(f"- Accuracy: {rematch.get('accuracy', 0):.1%}")
            md.append(f"- Head-to-head history factor is {'effective' if rematch.get('accuracy', 0) > summary['winner_accuracy'] else 'needs refinement'}\n")

        title = report.get('title_fight_analysis', {})
        if title.get('total', 0) > 0:
            md.append(f"### Title Fights")
            md.append(f"- Total: {title['total']} fights")
            md.append(f"- Accuracy: {title.get('accuracy', 0):.1%}\n")

        # Upset Analysis
        upset = report.get('upset_analysis', {})
        md.append("## Upset Analysis\n")
        md.append(f"**Total upsets (wrong predictions):** {upset.get('total_upsets', 0)} ({upset.get('upset_rate', 0):.1%})\n")
        md.append("### Upsets by Method:")
        for method, count in upset.get('upsets_by_method', {}).items():
            md.append(f"- {method}: {count}")
        md.append("")

        # Methodology
        md.append("## Test Methodology\n")
        md.append("### Fight Selection Criteria:")
        md.append("1. **Weight Class Coverage**: 5 fights per major weight class")
        md.append("2. **KO/TKO Finishes**: 30 randomly selected KO/TKO victories")
        md.append("3. **Submission Finishes**: 25 randomly selected submission victories")
        md.append("4. **Title Fights**: 30 championship bouts")
        md.append("5. **Rematches**: 25 fights where fighters had met before")
        md.append("6. **Recent High-Profile**: 30 main events from 2020+")
        md.append("")
        md.append("### Simulation Process:")
        md.append("1. Load both fighters' profiles and statistics")
        md.append("2. Calculate head-to-head history (if any)")
        md.append("3. Run 5 simulation models (statistical, momentum, stylistic, historical, ensemble)")
        md.append("4. Select most realistic result based on realism score")
        md.append("5. Compare predicted winner and method to actual outcome")

        return "\n".join(md)


def main():
    """Run the exhaustive simulation test."""
    print("=" * 70)
    print("UFC FIGHT SIMULATION - EXHAUSTIVE TEST")
    print("=" * 70)

    tester = ExhaustiveSimulationTester()

    # Select diverse test fights
    print("\nPhase 1: Selecting diverse test fights...")
    fights = tester.select_diverse_test_fights(total_fights=200)

    # Run all simulations
    print("\nPhase 2: Running simulations...")
    tester.run_all_tests()

    # Generate and save results
    print("\nPhase 3: Generating analysis and saving results...")
    files = tester.save_results()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    # Print summary
    report = tester.generate_analysis_report()
    print(f"\nFinal Results:")
    print(f"  Winner Accuracy: {report['summary']['winner_accuracy']:.1%}")
    print(f"  Method Accuracy: {report['summary']['method_accuracy']:.1%}")
    print(f"  Both Correct: {report['summary']['both_correct_accuracy']:.1%}")

    print(f"\nFiles saved to:")
    for name, path in files.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
