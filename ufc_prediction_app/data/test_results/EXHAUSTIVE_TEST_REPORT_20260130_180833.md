# UFC Fight Simulation - Exhaustive Test Report

**Generated:** 2026-01-30 18:08:33

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Fights Tested | 195 |
| Winner Prediction Accuracy | **65.1%** |
| Method Prediction Accuracy | **45.1%** |
| Both Correct (Winner + Method) | **31.8%** |
| Errors/Failures | 0 |

## Key Recommendations

1. KO/TKO prediction accuracy (42.3%) is low. Consider increasing the weight of power_puncher tendency and chin_issues vulnerability.

2. Submission prediction accuracy (14.6%) is low. Submissions are inherently unpredictable, but consider boosting submission_hunter tendency weight.

3. Rematch predictions (59.1%) underperform overall accuracy. Head-to-head history factor may need adjustment.

4. Underperforming weight classes: Lightweight (54.2%), Middleweight (52.9%), Women's Bantamweight (46.2%), Catch Weight (50.0%). Consider weight-class-specific adjustments.

5. Best performing model: ensemble (68.7%). Consider weighting this model more heavily in ensemble.

## Performance by Method

| Method | Fights | Winner Accuracy | Method Accuracy |
|--------|--------|-----------------|-----------------|
| KO/TKO | 78 | 70.5% | 42.3% |
| Decision | 69 | 68.1% | 69.6% |
| Submission | 48 | 52.1% | 14.6% |

## Confidence Calibration

| Confidence Level | Fights | Actual Accuracy | Calibration |
|------------------|--------|-----------------|-------------|
| low (50-60%) | 13 | 38.5% | Overconfident |
| medium (60-70%) | 62 | 53.2% | Overconfident |
| high (70-80%) | 68 | 70.6% | Well calibrated |
| very_high (80%+) | 52 | 78.8% | Overconfident |

## Performance by Weight Class

| Weight Class | Fights | Accuracy |
|--------------|--------|----------|
| Lightweight | 24 | 54.2% |
| Welterweight | 22 | 81.8% |
| Bantamweight | 22 | 72.7% |
| Light Heavyweight | 19 | 63.2% |
| Middleweight | 17 | 52.9% |
| Heavyweight | 17 | 58.8% |
| Featherweight | 16 | 75.0% |
| Flyweight | 14 | 71.4% |
| Women's Bantamweight | 13 | 46.2% |
| Women's Strawweight | 12 | 66.7% |
| Women's Flyweight | 7 | 71.4% |
| Catch Weight | 6 | 50.0% |
| Women's Featherweight | 6 | 83.3% |

## Model Performance Comparison

| Model | Accuracy | Notes |
|-------|----------|-------|
| ensemble | 68.7% | Primary recommendation model |
| historical | 68.2% | Experience weighted |
| statistical | 63.6% | Pure stats-based |
| momentum | 61.5% | Recent form weighted |
| stylistic | 59.5% | Style matchup focused |

## Special Categories

### Rematches
- Total: 22 fights
- Accuracy: 59.1%
- Head-to-head history factor is needs refinement

### Title Fights
- Total: 70 fights
- Accuracy: 71.4%

## Upset Analysis

**Total upsets (wrong predictions):** 68 (34.9%)

### Upsets by Method:
- KO/TKO: 23
- Decision: 22
- Submission: 23

## Test Methodology

### Fight Selection Criteria:
1. **Weight Class Coverage**: 5 fights per major weight class
2. **KO/TKO Finishes**: 30 randomly selected KO/TKO victories
3. **Submission Finishes**: 25 randomly selected submission victories
4. **Title Fights**: 30 championship bouts
5. **Rematches**: 25 fights where fighters had met before
6. **Recent High-Profile**: 30 main events from 2020+

### Simulation Process:
1. Load both fighters' profiles and statistics
2. Calculate head-to-head history (if any)
3. Run 5 simulation models (statistical, momentum, stylistic, historical, ensemble)
4. Select most realistic result based on realism score
5. Compare predicted winner and method to actual outcome