# UFC Fight Simulation - Exhaustive Test Report

**Generated:** 2026-01-30 18:14:19

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Fights Tested | 200 |
| Winner Prediction Accuracy | **68.5%** |
| Method Prediction Accuracy | **53.0%** |
| Both Correct (Winner + Method) | **40.0%** |
| Errors/Failures | 0 |

## Key Recommendations

1. Decision fights have lower prediction accuracy than average. Consider improving the round-by-round scoring simulation.

2. Rematch predictions (68.0%) underperform overall accuracy. Head-to-head history factor may need adjustment.

3. Underperforming weight classes: Women's Strawweight (58.3%), Women's Flyweight (58.3%), Women's Featherweight (50.0%). Consider weight-class-specific adjustments.

4. Best performing model: statistical (70.5%). Consider weighting this model more heavily in ensemble.

## Performance by Method

| Method | Fights | Winner Accuracy | Method Accuracy |
|--------|--------|-----------------|-----------------|
| KO/TKO | 82 | 74.4% | 63.4% |
| Decision | 77 | 62.3% | 44.2% |
| Submission | 41 | 68.3% | 48.8% |

## Confidence Calibration

| Confidence Level | Fights | Actual Accuracy | Calibration |
|------------------|--------|-----------------|-------------|
| low (50-60%) | 68 | 54.4% | Well calibrated |
| medium (60-70%) | 46 | 76.1% | Underconfident |
| high (70-80%) | 62 | 72.6% | Well calibrated |
| very_high (80%+) | 14 | 92.9% | Underconfident |

## Performance by Weight Class

| Weight Class | Fights | Accuracy |
|--------------|--------|----------|
| Lightweight | 28 | 78.6% |
| Heavyweight | 23 | 78.3% |
| Middleweight | 22 | 59.1% |
| Featherweight | 22 | 63.6% |
| Welterweight | 21 | 71.4% |
| Bantamweight | 14 | 71.4% |
| Light Heavyweight | 12 | 83.3% |
| Flyweight | 12 | 66.7% |
| Women's Strawweight | 12 | 58.3% |
| Women's Flyweight | 12 | 58.3% |
| Women's Bantamweight | 10 | 60.0% |
| Catch Weight | 6 | 66.7% |
| Women's Featherweight | 6 | 50.0% |

## Model Performance Comparison

| Model | Accuracy | Notes |
|-------|----------|-------|
| statistical | 70.5% | Pure stats-based |
| stylistic | 66.5% | Style matchup focused |
| ensemble | 66.0% | Primary recommendation model |
| historical | 66.0% | Experience weighted |
| momentum | 63.0% | Recent form weighted |

## Special Categories

### Rematches
- Total: 25 fights
- Accuracy: 68.0%
- Head-to-head history factor is needs refinement

### Title Fights
- Total: 71 fights
- Accuracy: 64.8%

## Upset Analysis

**Total upsets (wrong predictions):** 63 (31.5%)

### Upsets by Method:
- Submission: 13
- Decision: 29
- KO/TKO: 21

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