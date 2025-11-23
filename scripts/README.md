# Project Management Scripts

This directory contains scripts to help analyze and manage project progress.

## Scripts

### `analyze_progress.py`
Analyzes weekly progress reports and generates project management summaries.

**Usage:**
```bash
python3 scripts/analyze_progress.py
```

**Output:**
- `PROJECT_MANAGEMENT_SUMMARY.md` - Detailed summary with statistics
- `project_tickets.json` - Ticket data in JSON format for programmatic access

**Features:**
- Extracts completed and incomplete tasks from progress reports
- Categorizes tasks by type (Data Processing, Model Implementation, etc.)
- Calculates completion rates and velocity metrics
- Generates ticket lists for GitHub Projects

## Generated Reports

### PROJECT_MANAGEMENT_SUMMARY.md
Quick summary with:
- Overall statistics
- Weekly breakdown
- Category breakdown
- Velocity analysis
- Ticket lists

### PROJECT_MANAGEMENT_REPORT.md
Comprehensive report with:
- Executive summary
- Detailed weekly progress
- Work breakdown by category
- Velocity trends
- Milestones
- Recommendations for GitHub Projects

### project_tickets.json
Machine-readable ticket data for:
- Integration with GitHub API
- Custom reporting tools
- Project management dashboards

## Running the Analysis

```bash
# From project root
cd /Users/sirishag/Documents/GitHub/fall-2025-group8
python3 scripts/analyze_progress.py
```

## Customization

To modify categories or add new analysis:
1. Edit `categorize_task()` function in `analyze_progress.py`
2. Modify `generate_summary()` for different report formats
3. Add new output formats in `generate_ticket_list()`

