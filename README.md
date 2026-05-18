# Tidify 

Upload a messy CSV or Excel file, get a clean dataset and a bunch of charts. That's basically it.

I built this because I kept doing the same tedious stuff every time I got a new dataset — hunting for nulls, finding duplicates, converting columns that pandas read as strings when they're obviously numbers. Tidify just does all of that automatically so I can jump straight to the interesting part.

---

## What it does

**Cleaning** — when you upload a file, it runs through a few steps:
- Strips whitespace and normalises all the weird ways people write "nothing" — `N/A`, `null`, `NULL`, empty strings — all treated as proper NaNs
- Drops rows and columns that are completely empty (they're just noise)
- Removes duplicate rows
- Tries to convert columns that look numeric but got imported as strings
- Detects date columns by name and parses them properly
- Fills in remaining gaps with the median (for numbers) or the most common value (for text)

After cleaning you get a badge for each thing that was fixed, and a download button for the clean file.

**Visualizations** — 7 chart types, each in its own tab:
- Missing values bar chart (only shows up if there were actually missing values)
- Histograms for every numeric column, with the mean marked
- Correlation heatmap
- Box plots to see spread and outliers
- Bar charts for categorical columns
- Scatter plot of the first two numeric columns, coloured by category if one exists
- Pie chart for low-cardinality columns

Every chart has a download button too.

---

## Getting started

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload a file from the sidebar and you're done.

---

## Requirements

- Python 3.9+
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- openpyxl (for .xlsx files)

All pinned in `requirements.txt`.

---

## File structure

```
tidify/
├── app.py            # everything lives here
├── requirements.txt
└── README.md
```

Deliberately one file — no point splitting it up when the whole thing is this small.

---

## Known limitations

- Big files (100k+ rows) will be slow on chart generation — matplotlib isn't exactly speedy
- Date parsing is heuristic, so it won't catch every format
- Scatter plot always uses the first two numeric columns — there's no column picker yet

---

## Stack

Streamlit · pandas · matplotlib · seaborn
