# Ideal Function Matching Project
---

## 1. Features

- Load and clean CSV datasets (training, ideal, test)
- Compute SSE between training and 50 ideal functions
- Select 4 unique ideal functions
- Compute maximum deviation per training–ideal pair
- Map test points to ideal functions using threshold rules
- Store results in an SQLite database
- Generate an interactive HTML visualization
- Save metadata including selected ideals and deviations

---

## 2. Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

### requirements.txt

```
pandas
numpy
sqlalchemy
bokeh
scipy
```

---

## 3. Project Structure

```
project/
│
├── train.csv
├── ideal.csv
├── test.csv
│
├── least_squares.py
├── tests.py
│
├── assignment.db        # generated output
├── visualization.html   # generated output
├── metadata.json        # generated output
│
└── README.md
```

---

## 4. Running the Project

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Ensure CSV files are available
Place the following in the project folder:
```
train.csv
ideal.csv
test.csv
```

### Step 3 — Run the program
```bash
python main.py
```

### Output files created:
- `assignment.db` — SQLite database containing all tables
- `metadata.json` — information about selected ideal functions and deviations
- `visualization.html` — interactive plot

---

## 5. Output Details

### SQLite Database (`assignment.db`)
Tables generated:
- **training**
- **ideal_functions**
- **test_raw**
- **test_mapped**

### Visualization (`visualization.html`)
Shows:
- Solid lines → training functions  
- Dashed lines → selected ideal functions  
- Circles → mapped test points  
- Crosses → unmapped points  
- Hover tool with X, Y, Ideal Function, Deviation  

Open it with:
```bash
start visualization.html     # Windows
open visualization.html      # macOS
```

### Metadata (`metadata.json`)
Includes:
- Selected ideal functions
- Training-to-ideal mapping
- Max deviations
- SSE summary

---

## 6. Running Tests 


pytest tests.py
```
