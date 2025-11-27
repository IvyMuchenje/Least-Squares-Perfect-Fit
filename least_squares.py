from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import json
import math
import logging
from sqlalchemy import create_engine
from bokeh.plotting import figure, output_file, save
from bokeh.models import HoverTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Least Squares Assignment Visualization")


def linear_interpolate(source_x: np.ndarray, source_y: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    """ Linear Interpolation """
    return np.interp(target_x, source_x, source_y)


class CSVLoader:
    
    """Loads and normalizes CSV inputs."""
    
    def __init__(self, train_path: Path, ideal_path: Path, test_path: Path):
        self.train_path = Path(train_path)
        self.ideal_path = Path(ideal_path)
        self.test_path = Path(test_path)
        
    def _load_csv(self, path: Path) -> pd.DataFrame:
        """loads csv"""
        if not path.exists():
            raise Exception
        df = pd.read_csv(path)
        if df.shape[1] < 2:
            raise Exception
        return df
        
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = self._load_csv(self.train_path)
        ideal = self._load_csv(self.ideal_path)
        test = self._load_csv(self.test_path)

        # Renaming the first columns to 'X' and following columns sequentially
        train = train.rename(columns = {train.columns[0]: "X"})
        ideal = ideal.rename(columns = {ideal.columns[0]: "X"})
        test = test.rename(columns = {test.columns[0]: "X",test.columns[1]: "Y"})

        # Renaming the training Y columns to Y1....Yn
        train_rename = {c: f"Y{i+1} " for i, c in enumerate(train.columns[1:])}
        train = train.rename(columns = train_rename)
        
        ideal_rename = {c: f"Y{i+1} " for i, c in enumerate(ideal.columns[1:])}
        ideal = ideal.rename(columns = ideal_rename)
        
        return train, ideal, test
    
class DatabaseManager:
    """Handles SQLite storage via SQLAlchemy."""
    
    def __init__(self,db_path: Path):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
    
    def  save_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = "replace"):
        df.to_sql(table_name, con=self.engine, if_exists=if_exists, index=False)
        logger.info(f"Wrote {table_name} ({df.shape[0]} rows) to {self.db_path}")
        
        
class IdealMatcher:
    """
    Matches training functions to ideal functions using least-squares and maps test points to chosen ideals using the sqrt(2) criterion.
    """
    
    def __init__(self, train_df: pd.DataFrame, ideal_df: pd.DataFrame):
        self.train_df = train_df
        self.ideal_df = ideal_df
        self.train_x = train_df["X"].values
        self.ideal_cols = [c for c in ideal_df.columns if c != "X"]
        
    def compute_sse_and_maxdev(self) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:

        """
        For each pair (train_col, ideal_col) compute:
        - SSE (sum squared errors) over training_x
        - max absolute deviation over training_x
        Returns two dicts keyed by (train_col, ideal_col)
        """
        
        sse = {}
        maxdev = {}
        for tcol in [c for c in self.train_df.columns if c != "X"]:
            ytrain = self.train_df[tcol].values
            for icol in  self.ideal_cols:
                yideal_intercep = linear_interpolate(self.ideal_df["X"].values, self.ideal_df[icol].values, self.train_x)
                res = ytrain - yideal_intercep
                sse[(tcol, icol)] = float((res ** 2).sum())
                maxdev[(tcol, icol)] = float(np.abs(res).max())
        return sse, maxdev
        
    def choose_four_ideals(self, sse: Dict[Tuple[str, str], float]) -> Dict[str, str]:
        
        """
        Selects an ideal function for each training function minimizing SSE.
        Enforces uniqueness greedily: if two training functions pickthe same ideal,
        the second will take the next-best unused ideal.
        Returns mapping {train_col: ideal_col} (4  entries).
        """
        selected = {}
        used = set()
        train_cols = [c for c in self.train_df.columns if c != "X"]
        for tcol in train_cols:
            # candidates sorted by sse
            candidates = sorted(self.ideal_cols, key=lambda ic : sse[(tcol, ic)])
            chosen = None
            for c in candidates:
                if c not in used:
                    chosen = c
                    break 
            if chosen is None:
                #All used: fall back to best candidate
                chosen = candidates[0]
            selected[tcol] = chosen
            used.add(chosen)

        # ensure mapping returned covers training columns; keep selected mapping as-is.
        if len(used) <4:
            pairs_sorted = sorted([(val, k1, k2) for (k1, k2), val in sse.items()], key=lambda x: x[0])
            for val, tcol, icol in pairs_sorted:
                if icol not in used:
                    used.add(icol)
                if len(used) >= 4:
                    break
        chosen_list = list(used)[:4]
        # ensure mapping returned covers training columns; keep selected mapping as-is.
        return selected
        
    def map_test_points(self, chosen_ideals: List[str], max_dev: Dict[Tuple[str, str], float], test_df: pd.DataFrame) -> pd.DataFrame:
        """
        For each test point, interpolate chosen ideals to test X and map the point
        when abs(y - yideal) <= max_dev_for_that_ideal * sqrt(2). If multiple ideals match,
        pick the one with smallest delta.
        Returns a DataFrame with columns ['X', 'Y', 'DeltaY', 'IdealFunc'].
        """
        tx = test_df['X'].values
        ty = test_df['Y'].values
        # For chosen ideals we might not have a single training assignment; derive threshold per chosen ideal.
        # Build threshold = max_dev(training_for_that_ideal) if assigned; else minimal across training columns.
        thresholds = {}
        # compute per-ideal maximum deviation over training functions
        per_ideal_maxdev = {}
        # Reconstruct max_dev dict keyed by (traincol, idealcol)
        # For simplicity: take min across training cols for ideals that are not assigned
        for ideal in chosen_ideals:
            vals = []
            for tcol in [c for c in self.train_df.columns if c != "X"]:
                vals.append(max_dev.get((tcol, ideal), np.nan))
            per_ideal_maxdev[ideal] = float(np.nanmin(vals))

        # Precompute ideal interpolations at test X
        ideal_at_test = {ideal: linear_interpolate(self.ideal_df['X'].values, self.ideal_df[ideal].values, tx) for ideal in chosen_ideals}

        results = []
        for i in range(len(tx)):
            xi = float(tx[i]); yi = float(ty[i])
            best = None
            for ideal in chosen_ideals:
                yideal = float(ideal_at_test[ideal][i])
                delta = abs(yi - yideal)
                thresh = per_ideal_maxdev[ideal] * math.sqrt(2)
                if delta <= thresh:
                    if best is None or delta < best['delta']:
                        best = {'ideal': ideal, 'delta': float(delta)}
            if best:
                results.append({'X': xi, 'Y': yi, 'DeltaY': best['delta'], 'IdealFunc': best['ideal']})
            else:
                results.append({'X': xi, 'Y': yi, 'DeltaY': None, 'IdealFunc': None})
        return pd.DataFrame(results)
    
    
class Visualizer:
    """Produces a Bokeh interactive visualization."""
    def __init__(self, train_df: pd.DataFrame, ideal_df: pd.DataFrame, mapped_df: pd.DataFrame, chosen_ideals: List[str]):
        self.train_df = train_df
        self.ideal_df = ideal_df
        self.mapped_df = mapped_df
        self.chosen_ideals = chosen_ideals
    def save(self, outfile: Path):
        output_file(str(outfile), title = "Least Squares Assignment Visualization")
        p = figure(title = "Training, Chosen ideal functions and Test mappings", 
                   width = 900, height = 600, x_axis_label = 'X', y_axis_label = 'Y',
                   tools = "pan, wheel_zoom, box_zoom, reset, save")
        # Training lines
        for tcol  in [c for c in self.train_df.columns if c != "X"]:
            p.line(self.train_df['X'].values, self.train_df[tcol].values, line_width = 2, legend_label = f'train{tcol}')
        # Chosen ideals dashed
        for ideal  in self.chosen_ideals:
            yvals = linear_interpolate(self.ideal_df['X'].values, self.ideal_df[ideal].values, self.train_df['X'].values)
            p.line(self.train_df['X'].values, yvals, line_width = 2, line_dash = 'dashed', legend_label = f'Ideal {ideal}')
        # Mapped/unmapped test points
        mapped = self.mapped_df[self.mapped_df['IdealFunc'].notna()]
        unmapped = self.mapped_df[self.mapped_df['IdealFunc'].isna()]
        
        if not mapped.empty:
            p.circle(mapped['X'], mapped['Y'], size = 8, legend_label = 'Mapped')
        if not unmapped.empty:
            p.cross(unmapped['X'], unmapped['Y'], size = 8, legend_label = 'Unmapped')
            
        hover = HoverTool(tooltips=[("X", "@x"), ("Y", "@y"), ("Ideal", "@IdealFunc"), ("Delta", "@DeltaY")])
        p.add_tools(hover)
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        
        save(p)
        logger.info(f"Saved visualization to {outfile}")
        
def main():
    base = Path(".")
    
    train_csv = base /  "train.csv"
    ideal_csv = base / "ideal.csv"
    test_csv = base /  "test.csv"
    db_file = base / "assignement.db"
    viz_file = base / "visualization.html"
    meta_file = base / "metadata.json"
    
    loader =  CSVLoader(train_csv, ideal_csv, test_csv)
    train_df, ideal_df, test_df = loader.load_all()
    
    db = DatabaseManager(db_file)
    db.save_dataframe(train_df, "training")
    db.save_dataframe(ideal_df, "ideal_functions")
    db.save_dataframe(test_df, "test_raw")
    
    matcher = IdealMatcher(train_df, ideal_df)
    sse, maxdev = matcher.compute_sse_and_maxdev()
    assigned = matcher.choose_four_ideals(sse)
    chosen_ideals = list({v for v in assigned.values()})[:4]
    if len(chosen_ideals) < 4:
        pairs = sorted([(val, 1, k2) for (k1, k2), val in sse.items()], key = lambda x: [0])
        for _, tcol, icol in pairs:
            if icol not in chosen_ideals:
                chosen_ideals.append(icol)
            if len(chosen_ideals)>= 4:
                break
                
    mapped_df = matcher.map_test_points(chosen_ideals, maxdev, test_df)
    db.save_dataframe(mapped_df, "test_mapped")
    metadata = {
        "selected_ideals": chosen_ideals,
        "assigned_training": assigned,
        "train_max_dev": {f"{tcol}| {icol}": maxdev[(tcol, icol)] for (tcol, icol) in maxdev}
    }
    meta_file.write_text(json.dumps(metadata, indent=2))
    
    viz = Visualizer(train_df, ideal_df, mapped_df, chosen_ideals)
    viz.save(viz_file)
    
    logger.info("All done. Outputs: %s, %s", db_file, viz_file)

if __name__ == "__main__":
    main()



    