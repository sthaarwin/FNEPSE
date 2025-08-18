import kagglehub
import pandas as pd
import pathlib as Path

def download_data(out_dir="data/raw/candle-stick-patterns"):
    
    downloaded_path = kagglehub.dataset_download(
        "mineshjethva/candle-stick-patterns"
    )

    out_path = Path.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for f in Path.Path(downloaded_path).glob("*.csv"):
        df = pd.read_csv(f)
        df.to_csv(out_path / f.name, index=False)
        print (f"Saved {f.name} to {out_path / f.name}")
    
    return out_path

def merge_patterns(data_dir="data/raw/candle-stick-patterns", out_csv="data/raw/merged-candle-stick-patterns.csv"):
    files = Path.Path(data_dir).glob("*.csv")

    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    dfs = []

    for f in files:
        pattern_name = f.stem
        df = pd.read_csv(f)
        df["pattern"] = pattern_name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(out_csv, index=False)
    print(f"Merged data saved to {out_csv}")
    return combined

if __name__ == "__main__":
    download_data()
    merge_patterns()
    print("Data download and merge completed.")