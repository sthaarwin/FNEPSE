import kagglehub
import pandas as pd
import pathlib as Path

def download_data(out_dir="data/raw/stock-data"):
    downloaded_path = kagglehub.dataset_download(
        "borismarjanovic/price-volume-data-for-all-us-stocks-etfs"
    )

    out_path = Path.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    txt_files = list(Path.Path(downloaded_path).glob("**/*.txt"))
    print(f"Found {len(txt_files)} txt files")
    
    processed = 0
    for f in txt_files[:30]:  
        try:
            df = pd.read_csv(f, sep=',')
            required_cols = ['Open', 'High', 'Low', 'Close']
            if all(col in df.columns for col in required_cols):
                df = df.rename(columns={
                    'Open': 'open', 
                    'High': 'high', 
                    'Low': 'low', 
                    'Close': 'close'
                })
                
                df = df.dropna(subset=['open', 'high', 'low', 'close'])
                
                if len(df) > 100:
                    stock_symbol = f.stem.split('.')[0] 
                    df['pattern'] = stock_symbol
                    
                    output_file = out_path / f"{stock_symbol}.csv"
                    df.to_csv(output_file, index=False)
                    print(f"Saved {stock_symbol}: {len(df)} rows")
                    processed += 1
                
        except Exception as e:
            print(f"Error processing {f.name}: {e}")
    
    print(f"Successfully processed {processed} stock files")
    return out_path

def merge_patterns(data_dir="data/raw/stock-data", out_csv="data/raw/merged-stock-data.csv"):
    files = list(Path.Path(data_dir).glob("*.csv"))
    
    if not files:
        print(f"No CSV files found in {data_dir}")
        return None
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                dfs.append(df)
                print(f"Loaded {f.name}: {len(df)} rows, pattern: {df['pattern'].iloc[0]}")
        except Exception as e:
            print(f"Error loading {f.name}: {e}")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        Path.Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_csv, index=False)
        print(f"Merged data saved to {out_csv}")
        print(f"Total rows: {len(combined)}")
        print(f"Unique patterns: {combined['pattern'].nunique()}")
        print(f"Pattern distribution:\n{combined['pattern'].value_counts()}")
        return combined
    else:
        print("No valid data found to merge")
        return None

if __name__ == "__main__":
    download_data()
    merge_patterns()
    print("Data download and merge completed.")