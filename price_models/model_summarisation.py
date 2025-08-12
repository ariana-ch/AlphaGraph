import pandas as pd
from pathlib import Path
from typing import Optional


def summary(root_dir: Optional[Path] = None, version: str = 'test_summary.csv', last_N: int = 10) -> pd.DataFrame:
    root_dir =  root_dir or Path(__file__).parent / 'logs'
    paths = root_dir.rglob(version)
    data = []
    for p in paths:
        params = str(p.parent.parent).rsplit('/', 1)[-1]
        df = pd.read_csv(p, index_col=False)
        model = str(p).split('/')[-4]
        epochs = len(df)
        df = df[['Annualised Sharpe', 'Average Components']]
        df = df[-last_N:]
        data.append(
            df.apply({c: ['mean', 'std'] for c in df.columns}).values.flatten().tolist() + [epochs, params, model]
        )
    df = pd.DataFrame(data, columns=['Annualised Sharpe Mean', 'Average Components Mean', 'Annualised Sharpe Std',
                                     'Average Components Std', 'Epochs', 'Params', 'Model'])
    df.to_csv(root_dir.parent / f'summary.csv', index=False)
    return df

if __name__ == '__main__':
    summary()
    # Example usage:
    # df = todf(Path('/path/to/logs'), 'test_summary.csv', 5)
    # print(df)
    # df.plot()
    # plt.show()