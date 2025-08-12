import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def to_df(root_dir: Optional[Path] = None, version: str = 'test_summary.csv', last_N: int = 10, model: str = 'bidirectional_lstm') -> pd.DataFrame:
    root_dir =  root_dir or Path(__file__).parent / 'logs'
    paths = [p for p in root_dir.rglob(version) if model in str(p)]
    data = []
    for p in paths:
        params = str(p.parent.parent).rsplit('/', 1)[-1]
        df = pd.read_csv(p, index_col=False)
        epochs = len(df)
        df = df[['Annualised Sharpe', 'Average Components']]
        df = df[-10:]
        data.append([df['Annualised Sharpe'].mean(), df['Annualised Sharpe'].std(), df['Average Components'].mean(), df['Average Components'].std(), epochs, params, model])
    df = pd.DataFrame(data, columns=['Annualised Sharpe Mean', 'Annualised Sharpe Std', 'Average Components Mean', 'Average Components Std', 'Epochs', 'Params', 'Model'])
    df.to_csv(root_dir / f'{model}_summary.csv', index=False)
    return df

if __name__ == '__main__':
    to_df()
    # Example usage:
    # df = todf(Path('/path/to/logs'), 'test_summary.csv', 5)
    # print(df)
    # df.plot()
    # plt.show()