<h1 align="center">Community Guidelines Violation Check: A Multi-Class Classification Problem</h1> 

A project that uses different ML and DL algorithms to determine whether a comment violates a particular Reddit community guideline.

## Key Files

- Data: ```data/raw/train.csv```, ```data/raw/test.csv```
- Processed: ```data/processed/train_cleaned.csv``` (already generated)
- Trained Model: ```notebooks/fasttext_baseline.bin``` (already trained)
- Source Modules: ```src/preprocessing.py```, ```src/eval.py```, ```src/models/fasttext.py```

## Algorithms Used
  
#### Virtual Environment Created

| Component      | Details                                  |
|----------------|------------------------------------------|
| Location       | ./venv                                   |
| NumPy version  | 1.26.4 (< 2.0, compatible with fasttext) |
| Jupyter kernel | "AI6102 (venv)"                          |

##### To Use

From terminal:
```source venv/bin/activate```

In Jupyter notebooks:
Select the kernel "AI6102 (venv)" from the kernel picker in your IDE or Jupyter.
