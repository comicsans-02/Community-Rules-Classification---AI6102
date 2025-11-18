import numpy as np

def write_fasttext_file(df, text_col="combined_text", label_col="rule_violation", out_path="fasttext_train.txt"):
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            text = str(row[text_col]).strip()
            label = int(row[label_col])

            if not text or text == "[SEP]" or len(text) < 3:
                continue

            text = text.replace("\n", " ")

            f.write(f"__label__{label} {text}\n")

def fasttext_predict(model, texts):
    """
    Returns prediction probabilities using the trained Fasttext model.
    """
    probs = []
    preds = []

    for text in texts:
        labels, p = model.predict(text, k=1)

        label_str = labels[0]
        prob = p[0]

        # predicted class
        pred = 1 if label_str == "__label__positive" else 0

        # probability of class "1"
        prob_class1 = prob if pred == 1 else (1 - prob)

        preds.append(pred)
        probs.append(prob_class1)

    return np.array(preds), np.array(probs)
