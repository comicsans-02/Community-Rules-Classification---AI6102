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
    preds = []
    probs = []

    for text in texts:
        # Skip empty text
        if not isinstance(text, str) or text.strip() == "":
            preds.append(0)
            probs.append(0.0)
            continue

        # Safe call to FastText
        labels, p = model.predict(text, k=1)

        # Convert tuple → list
        labels = list(labels) if labels is not None else []

        labels = np.asarray(labels) if labels is not None else np.asarray([])
        p = np.asarray(p) if p is not None else np.asarray([])

        # Handle missing predictions
        if len(labels) == 0 or len(p) == 0:
            preds.append(0)
            probs.append(0.0)
            continue

        fasttext_label = labels[0]   # "__label__1", "__label__0"
        prob = float(p[0])

        # convert FastText label → 0/1
        pred = 1 if "__label__1" in fasttext_label else 0
        
        # probability of class 1
        prob_class1 = prob if pred == 1 else 1 - prob

        preds.append(pred)
        probs.append(prob_class1)

    return np.asarray(preds), np.asarray(probs)
