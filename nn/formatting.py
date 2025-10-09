def format_results(X, y_true, y_pred, num_samples=10):
    print("\nSample results (first 10):")
    print("-" * 50)
    print(f"{'Input':>15} | {'True':>5} | {'Network':>7} | {'Correct?':>9}")
    print("-" * 50)
    correct = 0
    for i in range(min(num_samples, X.shape[1])):
        x_vals = ", ".join([f"{v:.2f}" for v in X[:, i]])
        is_correct = "YES" if int(y_true[0, i]) == int(y_pred[0, i]) else "NO"
        if is_correct == "YES":
            correct += 1
        print(f"{'[' + x_vals + ']':>15} | {int(y_true[0, i]):>5} | {int(y_pred[0, i]):>7} | {is_correct:>9}")
    print("-" * 50)
    print(f"Correct predictions: {correct}/{num_samples}")
    acc = (y_pred == y_true).mean()
    print(f"\nNetwork accuracy: {acc*100:.2f}%")
