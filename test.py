import compare


for model in compare.available_models:
    c = compare.TextComparator(model)

    A = "A"
    B = "B"

    P, R, F1 = c.score(A, B)
    print(P.item())