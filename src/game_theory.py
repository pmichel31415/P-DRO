import numpy as np


def fictitious_play(A, iterations=100):
    """Returns an approximate mixed nash equilibrium for the two player zero
    sum game with payoff matrix A.

    Specifically this will return two probability distributions p and q and a
    value  v such that

    p.dot(A).dot(q) >= p_.dot(A).dot(q) for any distribution p_ over rows
    p.dot(A).dot(q) <= p.dot(A).dot(q_) for any distribution q_ over columns

    and v = p.dot(A).dot(q)

    The code is adapted (numpifyed, python3-fied and commented) from:
    code.activestate.com/recipes/496825-game-theory-payoff-matrix-solver/

    Args:
        A (np.ndarray): Payoff matrix. Row player wants to maximize her
            payoff. Column player wants to minimize it.
        iterations (int, optional): Maximum number of iterations.
            Defaults to 100.

    Returns:
        tuple: (p, q, v)
    """
    'Return the oddments (mixed strategy ratios) for a given payoff matrix'
    m, n = A.shape
    # Cumulative payoffs
    row_cum_payoff = np.zeros(m)
    col_cum_payoff = np.zeros(n)
    # Number of times a pure strategy is played
    rowcnt = np.zeros(m)
    colcnt = np.zeros(n)
    for step in range(iterations):
        # Row player chooses historically best pure strategy
        # Identify all best strategies
        best_strats = (row_cum_payoff == row_cum_payoff.max()).astype(float)
        # Sample one at random
        active_row = np.random.choice(m, p=best_strats/best_strats.sum())
        # active_row = np.argmax(row_cum_payoff)
        # row player plays active_row strategy
        rowcnt[active_row] += 1
        # Update historical cumulative payoff for column player
        col_cum_payoff += A[active_row]
        # Column player chooses optimal strategy
        # Identify all best strategies
        best_strats = (col_cum_payoff == col_cum_payoff.min()).astype(float)
        # Sample one at random
        active_col = np.random.choice(n, p=best_strats/best_strats.sum())
        # active_col = np.argmin(col_cum_payoff)
        # Play this strategy
        colcnt[active_col] += 1
        # Update historical payoff for the row player
        row_cum_payoff += A[:, active_col]
    value_of_game = (max(row_cum_payoff) +
                     min(col_cum_payoff)) / 2.0 / iterations
    return rowcnt/rowcnt.sum(), colcnt/colcnt.sum(), value_of_game


if __name__ == "__main__":
    A = np.asarray([
        [1, 3, 1],
        [2, 1, 1]
    ])
    p, q, v = fictitious_play(A, 1000)
    print(p)
    print(q)
    print(A.dot(q))
    print(p.dot(A.dot(q)))
    print(v)
