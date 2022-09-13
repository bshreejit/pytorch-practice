# Steps for Linear Regression

## From Scratch (initial_mse.py, gradient_descent.py)

1) Find out the initial mean squared error i.e, loss (initial_mse.py)
    At the initial instance the loss or mse may be way over the target/expectation, so we need the model to learn , i.e, to figure out the set of weights and biases by looking at the training data to make the accurate predictions for the new data
    This is done by adjusting the weights slightly many times to make better predictions, using an optimization technique called Gradient Descent.

2) Gradient Descent Algorithm (gradient_descent.py)
    In this step we adjusts the weights and the biases to reduce the loss iteratively
        P(n+1) = p(n) - (l.r)*grad(p(n))
        Here, p(n) -> initial point
        p(n+1) -> next point
        l.r -> learning rate -> It has a strong influence on the performance
        grad(p(n)) -> gradient at the current position

    So, this algorithm iteratively calculates the next point using gradient at the current position, scales it (by a learning rate) and subtract obtained value from the current position.

## Using pyTorch built-ins (using_builtins.py)


