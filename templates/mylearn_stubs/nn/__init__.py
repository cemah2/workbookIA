"""Neural-network building blocks — mylearn, chapter 16 (Feed-Forward Networks).

This sub-package holds, in pure NumPy, the fully connected networks you write
by hand in chapters 16 to 20, before switching to PyTorch: dense layers and
weight initialisation (``layers``, chapter 16), activation functions
(``activations``, chapter 17), losses and backpropagation (``backward``,
chapter 18), regularisation and normalisation (``regularization``, chapter 20).

This file deliberately imports nothing, so it never has to change when a new
module is added. Import what you need from each module explicitly, for example
``from mylearn.nn.layers import dense_forward`` (or, inside the package,
``from .nn.layers import dense_forward``).
"""
