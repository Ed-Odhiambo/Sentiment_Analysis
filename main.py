import tkinter as tk

import joblib
import matplotlib
import nltk
import numpy as np
import pandas as pd
import sklearn
from PIL import Image

"""Main entry point for the sentiment analysis application."""

from src.gui import run_gui


if __name__ == "__main__":
    run_gui()