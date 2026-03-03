import numpy as np
import pandas as pd
from numpy.typing import NDArray

class Aircraft_parameter:
    def __init__(self, para_file : str) -> None:
        self.origin_para = self.read_csv(para_file)

    def read_csv(self, csv_file : str) -> NDArray:

        return

    

if __name__ == "__main__":
    air_para = 