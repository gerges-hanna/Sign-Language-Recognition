"""
@author: Gerges_Hanna
"""

import enum
from src.Models_module.Models_Types.GRU.GRU import GRU
from src.Models_module.Models_Types.LSTM.LSTM import LSTM
from src.Models_module.Models_Types.BILSTM.BILSTM import BILSTM
from src.Models_module.Models_Types.MLP.MLP import MLP
from src.Models_module.Models_Types.CNN.CNN import CNN


class Models(enum.Enum):
    CNN = CNN()
    MLP = MLP()
    BILSTM = BILSTM()
    LSTM = LSTM()
    GRU = GRU()
