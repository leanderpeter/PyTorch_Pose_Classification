
import sys
from rtpose_vgg import get_model

class posemodelloader:

    def __init__(self):
        super().__init__()
        self._model_path = None
        self._classifier = None
        self._input_size = None
        self._output_size = None

    def load_model(self):
        self._model = get_model()
        self._model.load_state_dict(torch.load(_model_path))

        #Switch between CUDA or CPU
        self._model = torch.nn.DataParallel(model).cuda()
        #model = torch.nn.DataParallel(model)
        self._model.float()
        self._model.eval()

        return self._model

