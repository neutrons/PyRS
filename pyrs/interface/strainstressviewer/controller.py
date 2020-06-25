class Controller:
    def __init__(self, model):
        self._model = model

    def fileSelected(self, name, filename):
        print(f'{name=}, {filename=}')
        self._model.set_workspace(name.replace("ε", "e"), filename)

    def peakSelected(self, name):
        if name != "":
            self._model.selectedPeak = name

    def update_d0(self, d0):
        self._model.d0 = d0
