class Controller:
    def __init__(self, model):
        self._model = model

    def fileSelected(self, name, filename):
        self._model.set_workspace(name, filename)
        return getattr(self._model, name) is not None

    def peakSelected(self, name):
        if name != "":
            self._model.selectedPeak = name

    def update_d0(self, d0):
        self._model.d0 = d0

    def calculate_stress(self, stress_case, youngModulus, poissonsRatio):
        self._model.calculate_stress(stress_case.replace(' ', '-'),
                                     float(youngModulus),
                                     float(poissonsRatio))

    def validate_selection(self, direction, twoD):
        if twoD and direction == '33':
            return "Cannot plot peak parameter for unused 33 direction in 2D stress case"

        return self._model.validate_selection(direction)

    def validate_stress_selection(self, stress_case, youngModulus, poissonsRatio):
        errors = ""

        directions = ('11', '22', '33') if stress_case == 'diagonal' else ('11', '22')

        for direction in directions:
            valid = self._model.validate_selection(direction)
            if valid:
                errors += valid + '\n'

        try:
            float(youngModulus)
        except ValueError:
            errors += "Need to specify Young Modulus\n"

        try:
            float(poissonsRatio)
        except ValueError:
            errors += "Need to specify Poissons Ratio\n"

        if errors:
            return errors
        else:
            return None

    def write_stress_to_csv(self, filename):
        self._model.write_stress_to_csv(filename)
