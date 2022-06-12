class TextureFittingCrtl:
    def __init__(self, peak_fit_model):
        self._model = peak_fit_model

    def load_projectfile(self, filename):
        self._model.load_hidra_project_file(filename)
        # return getattr(self._model) is not None

    def peakSelected(self, name):
        if name != "":
            self._model.selectedPeak = name

    def calculate_stress(self, stress_case, youngModulus, poissonsRatio, d0):
        self._model.calculate_stress(stress_case.replace(' ', '-'),
                                     float(youngModulus),
                                     float(poissonsRatio),
                                     d0)

    def get_log_plot(self, xname, yname):

        if xname == "sub-runs":
            xdata = self._model.sub_runs
        else:
            xdata = self._model.ws.get_sample_log_values(xname)

        if yname == "sub-runs":
            ydata = self._model.sub_runs
        else:
            ydata = self._model.ws.get_sample_log_values(yname)

        return xdata, ydata

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

    def write_stress_to_csv(self, filename, detailed):
        self._model.write_stress_to_csv(filename, detailed)

    def save(self, filename):
        self._model.to_json(filename)

    def load(self, filename):
        self._model.from_json(filename)
