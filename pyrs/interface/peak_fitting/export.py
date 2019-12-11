class Export:

    def __init__(self, parent=None):
        self.parent = parent

    def csv(self):

        print("project name is : {}".format(self.parent._project_name))


        pass
