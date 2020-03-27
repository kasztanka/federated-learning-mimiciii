class Metric:
    def __init__(self, name, function, use_soft):
        self.name = name
        self.function = function
        self.use_soft = use_soft
        self.scores = []
