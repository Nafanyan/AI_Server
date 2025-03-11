class Result:
    def __init__(self, result, errors):
        self.isSuccess = errors is None
        self.errors = errors

        if self.isSuccess:
            self.result = result
            return
        
        self.result = None
            