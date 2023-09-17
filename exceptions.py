class SourceFilesMissing(Exception):
    def __init__(self, message=None):
        self.message = message
        super(SourceFilesMissing, self).__init__(self.message)

