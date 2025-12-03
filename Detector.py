class DetectorBase:
    def __init__(self):
        self.image = None;
    def getDepth(self, x, y, properties) -> str:
        raise "Перезапиши цю функцію!"