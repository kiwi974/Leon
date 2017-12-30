from threading import Thread
import classement


class ClassementParallel(Thread):

    """Thread chargé de classer des variables et la variable sonde"""
    def __init__(self, y, variables, sonde,ordre):
        Thread.__init__(self)
        self.y = y
        self.variables = variables
        self.sonde = sonde
        self.ordre = ordre

    def run(self):
        """Code à exécuter pendant l'exécution du thread."""
        self.ordre = classement.classer(self.y, self.variables, self.sonde)