import numpy
import carla

class CarlaEnv:
    def __init__(self, ip, port):
        self.client = carla.Client(ip, port)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
