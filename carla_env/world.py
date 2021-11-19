from carla import MapLayer

class World():
    def __init__(self, client):
        self.world = client.get_world()
        if(self.world.get_map().name != "Carla/Maps/Town02_Opt"): #Town02_Opt
            self.world = client.load_world('Town02_Opt', map_layers=MapLayer.NONE)
            print("loading basic map:", self.world.get_map().name)
        self.map = self.get_map()
        self.actor_list = []

    def tick(self):
        for actor in list(self.actor_list):
            actor.tick()
        self.world.tick()

    def destroy(self):
        print("Destroying all spawned actors")
        for actor in list(self.actor_list):
            actor.destroy()

    def get_carla_world(self):
        return self.world

    def __getattr__(self, name):
        """Relay missing methods to underlying carla object"""
        return getattr(self.world, name)