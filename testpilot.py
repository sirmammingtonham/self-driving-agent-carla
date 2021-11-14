#######|  find carla module  |#######################################################
import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


#######|  run the sim         |#######################################################
import carla
import random
import time

#enables a constant velocity on vehicle
#returns False to break the game loop
def moveVehicle(endTime, vehicle):
    if(int(time.time()) > endTime):
        return False

    vehicle.enable_constant_velocity(carla.Vector3D(17, 0, 0))
    return True


def main():
    actorList = []
    print("testpilot script started")
    try:
        #creates the client
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        #retrieve the world and blueprint list from the server
        world = client.get_world()
        bpLibrary = world.get_blueprint_library()

        #picks a random vehicle from the library
        vehicleBP = random.choice(bpLibrary.filter("vehicle"))

        #picks a random spawn point from the given spawn points
        vehicleTransform = random.choice(world.get_map().get_spawn_points())

        #spawns a vehicleBP actor at vehicleTransform
        vehicle = world.spawn_actor(vehicleBP, vehicleTransform)
        actorList.append(vehicle)
        print("loaded vehicle %s at location (%s, %s)" % (vehicle.type_id, vehicleTransform.location.x, vehicleTransform.location.y))

        #determine time to kill this script
        startTime = int(time.time())
        endTime = startTime + 5

        #game loop
        while(moveVehicle(endTime, vehicle)):
            pass

    finally:
        print("destroying all actors")
        for actor in actorList: 
            actor.destroy() 
        print("closing testpilot")


if __name__ == '__main__':

    main()