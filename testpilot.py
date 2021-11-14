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
import math
import random
import time

#enables a constant velocity on vehicle
#returns False to break the game loop
def moveVehicle(endTime, vehicle):
    if(int(time.time()) > endTime):
        return False

    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=math.cos(endTime - time.time()) ))
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
        print(world.get_map().name)
        if(world.get_map().name != "Carla/Maps/Town02_Opt"): #Town02_Opt
            world = client.load_world('Town02_Opt', map_layers=carla.MapLayer.NONE) 
        bpLibrary = world.get_blueprint_library()

        #picks a random vehicle from the library
        vehicleBP = random.choice(bpLibrary.filter("vehicle"))

        #picks a random spawn point from the given spawn points
        vehicleTransform = world.get_map().get_spawn_points()[80]

        #spawns a vehicleBP actor at vehicleTransform
        vehicle = world.spawn_actor(vehicleBP, vehicleTransform)
        actorList.append(vehicle)
        print("loaded vehicle %s at location (%s, %s)" % (vehicle.type_id, vehicleTransform.location.x, vehicleTransform.location.y))

        #determine time to kill this script
        startTime = int(time.time())
        endTime = startTime + 5

        print("begin steering tests:\n > steering back and forth")
        vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1.0))
        #game loop
        while(moveVehicle(endTime, vehicle)):
            pass

        print(" > turning left, full throttle")
        vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1.0))
        time.sleep(5)
        print(" > straight, full brake")
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
        time.sleep(5)
        print(" > turning right, full throttle")
        vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1.0))
        time.sleep(5)

    finally:
        print("destroying all actors")
        for actor in actorList: 
            actor.destroy() 
        print("closing testpilot")


if __name__ == '__main__':

    main()