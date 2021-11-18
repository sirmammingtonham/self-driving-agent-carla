#######|  setup  |#############################################################################
###############################################################################################

import glob
import os
import sys

#find carla module
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import carla
from carla import ColorConverter as cc
import math
import numpy as np
import random
import pygame
import time
import weakref
import cv2

IM_WIDTH = 640
IM_HEIGHT = 480

#######|  manage the view window  |############################################################
###############################################################################################

#the camera whose output will be displayed in a secondary window, giving a third-person view of the agent
#this class is an edited-down verson of CARLA's manual_control.py (find at carla/PythonAPI/examples)
class TPCamera(object):
    #initializes the camera sensor used for a third-person view
    def __init__(self, vehicle):
        self.relTransform = None
        self.sensor = None
        self.surface = None
        self.vehicle = vehicle
        world = self.vehicle.get_world()

        if not self.vehicle.type_id.startswith("walker.pedestrian"):
            self.relTransform = (carla.Transform(
                carla.Location(x=-3*(0.5+self.vehicle.bounding_box.extent.x), y=0, z=2.25*(0.5+self.vehicle.bounding_box.extent.z))))
        else:
            self.relTransform = (carla.Transform(carla.Location(x=-1.5, z=1.7)))

        sensorBP = world.get_blueprint_library().find("sensor.camera.rgb")
        sensorBP.set_attribute("image_size_x", "1280")
        sensorBP.set_attribute("image_size_y", "720")
        sensorBP.set_attribute("fov", "110")

        self.sensor = world.spawn_actor(
            sensorBP, self.relTransform, attach_to=self.vehicle, attachment_type = carla.AttachmentType.Rigid)

        #each frame, the sensor calls parseImage on image, which is the sensor's output
        weakSelf = weakref.ref(self)
        self.sensor.listen(lambda image: TPCamera.parseImage(weakSelf, image))

    #parse the image coming from sensor.listen into self.surface
    @staticmethod
    def parseImage(weak_self, image):
        self = weak_self()
        if not self:
            return

        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    #takes the surface (computed in parseImage) and blits it onto the pygame display
    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    #returns a reference to self.sensor
    def getSensor():
        return self.sensor

class frontCamera:
    #initializes the camera sensor used for a first-person view
    def __init__(self, vehicle):
        self.relTransform = None
        self.sensor = None
        self.surface = None
        self.vehicle = vehicle
        world = self.vehicle.get_world()

        if not self.vehicle.type_id.startswith("walker.pedestrian"):
            self.relTransform = (carla.Transform(
                carla.Location(x=2.5*(0.5+self.vehicle.bounding_box.extent.x), y=0, z=0.7*(0.5+self.vehicle.bounding_box.extent.z))))
        else:
            self.relTransform = (carla.Transform(carla.Location(x=2.5, z=0.7)))

        sensorBP = world.get_blueprint_library().find("sensor.camera.rgb")
        sensorBP.set_attribute('image_size_x', f'{IM_WIDTH}')
        sensorBP.set_attribute('image_size_y', f'{IM_HEIGHT}')
        sensorBP.set_attribute("fov", "110")

        self.sensor = world.spawn_actor(
            sensorBP, self.relTransform, attach_to=self.vehicle, attachment_type = carla.AttachmentType.Rigid)

        #each frame, the sensor calls parseImage on image, which is the sensor's output
        weakSelf = weakref.ref(self)
        self.sensor.listen(lambda image: TPCamera.parseImage(weakSelf, image))
    #parse the image coming from sensor.listen into self.surface
    @staticmethod
    def parseImage(weak_self, image):
        elf = weak_self()
        if not self:
            return

        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    #takes the surface (computed in parseImage) and blits it onto the pygame display
    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    #returns a reference to self.sensor
    def getSensor():
        return self.sensor


#initializes and returns the display of the testpilot
def initWindow():
    #initialize pygame
    pygame.init()

    #create the display
    display = pygame.display.set_mode(
            (1280, 720),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
    display.fill((0,0,0))
    pygame.display.set_caption("Agent View")
    pygame.display.flip()

    return display


#enables a constant velocity on vehicle
#returns False to break the game loop
def moveVehicle(endTime, vehicle):
    if(int(time.time()) > endTime):
        return False

    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=math.cos(endTime - time.time()) ))
    return True


#######|  run the sim  |############################################################################
####################################################################################################

def main():
    actorList = []
    print("testpilot script started")
    try:
        #creates the client
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        #retrieve the world and blueprint list from the server
        world = client.get_world() 
        if(world.get_map().name != "Carla/Maps/Town02_Opt"): #Town02_Opt
            world = client.load_world('Town02_Opt', map_layers=carla.MapLayer.NONE)
            print("loading basic map:", world.get_map().name)
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
        endTime = startTime + 30

        display = initWindow()
        tpCam = TPCamera(vehicle)
        actorList.append(tpCam.sensor)

        tp2 = frontCamera(vehicle)
        actorList.append(tp2.sensor)

        #game loop
        world.wait_for_tick()
        clock = pygame.time.Clock()
        while(True):
            clock.tick_busy_loop(60)
            if(not moveVehicle(endTime, vehicle)):
                break
            tpCam.render(display)    
            tp2.render(display)        
            pygame.display.flip()

        #print("begin steering tests:\n > steering back and forth")
        #vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1.0))
        #print(" > turning left, full throttle")
        #vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1.0))
        #time.sleep(5)
        #print(" > straight, full brake")
        #vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
        #time.sleep(5)
        #print(" > turning right, full throttle")
        #vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1.0))
        #time.sleep(5)

    finally:
        print("destroying all actors")
        for actor in actorList: 
            print("destroying one")
            actor.destroy() 

        print("quitting pygame")
        pygame.quit()


if __name__ == '__main__':
    main()