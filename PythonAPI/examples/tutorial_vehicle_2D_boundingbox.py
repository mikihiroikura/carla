from cProfile import label
import carla
import math
import random
import time
import queue
import numpy as np
import cv2

car = ['vehicle.audi.a2', 'vehicle.audi.etron', 'vehicle.audi.tt', 'vehicle.bmw.grandtourer', 'vehicle.chevrolet.impala',
       'vehicle.citroen.c3', 'vehicle.dodge.charger_2020', 'vehicle.dodge.charger_police', 'vehicle.dodge.charger_police_2020',
       'vehicle.ford.crown', 'vehicle.ford.mustang', 'vehicle.jeep.wrangler_rubicon', 'vehicle.lincoln.mkz_2017', 
       'vehicle.lincoln.mkz_2020', 'vehicle.mercedes.coupe', 'vehicle.mercedes.coupe_2020', 'vehicle.micro.microlino', 
       'vehicle.mini.cooper_s', 'vehicle.mini.cooper_s_2021', 'vehicle.nissan.micra', 'vehicle.nissan.patrol', 'vehicle.nissan.patrol_2021', 
       'vehicle.seat.leon', 'vehicle.tesla.cybertruck', 'vehicle.tesla.model3', 'vehicle.toyota.prius']
truck = ['vehicle.carlamotors.carlacola', 'vehicle.carlamotors.firetruck', 'vehicle.ford.ambulance', 'vehicle.mercedes.sprinter', 
         'vehicle.volkswagen.t2', 'vehicle.volkswagen.t2_2021']
motorbicycle = ['vehicle.harley-davidson.low_rider', 'vehicle.kawasaki.ninja', 'vehicle.vespa.zx125', 'vehicle.yamaha.yzf']
bicycle = ['vehicle.bh.crossbike', 'vehicle.diamondback.century', 'vehicle.gazelle.omafiets']

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]


client = carla.Client('localhost', 2000)
world  = client.get_world()
bp_lib = world.get_blueprint_library()

# Get the map spawn points
spawn_points = world.get_map().get_spawn_points()

# spawn vehicle
vehicle_bp =bp_lib.find('vehicle.lincoln.mkz_2020')
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

# spawn camera
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_init_trans = carla.Transform(carla.Location(z=2))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
vehicle.set_autopilot(True)

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Create a queue to store and retrieve the sensor data
image_queue = queue.Queue()
camera.listen(image_queue.put)

# Get the world to camera matrix
world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

# Get the attributes from the camera
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()
fov = camera_bp.get_attribute("fov").as_float()

# Calculate the camera projection matrix to project from 3D -> 2D
K = build_projection_matrix(image_w, image_h, fov)

# Set up the set of bounding boxes from the level
# We filter for traffic lights and traffic signs
bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
bounding_box_set.extend(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))

# Remember the edge pairs
edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

# Add other vehicles
for i in range(50):
    vehicle_bp = random.choice(bp_lib.filter('vehicle'))
    npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    if npc:
        npc.set_autopilot(True)

# Loop
while True:

    # Retrieve and reshape the image
    world.tick()
    image = image_queue.get()

    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

    # Get the camera matrix 
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    # Save images
    imgpath = '/home/mikura/Github/carla/tutorial/output/%06d.png' % image.frame
    image.save_to_disk(imgpath)

    # Create and open label txt data
    labelpath = '/home/mikura/Github/carla/tutorial/labels/%06d.txt' % image.frame
    with open(labelpath, 'w') as f:

        for npc in world.get_actors().filter('*vehicle*'):

            # Filter out the ego vehicle
            if npc.id != vehicle.id:

                bb = npc.bounding_box
                dist = npc.get_transform().location.distance(vehicle.get_transform().location)

                # Filter for the vehicles within 50m
                if dist < 50:

                # Calculate the dot product between the forward vector
                # of the vehicle and the vector between the vehicle
                # and the other vehicle. We threshold this dot product
                # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                    forward_vec = vehicle.get_transform().get_forward_vector()
                    ray = npc.get_transform().location - vehicle.get_transform().location

                    if forward_vec.dot(ray) > 1:
                        # p1 = get_image_point(bb.location, K, world_2_camera)
                        verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                        x_max = -10000
                        x_min = 10000
                        y_max = -10000
                        y_min = 10000

                        for vert in verts:
                            p = get_image_point(vert, K, world_2_camera)
                            # Find the rightmost vertex
                            if p[0] > x_max:
                                x_max = p[0]
                            # Find the leftmost vertex
                            if p[0] < x_min:
                                x_min = p[0]
                            # Find the highest vertex
                            if p[1] > y_max:
                                y_max = p[1]
                            # Find the lowest  vertex
                            if p[1] < y_min:
                                y_min = p[1]

                        cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (0,0,255, 255), 1)
                        cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
                        cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,0,255, 255), 1)
                        cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,0,255, 255), 1)

                        # Write label data
                        # from IPython.terminal import embed
                        # ipshell = embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())
                        label_id = -1
                        if npc.type_id in car:
                            label_id = 0
                        elif npc.type_id in truck:
                            label_id = 1
                        elif npc.type_id in motorbicycle:
                            label_id = 2
                        elif npc.type_id in bicycle:
                            label_id = 3
                        center_x_percetage = (x_min + x_max) / 2.0 / img.shape[1]
                        center_y_percetage = (y_min + y_max) / 2.0 / img.shape[0]
                        width_percetage = (x_max - x_min) / img.shape[1]
                        height_percetage = (y_max - y_min) / img.shape[0]
                        label_result = '%01d %.6f %.6f %.6f %.6f' % (label_id, center_x_percetage, center_y_percetage, width_percetage, height_percetage)
                        print(label_result)
                        f.write(label_result + '\n')

        f.close()

        cv2.imshow('ImageWindowName',img)
        if cv2.waitKey(1) == ord('q'):
            break
cv2.destroyAllWindows()