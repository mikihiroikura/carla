from copy import deepcopy
import carla
import random
import queue
import numpy
import cv2
import argparse

# Class id
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

# Detection area
xrange = [10.0, 25.0]
yrange = [-4.0, 4.0]

# Spawn position
spawn_position_candidate = [[-166, 1873, 488], [-162, 1874, 488], [-158, 1875, 488],
                            [-155, 1875.5, 488], [-151, 1876.5, 488]]
random.seed()   # Initialize random

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * numpy.tan(fov * numpy.pi / 360.0))
    K = numpy.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = numpy.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = numpy.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = numpy.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]

def transformEvent2Img(eventArray):
    eventImgList = eventArray.to_image()
    image_heigth = eventArray.height
    image_width = eventArray.width
    eventImg = numpy.full((image_heigth, image_width, 3), (0, 0, 0), dtype=numpy.uint8)
    pixel_id = 0
    for i in range(image_heigth):
        for j in range(image_width):
            eventImg[i, j, 0] = eventImgList[pixel_id].b
            eventImg[i, j, 1] = eventImgList[pixel_id].g
            eventImg[i, j, 2] = eventImgList[pixel_id].r
            pixel_id = pixel_id + 1
    return eventImg

def main(args):
    ################
    # Setup
    ################
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    # Set dvs and rgb camera positions
    cam_posinit = [2.5, 0, 6, 0, 0, 0]
    if args.campos is not None:
        cam_posinit = args.campos

    # Spawn dvs camera
    dvs_bp = bp_lib.find('sensor.camera.dvs')
    dvs_bp.set_attribute("image_size_x", str(640))
    dvs_bp.set_attribute("image_size_y", str(480))
    dvs_bp.set_attribute("fov", str(105))
    dvs_location = carla.Location(cam_posinit[0], cam_posinit[1], cam_posinit[2])
    dvs_rotation = carla.Rotation(cam_posinit[3], cam_posinit[4], cam_posinit[5])
    dvs_transform = carla.Transform(dvs_location, dvs_rotation)
    stable_dvs = world.spawn_actor(dvs_bp, dvs_transform)

    # Debug: Spawn rgb camera
    rgb_bp = bp_lib.find('sensor.camera.rgb')
    rgb_bp.set_attribute("image_size_x", str(640))
    rgb_bp.set_attribute("image_size_y", str(480))
    rgb_bp.set_attribute("fov", str(105))
    rgb_location = carla.Location(cam_posinit[0], cam_posinit[1], cam_posinit[2])
    rgb_rotation = carla.Rotation(cam_posinit[3], cam_posinit[4], cam_posinit[5])
    rgb_transform = carla.Transform(rgb_location, rgb_rotation)
    stable_rgb = world.spawn_actor(rgb_bp, rgb_transform)

    # Set up the simulator in synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Create a queue to store and retrieve the dvs data
    dvs_queue = queue.Queue()
    stable_dvs.listen(dvs_queue.put)

    # Debug: Create a queue to store and retriece the rgb data
    rgb_queue = queue.Queue()
    stable_rgb.listen(rgb_queue.put)

    # Get the world to dvs camera matrix
    world_2_dvs = numpy.array(stable_dvs.get_transform().get_inverse_matrix())

    # Get the attributes from the camera
    image_w = dvs_bp.get_attribute("image_size_x").as_int()
    image_h = dvs_bp.get_attribute("image_size_y").as_int()
    fov = dvs_bp.get_attribute("fov").as_float()

    # Calculate the camera projection matrix to project from 3D -> 2D
    K = build_projection_matrix(image_w, image_h, fov)

    # Remember the edge pairs
    edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

    # Set parameter for spawining vehicles
    spawn_deltaseconds = 10.0  # Unit [s]
    spawn_loopcnt = int(spawn_deltaseconds / settings.fixed_delta_seconds)
    spawn_posid = 0

    # Place spectator on dvs camera
    spectator = world.get_spectator()
    spectator.set_transform(stable_dvs.get_transform())

    # Debug: Write detection area
    detectionArea = []
    for xarea in xrange:
        for yarea in yrange:
            pos = carla.Location(xarea, yarea, 0.0)
            p = get_image_point(pos, K, world_2_dvs)
            detectionArea.append(p)
    # from IPython.terminal import embed
    # ipshell = embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())

    ################
    # Loop
    ################
    while True:
        # Retrieve and convert events to img
        world.tick()
        dvs_events = dvs_queue.get()
        dvs_img = transformEvent2Img(dvs_events)
        dvs_img_saved = deepcopy(dvs_img)

        # Debug: Retrieve and reshape the rgb image
        image = rgb_queue.get()
        img = numpy.reshape(numpy.copy(image.raw_data), (image.height, image.width, 4))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # from IPython.terminal import embed
        # ipshell = embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())

        cv2.line(img, (int(detectionArea[0][0]),int(detectionArea[0][1])), (int(detectionArea[1][0]),int(detectionArea[1][1])), (0,0,255), 1)
        cv2.line(img, (int(detectionArea[1][0]),int(detectionArea[1][1])), (int(detectionArea[3][0]),int(detectionArea[3][1])), (0,0,255), 1)
        cv2.line(img, (int(detectionArea[3][0]),int(detectionArea[3][1])), (int(detectionArea[2][0]),int(detectionArea[2][1])), (0,0,255), 1)
        cv2.line(img, (int(detectionArea[2][0]),int(detectionArea[2][1])), (int(detectionArea[0][0]),int(detectionArea[0][1])), (0,0,255), 1)

        ###############
        # Spawn vehicles
        ###############
        # from IPython.terminal import embed
        # ipshell = embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())
        if dvs_events.frame % spawn_loopcnt == 0:
            # Randomize spawn position and time
            if args.randomspawn:
                spawn_posid_next = random.randint(0, len(spawn_position_candidate) - 1)
                while spawn_posid == spawn_posid_next:
                    spawn_posid_next = random.randint(0, len(spawn_position_candidate) - 1)
                spawn_posid = spawn_posid_next
                spawn_deltaseconds = float(random.randint(3, 10))
                spawn_loopcnt = int(spawn_deltaseconds / settings.fixed_delta_seconds)
            else:
                spawn_posid = (dvs_events.frame // spawn_loopcnt) % len(spawn_position_candidate)

            # Spawn vehicle
            vehicle_bp = random.choice(bp_lib.filter('vehicle'))
            spawn_pos = spawn_position_candidate[spawn_posid]
            spawn_transform = carla.Transform(carla.Location(spawn_pos[0], spawn_pos[1], spawn_pos[2]), carla.Rotation(-19.44432, 103.0468, 0))
            npc = world.try_spawn_actor(vehicle_bp, spawn_transform)
            if npc:
                npc.set_autopilot(True)
        
        ################
        # Create label text
        ################
        label_results = ''
        for npc in world.get_actors().filter('*vehicle*'):
            bb = npc.bounding_box
            bb_location = npc.get_transform().location
            if bb_location.x > xrange[0] and bb_location.x < xrange[1] and bb_location.y > yrange[0] and bb_location.y < yrange[1]:
                # from IPython.terminal import embed
                # ipshell = embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())

                verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                x_max = -10000
                x_min = 10000
                y_max = -10000
                y_min = 10000

                for vert in verts:
                    p = get_image_point(vert, K, world_2_dvs)
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

                cv2.rectangle(dvs_img, (int(x_min),int(y_min)), (int(x_max),int(y_max)), (0,255,0, 255), 1)

                # Check whether events are in bounding box
                xevents = numpy.array(dvs_events.to_array_x())
                yevents = numpy.array(dvs_events.to_array_y())
                eventsInBB = (xevents < x_max) * (xevents > x_min) * (yevents < y_max) * (yevents > y_min)
                if numpy.any(eventsInBB) is False:
                    continue

                # Write label data
                label_id = -1
                if npc.type_id in car:
                    label_id = 0
                elif npc.type_id in truck:
                    label_id = 1
                elif npc.type_id in motorbicycle:
                    label_id = 2
                elif npc.type_id in bicycle:
                    label_id = 3
                if label_id == -1:
                    continue
                center_x_percetage = (x_min + x_max) / 2.0 / img.shape[1]
                center_y_percetage = (y_min + y_max) / 2.0 / img.shape[0]
                width_percetage = (x_max - x_min) / img.shape[1]
                height_percetage = (y_max - y_min) / img.shape[0]
                label_result = '%01d %.6f %.6f %.6f %.6f' % (label_id, center_x_percetage, center_y_percetage, width_percetage, height_percetage)
                print(label_result)
                label_results += label_result + '\n'

        ################
        # Show event frames
        ################
        display_imgs = numpy.concatenate((img, dvs_img), axis=1)
        cv2.imshow('Left (RGB) and Right (DVS) frames', display_imgs)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('d'):
            from IPython.terminal import embed
            ipshell = embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())


        ################
        # Save dataset
        ################
        if args.savedataset and len(label_results) > 0:
            # Event images
            dvspath = args.path + 'output/%06d.png' % dvs_events.frame
            cv2.imwrite(dvspath, dvs_img_saved)

            # Raw event stream
            if args.eventstream:
                eventpath = args.path + 'events/%06d.txt' % dvs_events.frame
                x_events = dvs_events.to_array_x()
                y_events = dvs_events.to_array_y()
                t_events = dvs_events.to_array_t()
                p_events = dvs_events.to_array_pol()
                events = numpy.array([t_events, x_events, y_events, p_events]).transpose()
                numpy.savetxt(eventpath, events)

            # Label
            labelpath = args.path + 'labels/%06d.txt' % dvs_events.frame
            with open(labelpath, 'w') as f:
                f.write(label_results)
            f.close()

    ################
    # Loop end process
    ################
    cv2.destroyAllWindows()

    # Destroy actors
    stable_dvs.stop()
    stable_dvs.destroy()

    stable_rgb.stop()
    stable_rgb.destroy()

    for npc in world.get_actors().filter('*vehicle*'):
        npc.destroy()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--path',
        default='/home/mikura/Github/carla/tutorial/',
        help='Path for saving dataset (imgs and labels)'
    )
    argparser.add_argument(
        '--savedataset',
        default=False,
        action='store_true',
        help='Flag for saving dataset'
    )
    argparser.add_argument(
        '--eventstream',
        default=False,
        action='store_true',
        help='Flag for saving raw event stream'
    )
    argparser.add_argument(
        '--campos',
        nargs='+',
        type=float,
        default=[-166.166229, 1912.735474, 498.677490, -19.44432, 103.046799, 1.0], # providentia++ camera default position
        help='Set dvs camera position (x, y, z[m], pitch, yaw, roll[deg])'
    )
    argparser.add_argument(
        '--randomspawn',
        default=False,
        action='store_true',
        help='Spawn vehicles randomly (related to time and position)'
    )
    args = argparser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone with stable dvs and bounding boxes.')