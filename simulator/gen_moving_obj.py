import os
import argparse
import csv
import cv2
import numpy as np
import pybullet
import uuid
import pdb
from PIL import Image
import time
import generate_urdf as urdf
import math

class ImageGenerator:
    image_size = (256, 256)  # [px]

    camera_distance = 18.0  # [m]
    pixel_size = 3.25  # [px/m]

    rotation_distance = 2 * camera_distance  # [m]
    object_types = ['box', 'cylinder', 'sphere']

    def __init__(self, data_directory, image_directory, rollout_len, segmented_directory, objects_directory='/tmp/generated'):
        self.data_dir = data_directory
        self.objects_directory = objects_directory
        self.rollout_len = rollout_len
        self.image_directory = image_directory
        self.segm_directory = segmented_directory

        # Start pybullet 'DIRECT'ly without GUI
        pybullet.connect(pybullet.GUI)
        pybullet.setGravity(0, 0, -1000.0)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, 0)


    def generate_random_object(self):
        # Create and save random object
        object_type = np.random.choice(self.object_types)
        if object_type == 'cylinder':
            urdf_string, params = urdf.generate_cylinder()
            height = params[1]
            params = np.multiply(params, 100)
            file_name = 'cylinder_r_{}_h_{}.urdf'.format(int(params[0]), int(params[1]))
        elif object_type == 'box':
            urdf_string, params = urdf.generate_box()
            height = params[2]
            params = np.multiply(params, 100)
            file_name = 'box_x_{}_y_{}_z_{}.urdf'.format(int(params[0]), int(params[1]), int(params[2]))
        elif object_type == 'sphere':
            urdf_string, params = urdf.generate_sphere()
            height = 2 * params[0]
            params = np.multiply(params, 100)
            file_name = 'sphere_r_{}.urdf'.format(int(params[0]))

        return urdf_string, file_name, height

    def generate_simulated_images_for_random_scene(self):
        #pybullet.loadURDF('objects/plane.urdf', [0, 0, 0], useFixedBase=True)
        num_objects = np.random.choice([1, 2, 3])
        for i in range(num_objects):
            collision = False
            urdf_string, file_name, height = self.generate_random_object()
            file_path = os.path.join(self.objects_directory, file_name)

            with open(file_path, 'w') as file:
                file.write(urdf_string)

            position = [0.0, 0.0, 0.0]
            #position[0] = np.random.uniform(-6,6)
            #position[1] = np.random.uniform(-6,6)
            position[2] = height / 2  # insert the object at half object height so it does not collide with the plane

            orientation = [0.0, 0.0, 0.0]
            #orientation[0] = np.random.uniform(0, 360)
            #orientation[1] = np.random.uniform(0, 360)
            # orientation[2] = np.random.uniform(0, 360)


            if i > 0:
                distance = math.sqrt(np.sum((np.array(position)-old_position)**2))
                if distance < (height+old_height)/2:
                    collision = True
            old_position = np.array(position)
            old_height = height
            if not collision:
                model_id = pybullet.loadURDF(file_path, position, pybullet.getQuaternionFromEuler(orientation))
        # camera settings

        camera_position = np.random.uniform(0.0, 1.0, size=(3))
        camera_orientation = [0.0, 0.0, 0.0]  # pitch, roll, yaw [rad]
        camera_field_of_view = 60

        near_plane = 1.0
        far_plane = 100.0
        orientation = [0.0, 0.0, 0.0]
        orientation[0] = np.random.uniform(0, 360)
        orientation[1] = np.random.uniform(0, 360)


        #pybullet.resetBasePositionAndOrientation(model_id,  np.random.uniform(0.0, 1.0, size=(3)), pybullet.getQuaternionFromEuler(orientation))
        for i in range(1):
            pybullet.stepSimulation()

        depth_images = []
        segm_images = []
        first = True
        yaw = camera_orientation[2]
        pitch = -np.pi / 2
        roll = camera_orientation[1]
        pose_deltas = []
        start_time = time.time()
        dt = 0.05 #measured
        COEFF = 1.5
        frame = np.random.choice([7.0, 8.0, 11])
        print("Frame size: {}".format(frame))
        for i in range(0, self.rollout_len):
            delta = [0.0, 0.0, 0.0]
            pos, orn = pybullet.getBasePositionAndOrientation(model_id)
            if i == 0:
                obj_delta =[np.random.choice([0.5, 0.0, -0.5]), np.random.choice([0.5, 0.0, -0.5])]
            if not frame > pos[0] > -frame or not frame > pos[1] > -frame:
                obj_delta[0]*=-1
                obj_delta[1]*=-1

                newpos = [pos[0] + obj_delta[0], pos[1] + obj_delta[1], pos[2]]
            else:
                newpos = [pos[0]+ obj_delta[0], pos[1]+ obj_delta[1], pos[2]]
            #print(newpos)
            pybullet.resetBasePositionAndOrientation(model_id, newpos, orn)
            #print(start_time-time.time())
            start_time=time.time()
            if first:
                delta = [0.0, 0.0, 0.0]
                delta_pos = [0.0, 0.0, 0.0]
                first = True
            else:
                delta[0] = np.clip(pose_deltas[-1][3] + math.sqrt(dt)* np.random.randn(), -0.00001, 0.00001)
                delta[1] = np.clip(pose_deltas[-1][4] + math.sqrt(dt)* np.random.randn(), -0.001, 0.001)
                delta[2] = np.clip(pose_deltas[-1][5] + math.sqrt(dt)* np.random.randn(), -0.00, 0.00)
                delta_pos[0] = np.clip(COEFF*delta_pos[0] + math.sqrt(dt)* np.random.randn(), -0.1, 0.1)
                delta_pos[1] = np.clip(COEFF*delta_pos[1] + math.sqrt(dt)* np.random.randn(), -0.1, 0.1)
                delta_pos[2] = np.clip(COEFF*delta_pos[2] + math.sqrt(dt)* np.random.randn(), -0.02, 0.02)
            yaw += delta[2]
            pitch +=delta[0]
            roll += delta[1]
            camera_position_candidate =camera_position + delta_pos
            if not 3.0 > camera_position_candidate[0] > -3.0:
                delta_pos[0] = 0.0
            if not 3.0 > camera_position_candidate[1] > -3.0:
                delta_pos[1] = 0.0
            if not 3.0 > camera_position_candidate[2] > -3.0:
                delta_pos[2] = 0.0

            camera_position += delta_pos
            view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
                camera_position,
                self.camera_distance,
                np.rad2deg(yaw),
                np.rad2deg(pitch),
                np.rad2deg(roll),
                2
            )
            projection_matrix = pybullet.computeProjectionMatrixFOV(
                camera_field_of_view,
                self.image_size[0] / self.image_size[1],
                near_plane,
                far_plane
            )

            image = pybullet.getCameraImage(
                self.image_size[0],
                self.image_size[1],
                view_matrix,
                projection_matrix,
                shadow=0,
                lightDirection=[1, 1, 1],
                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
            )
            segm_image = image[4]
            color_image = np.reshape(np.array(image[2])[:, :, :3], (image[1], image[0], 3))

            segm_images.append(segm_image)
            depth_images.append(color_image)
            full_delta = np.concatenate((delta_pos, delta), axis=0)
            #print(full_delta)
            #print(camera_position)
            pose_deltas.append(full_delta)
        pybullet.resetSimulation()
        return depth_images, pose_deltas, segm_images

    def generate_images_for_random_scene(self, num, combine_images=True):
        images, poses, segm_images = self.generate_simulated_images_for_random_scene()
        id = uuid.uuid4()

        # Front image
        j=0

        imgs_folder = os.path.join(self.image_directory, "rollout_"+str(num))
        segm_imgs_folder = os.path.join(self.segm_directory, "rollout_"+str(num))

        if not os.path.isdir(imgs_folder):
            os.mkdir(imgs_folder)
        if not os.path.isdir(segm_imgs_folder):
            os.mkdir(segm_imgs_folder)
        for image, pose, segm_image in zip(images, poses, segm_images):

            depth_image_path = os.path.join(imgs_folder,'sim{}_rgb_{}.jpg'.format(num, j))
            img = Image.fromarray(image.astype(np.uint8))

            segm_image_path = os.path.join(segm_imgs_folder,'sim{}_sgm_{}.jpg'.format(num, j))
            segm_img = Image.fromarray(segm_image.astype(np.uint8))
            img.save(depth_image_path)
            segm_img.save(segm_image_path)
            j+=1
            yield [j, pose, depth_image_path, segm_image_path]

    def generate_images(self, number_simulations):
        for i in range(0, number_simulations):
            csv_name = "database"+str(i)+".csv"
            database_file = os.path.join(self.data_dir, "csv_files", csv_name)
            csv_file = open(database_file, mode='a')
            writer = csv.writer(csv_file)

            # If file is empty, write a header row
            if csv_file.tell() == 0:
                writer.writerow(['id', 'pose', 'image_path', 'segmented_image_path'])

            for row in gen.generate_images_for_random_scene(i):  # First value should be 0.0
                writer.writerow(row)

            csv_file.close()

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--dir", help="Output path")
    a.add_argument("--len_roll", help="Length of rollout", default=60)
    a.add_argument("--num_roll", help="Number of rollouts", default=1000)

    args = a.parse_args()
    out_dir = args.dir
    img_dir = os.path.join(out_dir, "imgs64")
    segm_imgs = os.path.join(out_dir, "segm_imgs")
    csv_dir = os.path.join(out_dir, "csv_files")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    if not os.path.exists(segm_imgs):
        os.mkdir(segm_imgs)
    if not os.path.exists(csv_dir):
        os.mkdir(csv_dir)

    len_rollout = int(args.len_roll)
    num_rollouts = int(args.num_roll)
    gen = ImageGenerator(out_dir, img_dir, len_rollout, segm_imgs)
    gen.generate_images(number_simulations=num_rollouts)
