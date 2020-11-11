#!/usr/bin/python3

import numpy as np


def generate_box():
    ''' Generates URDF strings for a box with uniformly distributed width, height, and length '''

    # generate random scaling between 1 and 10
    x = np.random.uniform(1, 5)
    y = np.random.uniform(1, 5)
    z = np.random.uniform(1, 5)

    # assume densitiy of 1 and calculate mass
    m = x * y * z

    # moments of inertia of the cuboid for rotations around x, y, and z axis
    i_x = 1 / 12 * m * (y * y + z * z)
    i_y = 1 / 12 * m * (z * z + x * x)
    i_z = 1 / 12 * m * (x * x + y * y)

    # generate urdf string for box
    urdf_string = """
<?xml version="1.0"?>
<robot name="physics">
    <link name="box">
        <visual>
            <material name="red">
                <color rgba="1 0 1 1"/>
            </material>
            <geometry>
                <box size="{:.2f} {:.2f} {:.2f}"/>
            </geometry>
        </visual>
        <collision>
            <geometry>
                <box size="{:.2f} {:.2f} {:.2f}"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="{:.2f}"/>
            <inertia ixx="{:.2f}" ixy="0.0" ixz="0.0" iyy="{:.2f}" iyz="0.0" izz="{:.2f}"/>
        </inertial>
    </link>
</robot>""".format(x, y, z, x, y, z, m, i_x, i_y, i_z)
    return urdf_string, [x, y, z]


def generate_cylinder():
    ''' Generates URDF strings for a cylinder with uniformly distributed radius and height '''

    # generate random scaling between 1 and 10
    r = np.random.uniform(0.5, 1.5)
    h = np.random.uniform(3, 6)

    # assume densitiy of 1 and calculate mass
    m = 2 * np.pi * r * r * h

    # moments of inertia of the cuboid for rotations around x, y, and z axis
    i_x = 1 / 12 * m * (3 * r * r + h * h)
    i_y = 1 / 12 * m * (3 * r * r + h * h)
    i_z = 1 / 2 * m * r * r

    # generate urdf string for cylinder
    urdf_string = """
<?xml version="1.0"?>
<robot name="physics">
    <link name="cylinder">
        <visual>
            <geometry>
                <cylinder length="{:.2f}" radius="{:.2f}"/>
            </geometry>
            <material name="blue">
                <color rgba="0 0 1 1"/>
            </material>
        </visual>
        <collision>
            <geometry>
                <cylinder length="{:.2f}" radius="{:.2f}"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="{:.2f}"/>
            <inertia ixx="{:.2f}" ixy="0.0" ixz="0.0" iyy="{:.2f}" iyz="0.0" izz="{:.2f}"/>
        </inertial>
    </link>
</robot>""".format(h, r, h, r, m, i_x, i_y, i_z)
    return urdf_string, [r, h]


def generate_sphere():
    r = np.random.uniform(1.5, 2.0)

    urdf_string = """
<?xml version="1.0"?>
<robot name="physics">
  <link name="sphere">
    <visual>
      <geometry>
        <sphere radius="{:.2f}"/>
      </geometry>
      <material name="green">
            <color rgba="0 1 0 1"/>
      </material>
    </visual>
  </link>
</robot>""".format(r)
    return urdf_string, [r]
