import enum
import numpy as np
import torch
import xml.etree.ElementTree as ET

from ..utils import torch_util

class JointType(enum.Enum):
    ROOT = 0
    HINGE = 1
    SPHERICAL = 2
    FIXED = 3

class Joint():
    def __init__(self, name, joint_type, axis):
        self.name = name
        self.joint_type = joint_type
        self.axis = axis
        self.dof_idx = -1
        return

    def get_dof_dim(self):
        if (self.joint_type == JointType.ROOT):
            dof_dim = 0
        elif (self.joint_type == JointType.HINGE):
            dof_dim = 1
        elif (self.joint_type == JointType.SPHERICAL):
            dof_dim = 3
        elif (self.joint_type == JointType.FIXED):
            dof_dim = 0
        else:
            assert(False), "Unsupported joint type: {:s}".format(self.joint_type)
        return dof_dim

    def get_joint_dof(self, dof):
        dof_idx = self.dof_idx
        dof_dim = self.get_dof_dim()
        j_dof = dof[..., dof_idx:dof_idx + dof_dim]
        return j_dof

    def set_joint_dof(self, j_dof, out_dof):
        dof_idx = self.dof_idx
        dof_dim = self.get_dof_dim()
        out_dof[..., dof_idx:dof_idx + dof_dim] = j_dof
        return

    def dof_to_rot(self, dof):
        rot_shape = list(dof.shape[:-1])
        rot_shape = rot_shape + [4]
        rot = torch.zeros(rot_shape, device=dof.device, dtype=dof.dtype)

        if (self.joint_type == JointType.ROOT):
            rot[..., -1] = 1
        elif (self.joint_type == JointType.HINGE):
            axis = self.axis
            axis_shape = rot[..., 0:3].shape
            axis = torch.broadcast_to(axis, axis_shape)
            dof = dof.squeeze(-1)
            rot[:] = torch_util.axis_angle_to_quat(axis, dof)
        elif (self.joint_type == JointType.SPHERICAL):
            rot[:] = torch_util.exp_map_to_quat(dof)
        elif (self.joint_type == JointType.FIXED):
            rot[..., -1] = 1
        else:
            assert(False), "Unsupported joint type: {:s}".format(self.joint_type)

        return rot

    def rot_to_dof(self, rot):
        dof_dim = self.get_dof_dim()
        dof_shape = list(rot.shape[:-1])
        dof_shape = dof_shape + [dof_dim]
        dof = torch.zeros(dof_shape, device=rot.device, dtype=rot.dtype)

        if (self.joint_type == JointType.ROOT):
            pass
        elif (self.joint_type == JointType.HINGE):
            j_axis = self.axis
            angle = torch_util.quat_twist_angle(rot, j_axis)
            dof[:] = angle.unsqueeze(-1)
        elif (self.joint_type == JointType.SPHERICAL):
            dof[:] = torch_util.quat_to_exp_map(rot)
        elif (self.joint_type == JointType.FIXED):
            pass
        else:
            assert(False), "Unsupported joint type: {:s}".format(self.joint_type)

        return dof


class KinCharModel():
    def __init__(self, device):
        self._device = device
        return

    def init(self, body_names, parent_indices, local_translation, local_rotation, joints):
        num_bodies = len(body_names)
        assert(len(parent_indices) == num_bodies)
        assert(len(local_translation) == num_bodies)
        assert(len(local_rotation) == num_bodies)
        assert(len(joints) == num_bodies)

        self._body_names = body_names
        self._parent_indices = torch.tensor(parent_indices, device=self._device, dtype=torch.long)
        self._local_translation = torch.tensor(np.array(local_translation), device=self._device, dtype=torch.float32)
        self._local_rotation = torch.tensor(np.array(local_rotation), device=self._device, dtype=torch.float32)
        self._joints = joints
        
        self._dof_size = self._label_dof_indices(self._joints)
        self._name_body_map = self._build_name_body_map()
        
        return

    def load_char_file(self, char_file):
        tree = ET.parse(char_file)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        assert(xml_world_body is not None)

        xml_body_root = xml_world_body.find("body")
        assert(xml_body_root is not None)

        body_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints = []

        default_joint_type = self._parse_default_joint_type(xml_doc_root)

        # recursively adding all bodies into the skel_tree
        def _add_xml_body(xml_node, parent_index, body_index, default_joint_type):
            body_name = xml_node.attrib.get("name")
            # parse the local translation into float list
            pos_data = xml_node.attrib.get("pos")
            if (pos_data is None):
                pos = np.array([0.0, 0.0, 0.0])
            else:
                pos = np.fromstring(pos_data, dtype=float, sep=" ")
            
            rot_data = xml_node.attrib.get("quat")
            if (rot_data is None):
                rot = np.array([0.0, 0.0, 0.0, 1.0])
            else:
                rot = np.fromstring(rot_data, dtype=float, sep=" ")
                rot_w = rot[..., 0].copy()
                rot[..., 0:3] = rot[..., 1:]
                rot[..., -1] = rot_w

            if (body_index == 0):
                curr_joint = self._build_root_joint()
            else:
                joint_data = xml_node.findall("joint")
                curr_joint = self._parse_joint(body_name, joint_data, default_joint_type)

            body_names.append(body_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(rot)
            joints.append(curr_joint)

            curr_index = body_index
            body_index += 1
            for child_body in xml_node.findall("body"):
                body_index = _add_xml_body(child_body, curr_index, body_index, default_joint_type)

            return body_index

        _add_xml_body(xml_body_root, -1, 0, default_joint_type)

        self.init(body_names=body_names,
                  parent_indices=parent_indices,
                  local_translation=local_translation,
                  local_rotation=local_rotation,
                  joints=joints)
        return

    def load_from_articulation(self, articulation):
        """Load character model from Isaac Lab Articulation (USD).
        
        Args:
            articulation: Isaac Lab Articulation object
        """
        # Get body/link names from articulation
        body_names = articulation.data.body_names
        num_bodies = len(body_names)
        
        # Build simple tree structure (assume all links connect to root for now)
        # In a full implementation, you'd need to parse the actual hierarchy
        parent_indices = [-1] + [0] * (num_bodies - 1)  # All connect to root
        
        # Initialize with identity transforms
        local_translation = [[0.0, 0.0, 0.0] for _ in range(num_bodies)]
        local_rotation = [[0.0, 0.0, 0.0, 1.0] for _ in range(num_bodies)]
        
        # Create joints - root joint for first body, spherical for others
        joints = []
        joints.append(self._build_root_joint())  # Root body
        
        # Create spherical joints for all other bodies (simplification)
        for i in range(1, num_bodies):
            joint = Joint(name=f"joint_{body_names[i]}", 
                         joint_type=JointType.SPHERICAL,
                         axis=torch.tensor([0.0, 0.0, 1.0], device=self._device))
            joints.append(joint)
        
        self.init(body_names=list(body_names),
                  parent_indices=parent_indices,
                  local_translation=local_translation,
                  local_rotation=local_rotation,
                  joints=joints)
        return
    
    def get_body_names(self):
        return self._body_names

    def get_joint(self, j):
        assert(j > 0)
        return self._joints[j]

    def get_parent_id(self, j):
        return self._parent_indices[j]

    def get_dof_size(self):
        return self._dof_size

    def get_joint_dof_idx(self, j):
        dof_idx = self.get_joint(j).dof_idx
        return dof_idx

    def get_joint_dof_dim(self, j):
        dof_dim = self.get_joint(j).get_dof_dim()
        return dof_dim

    def get_num_joints(self):
        return len(self._joints)

    def dof_to_rot(self, dof):
        num_joints = self.get_num_joints()

        rot_shape = list(dof.shape[:-1])
        rot_shape = rot_shape + [num_joints - 1, 4]
        joint_rot = torch.zeros(rot_shape, device=dof.device, dtype=dof.dtype)

        for j in range(1, num_joints):
            joint = self.get_joint(j)
            j_dof = joint.get_joint_dof(dof)
            j_rot = joint.dof_to_rot(j_dof)
            joint_rot[..., j - 1, :] = j_rot

        return joint_rot

    def rot_to_dof(self, rot):
        dof_shape = list(rot.shape[:-2])
        dof_shape = dof_shape + [self._dof_size]
        dof = torch.zeros(dof_shape, device=rot.device, dtype=rot.dtype)
        
        num_joints = self.get_num_joints()
        for j in range(1, num_joints):
            joint = self.get_joint(j)
            j_dof_dim = joint.get_dof_dim()
            if (j_dof_dim > 0):
                j_rot = rot[..., j - 1, :]
                j_dof = joint.rot_to_dof(j_rot)
                joint.set_joint_dof(j_dof, dof)

        return dof

    def forward_kinematics(self, root_pos, root_rot, joint_rot):
        num_joints = self.get_num_joints()
        body_pos = [None] * num_joints
        body_rot = [None] * num_joints

        body_pos[0] = root_pos
        body_rot[0] = root_rot

        for j in range(1, num_joints):
            j_rot = joint_rot[..., j - 1, :]
            local_trans = self._local_translation[j]
            local_rot = self._local_rotation[j]
            parent_idx = self._parent_indices[j]
            
            parent_pos = body_pos[parent_idx]
            parent_rot = body_rot[parent_idx]

            local_trans = torch.broadcast_to(local_trans, parent_pos.shape)
            local_rot = torch.broadcast_to(local_rot, parent_rot.shape)

            world_trans = torch_util.quat_rotate(parent_rot, local_trans)

            curr_pos = parent_pos + world_trans
            curr_rot = torch_util.quat_mul(local_rot, j_rot)
            curr_rot = torch_util.quat_mul(parent_rot, curr_rot)

            body_pos[j] = curr_pos
            body_rot[j] = curr_rot

        body_pos = torch.stack(body_pos, dim=-2)
        body_rot = torch.stack(body_rot, dim=-2)
        
        return body_pos, body_rot
    
    def compute_frame_dof_vel(self, joint_rot, dt):
        joint_rot0 = joint_rot[:-1, :, :]
        joint_rot1 = joint_rot[1:, :, :]
        dof_vel = self.compute_dof_vel(joint_rot0, joint_rot1, dt)
        
        final_vels = dof_vel[-1:, :]
        dof_vel = torch.cat([dof_vel, final_vels], dim=-2)
        return dof_vel

    def compute_dof_vel(self, joint_rot0, joint_rot1, dt):
        dof_size = self.get_dof_size()
        dof_shape = list(joint_rot0.shape[:-2])
        dof_shape = dof_shape + [dof_size]
        dof_vel = torch.zeros(dof_shape, device=joint_rot0.device, dtype=joint_rot0.dtype)

        drot = torch_util.quat_mul(torch_util.quat_conjugate(joint_rot0), joint_rot1)
        drot = torch_util.quat_normalize(drot)

        num_joints = self.get_num_joints()
        for j in range(1, num_joints):
            joint = self.get_joint(j)
            j_drot = drot[..., j - 1, :]

            if (joint.joint_type == JointType.ROOT):
                pass
            elif (joint.joint_type == JointType.HINGE):
                j_axis = joint.axis
                j_dof_vel = torch_util.quat_to_exp_map(j_drot) / dt
                j_dof_vel = torch.sum(j_axis * j_dof_vel, dim=-1, keepdim=True)
                joint.set_joint_dof(j_dof_vel, dof_vel)
            elif (joint.joint_type == JointType.SPHERICAL):
                j_dof_vel = torch_util.quat_to_exp_map(j_drot) / dt
                joint.set_joint_dof(j_dof_vel, dof_vel)
            elif (joint.joint_type == JointType.FIXED):
                pass
            else:
                assert(False), "Unsupported joint type: {:s}".format(joint.joint_type)

        return dof_vel
    
    def get_body_name(self, body_id):
        return self._body_names[body_id]
    
    def get_body_id(self, body_name):
        assert body_name in self._name_body_map
        return self._name_body_map[body_name]
    
    def get_joint_id(self, body_name):
        assert body_name in self._name_body_map
        return self._name_body_map[body_name] -1 # joint arrays exclude the root
    
    def _build_name_body_map(self):
        name_body_map = dict()

        for body_id, body_name in enumerate(self._body_names):
            name_body_map[body_name] = body_id

        return name_body_map
    
    def _build_root_joint(self):
        joint = Joint(name="root",
                      joint_type=JointType.ROOT,
                      axis=None)
        return joint
    
    def _parse_joint(self, body_name, xml_joint_data, default_joint_type):
        num_joints = len(xml_joint_data)

        if (num_joints == 0):
            joint = self._parse_fixed_joint(body_name)
        elif (num_joints == 3):
            joint = self._parse_sphere_joint(xml_joint_data, default_joint_type)
        elif (num_joints == 1):
            joint_type_str = xml_joint_data[0].attrib.get("type")
            if (joint_type_str is None):
                joint_type_str = default_joint_type

            if (joint_type_str == "hinge"):
                joint = self._parse_hinge_joint(xml_joint_data[0])
            else:
                assert(False), "Unsupported joint type: {:s}".format(joint_type_str)
        else:
            assert(False), "Series joints are not supported."
        
        return joint

    def _parse_hinge_joint(self, xml_joint_data):
        joint_name = xml_joint_data.attrib.get("name")

        joint_pos_data = xml_joint_data.attrib.get("pos")
        if (joint_pos_data is not None):
            joint_pos = np.fromstring(joint_pos_data, dtype=float, sep=" ")
            if (np.any(joint_pos)):
                assert(False), "Joint offsets are not supported"

        joint_axis = np.fromstring(xml_joint_data.attrib.get("axis"), dtype=float, sep=" ")
        joint_axis = torch.tensor(joint_axis, device=self._device, dtype=torch.float32)

        joint = Joint(name=joint_name,
                      joint_type=JointType.HINGE,
                      axis=joint_axis)
        return joint

    def _parse_sphere_joint(self, xml_joint_data, default_joint_type):
        # consolidate series of three hinge joints into a single spherical joint
        num_joints = len(xml_joint_data)
        assert(num_joints == 3)

        is_spherical = True
        for joint_data in xml_joint_data:
            joint_type_str = joint_data.attrib.get("type")
            if (joint_type_str is None):
                joint_type_str = default_joint_type

            joint_pos_data = joint_data.attrib.get("pos")
            if (joint_pos_data is not None):
                joint_pos = np.fromstring(joint_pos_data, dtype=float, sep=" ")
                if (np.any(joint_pos)):
                    assert(False), "Joint offsets are not supported"

            if (joint_type_str != "hinge"):
                is_spherical = False
                break

        if (is_spherical):
            joint_name = xml_joint_data[0].attrib.get("name")
            joint_name = joint_name[:joint_name.rfind('_')]
            joint = Joint(name=joint_name,
                               joint_type=JointType.SPHERICAL,
                               axis=None)
        else:
            assert(False), "Invalid format for a spherical joint"

        return joint

    def _parse_fixed_joint(self, body_name):
        joint = Joint(name=body_name,
                      joint_type=JointType.FIXED,
                      axis=None)
        return joint

    def _label_dof_indices(self, joints):
        dof_idx = 0
        for curr_joint in joints:
            if (curr_joint is not None):
                dof_dim = curr_joint.get_dof_dim()
                curr_joint.dof_idx = dof_idx
                dof_idx += dof_dim
        return dof_idx

    def _parse_default_joint_type(self, xml_node):
        joint_type_str = "hinge"

        default_data = xml_node.find("default")

        if (default_data is not None):
            default_data = default_data.findall("default")

            if (default_data is not None):
                for data in default_data:
                    class_data = data.attrib.get("class")
                    if (class_data == "body"):
                        joint_data = data.find("joint")
                        if (joint_data is not None):
                            joint_type_str = joint_data.attrib.get("type")
                            break

        return joint_type_str

    def _build_body_children_map(self):
        num_joints = self.get_num_joints()
        body_children = [[] for _ in range(num_joints)]
        for j in range(num_joints):
            parent_idx = self._parent_indices[j].item()
            if (parent_idx != -1):
                body_children[parent_idx].append(j)
        return body_children


    
    def output_xml(self, output_file):
        xml_template = """<mujoco model="character">
    <statistic extent="2" center="0 0 1"/>

    <default>
        <motor ctrlrange="-1 1" ctrllimited="true"/>
        <default class="body">
            <geom condim="1" friction="1.0 0.05 0.05" solimp=".9 .99 .003" solref=".015 1"/>
            <joint limited="true" solimplimit="0 .99 .01"/>
        </default>
    </default>

    <worldbody>
        <geom name="floor" type="plane" conaffinity="1" size="100 100 .2" material="grid"/>
{:s}
    </worldbody>

    <actuator>{:s}
    </actuator>
</mujoco>"""

        bodies_xml = self._build_bodies_xml()
        actuator_xml = self._build_actuators_xml()

        char_xml = xml_template.format(bodies_xml, actuator_xml)

        with open(output_file, "w") as out_file:
            out_file.write(char_xml)

        return

    def _build_bodies_xml(self):
        body_children = self._build_body_children_map()
        bodies_xml = self._build_body_xml(body_children=body_children, body_id=0)
        return bodies_xml

    def _build_body_xml(self, body_children, body_id):
        body_template = '''
        <body name="{:s}" pos="{:.4f} {:.4f} {:.4f}" rot="{:.4f} {:.4f} {:.4f} {:.4f}">{:s}{:s}{:s}
        </body>'''

        root_template = '''
        <body name="{:s}" pos="0 0 0" childclass="body">{:s}{:s}{:s}
        </body>'''
        
        body_name = self._body_names[body_id]
        pos = self._local_translation[body_id].cpu().numpy()
        rot = self._local_rotation[body_id].cpu().numpy()

        children_xml = ""
        children = body_children[body_id]
        if (len(children) > 0):
            for child_id in children:
                child_xml = self._build_body_xml(body_children=body_children, body_id=child_id)

                child_xml_lines = child_xml.splitlines()
                indented_xml_lines = ["\t" + l for l in child_xml_lines]
                child_xml = "\n".join(indented_xml_lines)

                children_xml += "\n" + child_xml

        joint_xml = self._build_joint_xml(body_children, body_id)
        geom_xml = self._build_geom_xml(body_children, body_id)

        is_root = body_id == 0
        if (is_root):
            body_xml = root_template.format(body_name,
                                            joint_xml,
                                            geom_xml,
                                            children_xml)
        else:
            body_xml = body_template.format(body_name,
                                            pos[0], pos[1], pos[2], 
                                            rot[3], rot[0], rot[1], rot[2], 
                                            joint_xml,
                                            geom_xml,
                                            children_xml)
        return body_xml

    def _build_joint_xml(self, body_children, body_id):
        root_template = '''
			<freejoint name="{:s}"/>'''

        joint_template = '''
            <joint name="{:s}_x" type="hinge" axis="1 0 0" range="-180 180" stiffness="100" damping="10" armature=".01"/>
            <joint name="{:s}_y" type="hinge" axis="0 1 0" range="-180 180" stiffness="100" damping="10" armature=".01"/>
            <joint name="{:s}_z" type="hinge" axis="0 0 1" range="-180 180" stiffness="100" damping="10" armature=".01"/>'''
            
        body_name = self._body_names[body_id]
        
        is_root = body_id == 0
        if (is_root):
            joint_xml = root_template.format(body_name)
        else:
            joint = self.get_joint(body_id)
            joint_type = joint.joint_type

            if (joint_type == JointType.HINGE):
                j_axis = joint.axis
                joint_template = '''
            <joint name="{:s}" type="hinge" axis="{:.4f} {:.4f} {:.4f}" range="-180 180" stiffness="100" damping="10" armature=".01"/>'''
                joint_xml = joint_template.format(body_name, j_axis[0], j_axis[1], j_axis[2])
            elif (joint_type == JointType.SPHERICAL):
                joint_template = '''
            <joint name="{:s}_x" type="hinge" axis="1 0 0" range="-180 180" stiffness="100" damping="10" armature=".01"/>
            <joint name="{:s}_y" type="hinge" axis="0 1 0" range="-180 180" stiffness="100" damping="10" armature=".01"/>
            <joint name="{:s}_z" type="hinge" axis="0 0 1" range="-180 180" stiffness="100" damping="10" armature=".01"/>'''
                joint_xml = joint_template.format(body_name, body_name, body_name)
            elif (joint_type == JointType.FIXED):
                joint_xml = ""
            else:
                assert(False), "Unsupported joint type: {:s}".format(joint.joint_type)

        return joint_xml

    def _build_geom_xml(self, body_children, body_id):
        joint_radius = 0.02
        bone_radius = 0.01
        joint_template = '''
            <geom type="sphere" name="{:s}" pos="0 0 0" size="{:.4f}" density="1000"/>'''
        bone_template = '''
            <geom type="capsule" name="{:s}" fromto="0 0 0 {:.4f} {:.4f} {:.4f}" size="{:.4f}" density="1000"/>'''

        body_name = self._body_names[body_id]

        geom_xml = ""

        joint_pos = self._local_translation[body_id]
        bone_len = np.linalg.norm(joint_pos)
        if (bone_len > 0):
            clamp_joint_radius = min(0.25 * bone_len, joint_radius)
            geom_xml += joint_template.format(body_name, clamp_joint_radius)

        children = body_children[body_id]
        joint_rot = self._local_rotation[body_id]
        for c in children:
            child_pos = self._local_translation[c]
            bone_pos = torch_util.quat_rotate(joint_rot, child_pos)

            child_bone_len = np.linalg.norm(bone_pos)
            if (child_bone_len > 0):
                geom_len = max(0.001, child_bone_len)
                geom_pos = geom_len * bone_pos / child_bone_len
                child_bone_radius = min(0.15 * child_bone_len, bone_radius)

                bone_xml = bone_template.format(body_name, geom_pos[0], geom_pos[1], geom_pos[2], child_bone_radius)
                geom_xml += bone_xml

        return geom_xml

    def _build_actuators_xml(self):
        motor_template = '''
        <motor name='{:s}'       	gear='100' 	joint='{:s}'/>'''

        actuators_xml = ""

        num_joints = self.get_num_joints()
        for j in range(1, num_joints):
            joint = self.get_joint(j)
            joint_type = joint.joint_type
            body_name = self._body_names[j]

            if (joint_type == JointType.HINGE):
                actuator_xml = motor_template.format(body_name, body_name)
            elif (joint_type == JointType.SPHERICAL):
                actuator_xml = motor_template.format(body_name + "_x", body_name + "_x")
                actuator_xml += motor_template.format(body_name + "_y", body_name + "_y")
                actuator_xml += motor_template.format(body_name + "_z", body_name + "_z")
            elif (joint_type == JointType.FIXED):
                actuator_xml = ""
            else:
                assert(False), "Unsupported joint type: {:s}".format(joint.joint_type)

            actuators_xml += actuator_xml

        return actuators_xml