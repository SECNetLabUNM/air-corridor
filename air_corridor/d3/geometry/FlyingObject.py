from pygame import gfxdraw

from air_corridor.tools._descriptor import Position, PositiveNumber
from air_corridor.tools.util import *

# from utils.memory import DebugTracking

'''
training and testing are very different
train: UAVs are only trained for a single corridor, 
        either DirectionalPartialAnnulusCorridor or RectangleCorridor
test: UAV only use well-trained model for testing in a multi-corridor environment, 
        with all positions reformed with relative positions to the centric of the current corridor.
So, accumulated reward in only calculated within one corridor, not across corridors.
'''


class FlyingObject:
    flying_list = []
    position = Position(3)

    '''
    corridors={'A':Corridor{'name','anchor'},
               'B':Corridor{'name','anchor'}}
    '''
    safe_distance = 1
    events = ['won', 'collided', 'breached', 'half', 'breached1', 'breached2', 'collided1', 'collided2']
    GAMMA = 0.99
    capacity = 6
    beta_adaptor_coefficient = 1.0
    num_corridor_in_state = 1

    # Flag indicating if the current environment is in the final corridor.
    # Essential for training in multi-corridor environments and
    # applicable to scenarios with a single-segment corridor in the state.
    corridor_index_awareness = False
    corridor_state_length = 16
    uniform_state = False

    def __int__(self,
                name,
                position=np.array([0, 0, 0]),
                position_delta=np.array([0, 0, 0]),
                next_position=np.array([0, 0, 0]),
                velocity=np.array([0, 0, 0]),
                next_velocity=np.array([0, 0, 0]),
                discrete=False,
                reduce_space=True,
                ):
        self.discrete = discrete
        self.name = name
        self.terminated = False
        self.truncated = False

        self.globalPosition = None

        self.position = position
        self.position_delta = position_delta
        self.next_position = next_position

        self.velocity = velocity
        self.next_velocity = next_velocity
        self.position_delta = None

        self.status = 'Normal'
        self.reduce_space = reduce_space

    def apply_acceleration(self, acc, dt):
        self.next_velocity, self.position_delta, reward_illegal_acc = apply_acceleration(self.velocity,
                                                                                         self.velocity_max,
                                                                                         acc,
                                                                                         dt)
        self.next_position = self.position + self.position_delta
        if np.linalg.norm(self.next_position) > 500:
            input('abnormal')
            apply_acceleration(self.velocity, self.velocity_max, acc, dt)
        return reward_illegal_acc

    def get_distance_to(self, other_flying_object):
        distance = distance_point_point(self.position, other_flying_object.position)
        return distance

    def render_self(self):
        """ render itself """
        pass

    @classmethod
    def action_adapter(cls, action):
        '''
        r, theta, phi  = action
        r     = [0, 1] -> [0,1]
        theta = [0, 1] -> [0, np.pi]
        phi   = [0, 1] -> [-np.pi, np.pi]*1.1, with beta base of 1, the selection concentrate on [2pi,0] is truncated.
        :param action:
        :return:
        '''
        return [action[0], action[1] * np.pi, (action[0] - 0.5) * 2 * np.pi * cls.beta_adaptor_coefficient]

        # return [(action[0] - 0.5) * 2 * np.pi, action[1] * np.pi, action[2]]


class UAV(FlyingObject):
    '''unmanned aerial vehicle'''
    corridor_graph = None
    corridors = None

    reduce_space = True

    def __init__(self,
                 init_corridor,
                 des_corridor=None,
                 discrete=False,
                 name=None,
                 # velocity_max=0.6,
                 # acceleration_max=0.6,
                 velocity_max=1.5,
                 acceleration_max=0.3,
                 plane_offset_assigned=None,
                 reduce_space=True):

        # if self.corridor_graph is None:
        #     print("Error: Have not graph the corridors.")
        #     sys.exit()
        super().__int__(name, discrete=discrete, reduce_space=reduce_space)

        self.plane_offset_assigned = plane_offset_assigned

        if discrete:
            self.discrete_action_space = 8

        self.velocity_max = velocity_max
        self.acceleration_max = acceleration_max

        self.init_corridor = init_corridor
        self.des_corridor = des_corridor
        self.enroute = None

        self.instant_reward = 0


        self.neighbors = []

        self.steps = 0
        self.outside_counter = None
        self.flying_list.append(self)

        self.accumulated_reward = 0

        self.trajectory = []
        self.reward = 0

    def update_position(self):
        self.position = self.next_position
        self.velocity = self.next_velocity
        # print(self.position)

    def decompose_target(self):

        assert (self.enroute['init'] in self.corridor_graph.keys() and
                self.enroute['des'] in self.corridor_graph.keys()), \
            "Error, the initial or the last corridor is not specified."
        path = bfs_find_path(self.corridor_graph, self.enroute['init'], self.enroute['des'])
        if path is None:
            self.enroute['path'] = None
            self.terminated = True
        else:
            self.enroute['path'] = path
            if len(path) > 1:
                self.enroute['next'] = path[1]

    #
    def take(self, action, dt):
        '''
        in take action on the base with reduced space, while output the "actual" values
        '''
        action = self.action_adapter(action)
        # r, theta, phi = action
        r = action[0]
        if self.reduce_space:
            # action is generated on based shape with direction of [0,0,1]
            heading_vector_on_base = polar_to_unit_normal(action[1:])
            heading_vector = self.corridors[self.enroute['current']].rotate_to_remote(heading_vector_on_base)
            # heading_vector = np.dot(self.corridors[self.enroute['current']].rotation_matrix_to_remote,
            #                         heading_vector_on_base)
        else:
            # action is generated on shape with different direction
            heading_vector = polar_to_unit_normal(action[1:])

        acc = self.acceleration_max * r * heading_vector
        reward_illegal_acc = self.apply_acceleration(acc, dt)
        self.steps += 1
        # here penalize with illegal actions in two parts,
        # 1) action range beyond pre-determined range
        # 2) action within range but enforce uav goes beyond velocity max
        # print(f"acc: {np.round(acc,3)},last vel: {np.round(self.velocity,3)}, "
        #       f"next vel:{np.round(self.next_velocity,3)}, position_delta:{np.round(self.position_delta,3)}")
        return 0  # reward_illegal_acc

    def reset(self):
        self.terminated = False
        self.truncated = False
        self.enroute = {'init': self.init_corridor,
                        'des': self.des_corridor,
                        'current': self.init_corridor,
                        'next': None,
                        'path': None}
        self.decompose_target()

        self.position = UAV.corridors[self.enroute['current']].release_uav(self.plane_offset_assigned)
        self.next_position = None
        self.velocity = np.array([0, 0, 0])
        self.next_velocity = None
        self.outside_counter = 0
        self.status = 'Normal'

    def update_accumulated_reward(self):
        self.accumulated_reward = self.accumulated_reward * UAV.GAMMA + self.instant_reward

    def _report_self(self):
        # 4+3*4=16
        cur = self.corridors[self.enroute['current']]
        if self.reduce_space:
            base_position = cur.project_to_base(self.position)
            first = [self.velocity_max, self.acceleration_max,
                     cur.distance_object_to_point(self.position),
                     np.linalg.norm(self.velocity)] + \
                    list(base_position) + \
                    list(cur.rotate_to_base(self.velocity))
            # if torus
            if cur.shapeType == [0, 1]:
                second = list(cur.convert_2_polar(self.position, self.reduce_space)) + \
                         list(cur.convert_vec_2_polar(self.position, self.velocity, self.reduce_space))
            # if cylinder
            elif cur.shapeType == [1, 0]:
                second = [0] * 6
            # indicate whether being in the last corridor
        else:
            first = [self.velocity_max, self.acceleration_max,
                     cur.distance_object_to_point(self.position),
                     np.linalg.norm(self.velocity)] + \
                    list(cur.point_relative_center_position(self.position)) + \
                    list(self.velocity)
            if cur.shapeType == [0, 1]:
                second = [0] * 6
            elif cur.shapeType == [1, 0]:
                second = [0] * 6
        if any(np.isnan(first + second)):
            print('nan in self')
            input("Press Enter to continue...")
        agent_status = first + second
        # if UAV.corridor_index_awareness:
        #     third = [1] if self.enroute['current'] == self.enroute['path'][-1] else [0]
        #     agent_status += third
        corridor_status = self._report_corridor()
        return agent_status + corridor_status

    def _report_other(self):
        other_uavs_status = []
        for agent in self.flying_list:
            if agent is self or agent.enroute['current'] != self.enroute['current']:
                continue
                # 4+3*6=22

            cur = self.corridors[agent.enroute['current']]
            if self.reduce_space:
                if UAV.uniform_state:
                    first = [float(not agent.terminated), agent.velocity_max, agent.acceleration_max,
                             np.linalg.norm(agent.position - self.position)] + list(
                        cur.project_to_base(agent.position)) + list(
                        cur.rotate_to_base(agent.velocity)) + list(
                        cur.rotate_to_base(agent.position - self.position)) + list(
                        cur.rotate_to_base(agent.velocity - self.velocity))
                    if cur.shapeType == [0, 1]:
                        second = list(cur.convert_2_polar(agent.position, self.reduce_space)) + \
                                 list(cur.convert_vec_2_polar(agent.position, agent.velocity, self.reduce_space))
                    elif cur.shapeType == [1, 0]:
                        second = [0] * 6
                else:
                    first = [float(not agent.terminated), agent.velocity_max, agent.acceleration_max,
                             np.linalg.norm(agent.position - self.position)] + list(
                        cur.project_to_base(agent.position)) + list(
                        cur.rotate_to_base(agent.velocity)) + list(
                        cur.rotate_to_base(agent.position - self.position)) + list(
                        cur.rotate_to_base(agent.velocity - self.velocity))
                    if cur.shapeType == [0, 1]:
                        second = list(cur.convert_2_polar(agent.position, self.reduce_space)) + \
                                 list(cur.convert_vec_2_polar(agent.position, agent.velocity, self.reduce_space))
                    elif cur.shapeType == [1, 0]:
                        second = [0] * 6

            else:
                first = ([float(not agent.terminated), agent.velocity_max, agent.acceleration_max,
                          np.linalg.norm(agent.position - self.position)] +
                         list(agent.position) +
                         list(agent.velocity) +
                         list(agent.position - self.position) +
                         list(agent.velocity - self.velocity))
                if cur.shapeType == [0, 1]:
                    second = [0] * 6
                elif cur.shapeType == [1, 0]:
                    second = [0] * 6

            # if np.any(np.isnan(one_agent_status)):
            #     print('nan in neighbor')
            #     input("Press Enter to continue...")
            agent_status = first + second
            # if UAV.corridor_index_awareness:
            #     third = [1] if agent.enroute['current'] == agent.enroute['path'][-1] else [0]
            #     agent_status += third
            corridor_status = agent._report_corridor()
            other_uavs_status.append(agent_status + corridor_status)
        while len(other_uavs_status) < self.capacity - 1:
            # base_elements = 23 if UAV.corridor_index_awareness else 22
            # other_uavs_status.append([0] * (base_elements + 17 * self.num_corridor_in_state))

            other_uavs_status.append([0] * (22 + UAV.corridor_state_length * self.num_corridor_in_state))
        return other_uavs_status

    def _report_corridor(self):
        # 16 elements
        cur = self.corridors[self.enroute['current']]
        corridor_status = []
        cur_index = self.enroute['path'].index(self.enroute['current'])
        res_path = self.enroute['path'][cur_index:]
        for i, key_corridor in enumerate(res_path):
            if i + 1 > self.num_corridor_in_state:
                break
            single_c_status = self.corridors[key_corridor].report(base=cur)

            if UAV.corridor_index_awareness:
                corridor_index_state = [0, 0, 0, 0]
                if UAV.corridor_index_awareness[-1] and res_path[-1] == key_corridor:
                    corridor_index_state[3] = 1
                elif UAV.corridor_index_awareness[0] and (
                        self.enroute['path'][0] == key_corridor or sum(UAV.corridor_index_awareness) == 2):
                    corridor_index_state[0] = 1
                elif UAV.corridor_index_awareness[1] and (
                        self.enroute['path'][1] == key_corridor or sum(UAV.corridor_index_awareness) == 3):
                    corridor_index_state[1] = 1
                elif UAV.corridor_index_awareness[2] and (
                        self.enroute['path'][2] == key_corridor or sum(UAV.corridor_index_awareness) == 4):
                    corridor_index_state[2] = 1


                assert sum(corridor_index_state) == 1, f"{UAV.corridor_index_awareness}"

            # indicating_being_the_last_corridor = [0] if res_path[-1] == key_corridor else [1]
            # if len(self.enroute['path'])==1:
            #     indicating_being_the_first_corridor=[0]
            # indicating_being_the_first_corridor = [1] if self.enroute['path'][0] == key_corridor else [0]
            # single_c_status += indicating_being_the_last_corridor
                single_c_status = single_c_status + corridor_index_state

            corridor_status += single_c_status
        corridor_status += [0] * (UAV.corridor_state_length * (self.num_corridor_in_state - len(res_path)))
        return corridor_status

    def report(self):
        '''
        corridor_status: 16*n, single is 16
        self= 16+16*n
        other_uav: 22+16*n
        :param padding:
        :param reduce_space:
        :return:
        '''

        uav_status = self._report_self()
        other_uavs_status = self._report_other()
        # print(f" corridor, {len(corridor_status)}")
        # print(f" uav_status, {len(uav_status)}")
        return {'self': uav_status, 'other': other_uavs_status}

        # 8
        # corridor_status = self._report_corridor()
        # uav_status = self._report_self()
        # other_uavs_status = self._report_other()
        # # print(f" corridor, {len(corridor_status)}")
        # # print(f" uav_status, {len(uav_status)}")
        # return {'self': uav_status + corridor_status, 'other': other_uavs_status}

    def render_self(self, surf):
        if self.status == 'won':
            gfxdraw.filled_circle(
                surf,
                int(OFFSET_x + self.position[0] * SCALE),
                int(OFFSET_y + self.position[1] * SCALE),
                FLYOBJECT_SIZE - 1,
                GREEN,
            )
        elif self.terminated:
            gfxdraw.filled_circle(
                surf,
                int(OFFSET_x + self.position[0] * SCALE),
                int(OFFSET_y + self.position[1] * SCALE),
                FLYOBJECT_SIZE - 1,
                RED,
            )
        else:
            gfxdraw.filled_circle(
                surf,
                int(OFFSET_x + self.position[0] * SCALE),
                int(OFFSET_y + self.position[1] * SCALE),
                FLYOBJECT_SIZE,
                PURPLE,
            )

    class NCFO(FlyingObject):
        """
        non-cooperative flying objects


        """
        boundary = [None] * 3
        velocity = PositiveNumber()

        def __int__(self, position, velocity):
            super().__int__(position)
            self.velocity = velocity
            self.flying_object_list.append(self)

        def setup_boundary(self, boundary):
            self.boundary = boundary / 2

        def is_boundary_breach(self, tentative_next_position):
            return True if any(tentative_next_position > self.boundary) or any(
                tentative_next_position < -self.boundary) else False

    class Baloon(NCFO):

        def __int__(self, position, speed, velocity):
            super().__int__(position, velocity)
            self.flying_object_list.append(self)

        def update_position(self):
            while True:
                tentative_next_position = self.position + self.direction * self.speed
                if not self.is_boundary_breach(tentative_next_position):
                    self.position = tentative_next_position
                    break
                v = np.random.randn(3)

    class Flight(NCFO):
        pass
