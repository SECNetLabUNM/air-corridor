import functools
from functools import reduce

import gymnasium as gym
import pygame
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from air_corridor.d3.corridor.corridor import CylinderCorridor, DirectionalPartialTorusCorridor
from air_corridor.d3.geometry.FlyingObject import UAV
from air_corridor.tools.util import *


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "rps_v2"}

    def __init__(self,
                 render_mode=None,
                 reduce_space=True):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.state = None
        self.env_moves = None
        self.corridors = None
        self.render_mode = render_mode
        self.isopen = True
        self.distance_map = None
        self.reduce_space = True
        self.liability = False
        self.collision_free = False
        self.dt = 1
        self.consider_boid = False

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return spaces.Dict(
            {'self': spaces.Box(low=-100, high=100, shape=(16 + 10,), dtype=np.float32),
             'other': spaces.Box(low=-100, high=100, shape=(22, (self.num_agents - 1)), dtype=np.float32)})

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """

        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if not hasattr(self, 'screen') or self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (SCREEN_WIDTH, SCREEN_HEIGHT)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        if not hasattr(self, 'clock') or self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.surf.fill(WHITE)

        for _, one_corridor in self.corridors.items():
            one_corridor.render_self(self.surf)
        for agent in self.agents:
            agent.render_self(self.surf)

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            # self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        try:
            if self.screen is not None:
                import pygame
                pygame.display.quit()
                pygame.quit()
                self.isopen = False
        except:
            pass

    def update_distance_map(self):
        count = len(self.agents)
        self.distance_map = np.ones([count, count]) / 1e-5

        for i in range(count):
            if self.agents[i].terminated:
                continue
            for j in range(i + 1, count):
                if self.agents[j].terminated:
                    continue
                dis = self.agents[i].get_distance_to(self.agents[j])
                self.distance_map[i, j] = dis

    def collision_detection(self, collisiion_distance=0.4):
        index = np.where(self.distance_map < collisiion_distance)
        collide_set = set(reduce((lambda x, y: x + y), [list(i) for i in index]))
        for i in collide_set:
            if not self.agents[i].terminated:
                self.agents[i].status = 'collided'

    def access_neighbor_info(self):
        info = []
        for agent_i in self.agents:
            single_info = []
            for agent_j in self.agents:
                if agent_i is agent_j:
                    continue
                else:
                    single_info.append(list(self.agent_j.position - agent_i.position) + list(self.agent_j.velocity))
            info.append(single_info)
        return info

    def random_combination(self, ratio, num):
        seq = []
        for i in range(num):
            if random.random() < ratio:
                seq.append('t')
            else:
                seq.append('c')
        return tuple(seq)

    def generate_structure(self, difficulty=1, seq=None, minor_radius=2.0, test=False):
        '''
        :param connect_plane_anchor: in base,
        :param connect_plane_orientation: in base,
        :param rotation_matrix: base to remote,
        :param anchor_point: base to remote,
        :return:
        1e-3
        '''
        if seq is not None:
            num = len(seq)
        for i in range(num):
            non_last_flag = num > i + 1
            name = chr(65 + i)
            minor_radius = minor_radius
            if i == 0:
                intial_anchor = np.random.rand(3) * 2
                initial_orientation_rad = [random.random() * np.pi, (random.random() - 0.5) * 2 * np.pi]  # theta,phi
                if seq[i] == 'c':
                    cor = CylinderCorridor(anchor_point=intial_anchor,
                                           orientation_rad=initial_orientation_rad,
                                           length=random_(difficulty, epsilon=self.epsilon, segment=True) * 18 + 2,
                                           width=minor_radius * 2,
                                           name=name,
                                           connections=[], reduce_space=self.reduce_space)
                    if non_last_flag:
                        connect_plane_orientation = cor.orientation_vec
                        connect_plane_anchor = cor.endCirclePlane.anchor_point - CORRIDOR_OVERLAP * connect_plane_orientation
                else:
                    begin_rad = np.pi * (2 * random.random() - 1)
                    if test:
                        end_rad = begin_rad + np.pi / 2
                    else:
                        end_rad = begin_rad + np.pi / 2 * random_(difficulty, epsilon=self.epsilon)
                    major_radius = 5 * (random.random() + 1)
                    cor = DirectionalPartialTorusCorridor(name=name,
                                                          anchor_point=intial_anchor,
                                                          orientation_rad=initial_orientation_rad,
                                                          major_radius=major_radius,
                                                          minor_radius=minor_radius,
                                                          begin_rad=begin_rad,
                                                          end_rad=end_rad,
                                                          connections=[],
                                                          reduce_space=self.reduce_space)
                    if non_last_flag:
                        connect_plane_orientation = cor.endCirclePlane.orientation_vec
                        connect_plane_anchor = cor.endCirclePlane.anchor_point - CORRIDOR_OVERLAP * connect_plane_orientation
                if non_last_flag:
                    cor.connections = ['B']
                    rotate_to_end_plane = cor.endCirclePlane.rotate_to_remote
                    # rotate_to_end_plane1 = cor.endCirclePlane.rotate_to_base
            else:
                if seq[i] == 'c':
                    length = random_(difficulty, epsilon=self.epsilon, segment=True) * 18 + 2
                    cor = CylinderCorridor(anchor_point=connect_plane_anchor + connect_plane_orientation * length / 2,
                                           orientation_vec=connect_plane_orientation,
                                           length=length,
                                           width=minor_radius * 2,
                                           name=name,
                                           connections=[], reduce_space=self.reduce_space)
                    if non_last_flag:
                        connect_plane_orientation = cor.orientation_vec
                        connect_plane_anchor = cor.endCirclePlane.anchor_point - CORRIDOR_OVERLAP * connect_plane_orientation
                else:
                    if seq[i - 1] == 't':
                        major_radius = self.corridors[chr(65 + i - 1)].major_radius
                    else:
                        major_radius = 5 * (random.random() + 1)
                    connect_plane_x = rotate_to_end_plane(X_UNIT)
                    connect_plane_y = rotate_to_end_plane(Y_UNIT)
                    random_rad = (random.random() * 2 - 1) * np.pi
                    unit_vec_connect_point_to_new_obj_anchor = (connect_plane_y * np.sin(random_rad) +
                                                                connect_plane_x * np.cos(random_rad))
                    new_obj_anchor = connect_plane_anchor + unit_vec_connect_point_to_new_obj_anchor * major_radius

                    orientation_vec = np.cross(-unit_vec_connect_point_to_new_obj_anchor, connect_plane_orientation)

                    new_obj_to_base_matrix = vec2vec_rotation(orientation_vec, Z_UNIT)
                    vec_on_base = np.dot(new_obj_to_base_matrix, -unit_vec_connect_point_to_new_obj_anchor)

                    begin_rad = np.arctan2(vec_on_base[1], vec_on_base[0])
                    if test:
                        end_rad = begin_rad + np.pi / 2
                    else:
                        end_rad = begin_rad + np.pi / 2 * random_(difficulty, epsilon=self.epsilon)

                    cor = DirectionalPartialTorusCorridor(name=name,
                                                          anchor_point=new_obj_anchor,
                                                          orientation_vec=orientation_vec,
                                                          major_radius=major_radius,
                                                          minor_radius=minor_radius,
                                                          begin_rad=begin_rad,
                                                          end_rad=end_rad,
                                                          connections=[],
                                                          reduce_space=self.reduce_space)
                    if non_last_flag:
                        connect_plane_orientation = cor.endCirclePlane.orientation_vec
                        connect_plane_anchor = cor.endCirclePlane.anchor_point - CORRIDOR_OVERLAP * connect_plane_orientation
                if non_last_flag:
                    rotate_to_end_plane = cor.endCirclePlane.rotate_to_remote
                    # rotate_to_end_plane1 = cor.endCirclePlane.rotate_to_base
                    cor.connections = [chr(65 + i + 1)]

            self.corridors[name] = cor

    def reset(self,
              seed=None,
              options=None,
              num_agents=3,
              reduce_space=True,
              level=10,
              ratio=1,
              liability=True,
              collision_free=False,
              beta_adaptor_coefficient=1.0,
              num_corridor_in_state=1,
              dt=1.0,
              consider_boid=False,
              corridor_index_awareness=False,
              velocity_max=1.5,
              acceleration_max=0.3,
              uniform_state=False,
              minor_radius_test=2.0,
              dynamic_minor_radius=False,
              epsilon=0.1,
              test=False):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `env_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.epsilon = epsilon
        self.dt = dt
        self.consider_boid = consider_boid
        self.liability = liability
        self.collision_free = collision_free
        self.reduce_space = reduce_space
        # setup corridors
        difficulty = 1 if options is None else options['difficulty']

        self.corridors = {}
        # the following 4 parameters used for generating training env only

        if level == 0:
            begin_rad = -np.pi
            end_rad = begin_rad + np.pi / 2
            major_radius = 10
            # orientation_rad = [0, 0]  #  theta,phi
        elif level == 1:
            begin_rad = np.pi * (2 * random.random() - 1)
            if difficulty <= 1:
                end_rad = begin_rad + np.pi / 2 * (difficulty + random.uniform(-0.1, 0.1))
            else:
                end_rad = begin_rad + np.pi / 2 * random.uniform(0.9, difficulty + 0.1)
            major_radius = 10
            orientation_rad = [random.random() * np.pi, (random.random() - 0.5) * 2 * np.pi]  # theta,
            if random.random() > ratio:
                self.corridors['A'] = CylinderCorridor(anchor_point=np.array([0, 0, 0]),
                                                       orientation_rad=orientation_rad,
                                                       length=random.random() * difficulty * 15 + 5,
                                                       width=4,
                                                       name='A',
                                                       connections=[], reduce_space=self.reduce_space)
            else:
                self.corridors['A'] = DirectionalPartialTorusCorridor(name='A',
                                                                      anchor_point=np.array([0, 0, 0]),
                                                                      orientation_rad=orientation_rad,
                                                                      major_radius=major_radius,
                                                                      minor_radius=2,
                                                                      begin_rad=begin_rad,
                                                                      end_rad=end_rad,
                                                                      connections=[],
                                                                      reduce_space=self.reduce_space)
        elif level == 2:
            # fixed ending degree and fixed radius, but gradually increase fixed ending degree
            seq = self.random_combination(ratio, num=1)
            self.generate_structure(difficulty, seq, test=test)
        elif level == 3:
            seq = random.choice([('t', 't'), ('t', 'c'), ('c', 't')])
            self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 10:
            seq = random.choice([('t'), ('c')])
            self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 11:
            seq = random.choice([('t', 't'), ('t', 'c'), ('c', 't')])
            self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 12:
            seq = random.choice([('t', 't', 'c'), ('c', 't', 't')])
            self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 13:
            if dynamic_minor_radius:
                minor_radius = np.random.uniform(1.8, 2.2)
            else:
                minor_radius = 2
            seq = random.choices([('t', 't', 'c'), ('c', 't', 't'), ('t', 'c', 't'), ('c', 't', 'c')],
                                 weights=[1.0, 1.0, 0.8, 0.8])[0]
            self.generate_structure(difficulty, seq=seq, test=test, minor_radius=minor_radius)
        elif level == 14:
            seq = ('c', 't', 't', 'c')
            self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 15:
            seq = random.choice([('c', 't', 't', 'c'), ('t', 'c', 't', 'c'), ('c', 't', 'c', 't'), ('t', 't', 'c', 't'),
                                 ('t', 'c', 't', 't')])
            self.generate_structure(difficulty, seq=seq, test=test)
        if not test and corridor_index_awareness:
            assert len(seq) >= sum(corridor_index_awareness)
        corridor_graph = self.corridors['A'].convert2graph(self.corridors)
        DirectionalPartialTorusCorridor.num_corridor_in_state = num_corridor_in_state
        CylinderCorridor.num_corridor_in_state = num_corridor_in_state

        # setup uavs
        plane_offsets = distribute_evenly_within_circle(radius=2, min_distance=1, num_points=num_agents)
        UAV.flying_list = []
        if len(self.corridors) == 1:
            self.agents = [UAV(init_corridor='A',
                               des_corridor='A',
                               name=None,
                               plane_offset_assigned=plane_offset,
                               velocity_max=velocity_max,
                               acceleration_max=acceleration_max) for plane_offset in plane_offsets]
        else:
            self.agents = [UAV(init_corridor='A',
                               des_corridor=chr(64 + len(self.corridors)),
                               name=None,
                               plane_offset_assigned=plane_offset,
                               velocity_max=velocity_max,
                               acceleration_max=acceleration_max) for plane_offset in plane_offsets]
        UAV.corridors = self.corridors
        UAV.reduce_space = reduce_space
        UAV.corridor_graph = corridor_graph
        UAV.beta_adaptor_coefficient = beta_adaptor_coefficient
        UAV.num_corridor_in_state = num_corridor_in_state
        UAV.capacity = max(num_agents, 2)
        UAV.corridor_index_awareness = corridor_index_awareness
        # index capability with 4 bits
        # index up to 2: [1,0,0,1]; up to 3: [1,1,0,1]; up to 4: [1,1,1,1].
        UAV.corridor_state_length = 20 if corridor_index_awareness else 16
        UAV.uniform_state = uniform_state

        [agent.reset() for agent in self.agents]
        self.env_moves = 0
        observations = {agent: agent.report() for agent in self.agents}
        self.state = observations
        if self.render_mode == "human":
            self.render()
        infos = {'corridor_seq': seq}
        return observations, infos

    def step(self, action_dic):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """

        rewards = {agent: agent.take(action, self.dt) for agent, action in action_dic.items()}

        # collision detection
        if not self.collision_free:
            self.update_distance_map()
            self.collision_detection()

        disaster = False
        for agent, _ in rewards.items():
            if not agent.terminated:
                if agent.status == 'collided':
                    reward_from_corridor = PENALTY_COLLISION
                else:
                    reward_from_corridor = agent.corridors[agent.enroute['current']].evaluate_action(agent)
                rewards[agent] += reward_from_corridor

                agent.instant_reward = rewards[agent]

        for agent in self.agents:
            if not agent.terminated:
                # if agent.status in UAV.events:
                if agent.status != 'Normal':
                    disaster = True
                    break

        for agent in self.agents:
            if not agent.terminated:
                if self.liability and disaster and agent.status != 'Normal':
                    rewards[agent] = rewards[agent] + LIABILITY_PENALITY
                agent.update_position()
                agent.update_accumulated_reward()

        self.env_moves += 1
        env_truncation = self.env_moves >= NUM_ITERS
        truncations = {agent: env_truncation for agent in self.agents}

        # terminations = {agent: agent.status in UAV.events for agent in self.agents}
        terminations = {}
        for agent in self.agents:
            # agent.terminated = agent.status in UAV.events
            agent.terminated = agent.status != 'Normal'
            terminations[agent] = agent.terminated

        observations = {agent: agent.report() for agent in self.agents}

        self.state = observations

        if self.render_mode == "human":
            self.render()
        infos = {agent: None for agent in self.agents}

        return observations, rewards, terminations, truncations, infos
