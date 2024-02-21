# from pettingzoo.mpe import simple_adversary_v3
import collections
import json
import multiprocessing
import pickle
from itertools import product
import numpy as np
import air_corridor.d3.scenario.D3shapeMove as d3
from air_corridor.tools.util import load_init_params
from rl_multi_3d_trans.ppo import PPO

env = d3.parallel_env(render_mode="")
max_round = 2000

def process_combination(combination):
    model_key, num_agents, level = combination
    #########################################################
    # print(os.getcwd() + "\n")
    if model_key == '1-1':
        loadFolder = '../well trained models/num1, train1'
        modelINdex = '9.75m'
        net_model = 'fc'
        trained_level = 2
        ratio = 0.8
    elif model_key == '1-2':
        loadFolder = '../well trained models/num1, train2'
        modelINdex = '5.75m'
        net_model = 'fc'
        trained_level = 3
    elif model_key == '1-3':
        loadFolder = '../well trained models/num1, train3'
        # modelINdex = '3.25m'
        modelINdex = '5.75m'
        net_model = 'fc'
        trained_level = 13
    elif model_key == '1-4':
        loadFolder = '../well trained models/num1, train4'
        modelINdex = '6.5m'
        net_model = 'fc'
        trained_level = 15


    kwargs = load_init_params(name='net_params', dir=loadFolder)
    opt = load_init_params(name='main_params', dir=loadFolder)
    kwargs['net_model'] = net_model
    model = PPO(**kwargs)
    try:
        model.load(folder=loadFolder, global_step=modelINdex)
    except:
        return
    # ani = vl(max_round, to_base=False)
    status = {}
    for z in range(max_round):
        # print(f"{i}/{max_round}")
        '''
        training scenarios can be different from test scenarios, so num_corridor_in_state and corridor_index_awareness
        need to match the setting for UAV during training.
        '''
        s, infos = env.reset(num_agents=num_agents,
                             level=level,
                             dt=opt['dt'],
                             num_corridor_in_state=opt['num_corridor_in_state'],
                             corridor_index_awareness=opt['corridor_index_awareness'],
                             test=True)
        step = 0
        # ani.put_data(agents={agent: agent.position for agent in env.agents}, corridors=env.corridors, round=z)
        while env.agents:
            s1 = {agent: s[agent]['self'] for agent in env.agents}
            s2 = {agent: s[agent]['other'] for agent in env.agents}
            s1_lst = [state for agent, state in s1.items()]
            s2_lst = [state for agent, state in s2.items()]
            a_lst, logprob_a_lst = model.evaluate(s1_lst, s2_lst)
            # a_lst, logprob_a_lst, alpha, beta = model.select_action(s1_lst, s2_lst)
            actions = {agent: a for agent, a in zip(env.agents, a_lst)}
            s, rewards, terminations, truncations, infos = env.step(actions)
            # ani.put_data(round=z, agents={agent: agent.position for agent in env.agents})
            for agent in env.agents:
                if agent.status != 'Normal' and agent not in status:
                    status[agent] = agent.status
            env.agents = [agent for agent in env.agents if not agent.terminated]
            step += 1
            # print(step)

    state_count = collections.Counter(status.values())
    total_agents = num_agents * max_round
    won_rate = round(state_count['won'] / total_agents, 2)
    print(f"{combination}  {trained_level}  {state_count['won']}/{total_agents}  {won_rate}")
    return model_key, num_agents, level, state_count['won'], won_rate


# ani.show_animation()
# Specify the file name where you want to save the array

def main():
    multiprocessing.set_start_method('spawn', force=True)

    # Your existing setup code
    model_set = reversed(['1-1', '1-2', '1-3', '1-4'])
    level_set = reversed([10, 11, 12, 13, 14, 15])
    agents_dic = {j: i for i, j in enumerate(range(2, 10))}
    # level_set = [14,15]
    # agents_dic = {j: i for i, j in enumerate(range(9, 10))}
    model_dic = {j: i for i, j in enumerate(model_set)}
    level_dic = {j: i for i, j in enumerate(level_set)}
    combinations = list(product(model_dic.keys(), agents_dic.keys(), level_dic.keys()))

    # Limit the number of processes to 4
    with multiprocessing.Pool(processes=12) as pool:
        results = pool.map(process_combination, combinations)
        print(results)

    with open('./plot/test_data.json', 'w') as file:
        json.dump(results, file, indent=4)

    won_matrix = np.zeros([len(model_dic.keys()), len(agents_dic.keys()), len(level_dic.keys())])
    won_rate_matrix = np.zeros([len(model_dic.keys()), len(agents_dic.keys()), len(level_dic.keys())])
    for result in results:
        if result:
            model_key, num_agents, level_key, won_value, won_rate = result
            won_matrix[model_dic[model_key], agents_dic[num_agents], level_dic[level_key]] = won_value
            won_rate_matrix[model_dic[model_key], agents_dic[num_agents], level_dic[level_key]] = won_rate

    arrays = {'model': model_dic,
              'agent': agents_dic,
              'level': level_dic,
              'won_times': won_matrix,
              'won_rate': won_rate_matrix}
    with open('./plot/array.pkl', 'wb') as f:
        pickle.dump(arrays, f)

    # file_name = 'test_data.npy'
    # np.save(file_name, won_record)


if __name__ == "__main__":
    main()
