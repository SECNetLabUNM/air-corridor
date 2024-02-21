import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import Dataset
import sys

sys.path.append('/home/kun/PycharmProjects/air-corridor/')
from rl_multi_3d_trans import net_nn_fc

net_models = {
    'fc': net_nn_fc,
    # 'fc1': net_nn_fc1,
    # 'fc2': net_nn_fc_2,
    # 'fc2_1': net_nn_fc_2_1,
    # 'fc2_2': net_nn_fc_2_2,
    # 'fc3': net_nn_fc_3,
    # 'tran': net_nn_tran,
    # 'tran2': net_nn_tran2,
    # 'tran2_1': net_nn_tran2_1,
    # 'tran2_1_1': net_nn_tran2_1_1,
    # 'tran2_1_3': net_nn_tran2_1_3,
    # 'tran2_1_4': net_nn_tran2_1_4,
    # 'tran2_1_2': net_nn_tran2_1_2,
    # 'tran2_2': net_nn_tran2_2,
    # 'tran3': net_nn_tran3,
    # 'tran4': net_nn_tran4,
    # add more mappings as needed
}


class MyDataset(Dataset):
    def __init__(self, data, env_with_Dead=True):
        self.data = data
        self.env_with_Dead = env_with_Dead

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        transition = self.data[idx]
        s1, s2, a, r, s1_prime, s2_prime, logprob_a, done, dw, td_target, adv = transition

        # If your environment does not include Dead, modify dw here
        if self.env_with_Dead:  # Replace with your condition
            dw = False

        # return {
        #     's1': s1, 's2': s2, 'a': a, 'r': [r], 's1_prime': s1_prime,
        #     's2_prime': s2_prime, 'logprob_a': logprob_a,
        #     'done': [done], 'dw': [dw],
        #     'adv': adv, 'td_target':td_target
        # }

        return {
            's1': torch.tensor(s1, dtype=torch.float),
            's2': torch.tensor(s2, dtype=torch.float),
            'a': torch.tensor(a, dtype=torch.float),
            'r': torch.tensor([r], dtype=torch.float),
            's1_prime': torch.tensor(s1_prime, dtype=torch.float),
            's2_prime': torch.tensor(s2_prime, dtype=torch.float),
            'logprob_a': torch.tensor(logprob_a, dtype=torch.float),
            'done': torch.tensor([done], dtype=torch.float),
            'dw': torch.tensor([dw], dtype=torch.float),
            'td_target': torch.tensor(td_target, dtype=torch.float),
            'adv': torch.tensor(adv, dtype=torch.float),

        }


class PPO(object):
    def __init__(
            self,
            state_dim=26,
            s2_dim=22,
            action_dim=3,
            env_with_Dead=True,
            gamma=0.99,
            lambd=0.95,
            # gamma=0.89,
            # lambd=0.88,
            clip_rate=0.2,
            K_epochs=10,
            net_width=256,
            a_lr=3e-4,
            c_lr=3e-4,
            l2_reg=1e-3,
            dist='Beta',
            a_optim_batch_size=64,
            c_optim_batch_size=64,
            entropy_coef=0,
            entropy_coef_decay=0.9998,
            writer=None,
            activation=None,
            share_layer_flag=True,
            anneal_lr=True,
            totoal_steps=0,
            with_position=False,
            token_query=False,
            num_enc=5,
            logger=None,
            dir=None,
            test=False,
            net_model='fc1',
            beta_base=1e-5
    ):
        # if net_model == 'fc':
        #     from rl_multi_3d_trans.net_nn_fc import BetaActorMulti, CriticMulti, MergedModel
        # elif net_model == 'fc1':
        #     from rl_multi_3d_trans.net_nn_fc import BetaActorMulti, CriticMulti, MergedModel
        # elif net_model == 'tran':
        #     from rl_multi_3d_trans.net_nn_tran import BetaActorMulti, CriticMulti, MergedModel
        # elif net_model == 'tran2':
        #     from rl_multi_3d_trans.net_nn_tran2 import BetaActorMulti, CriticMulti, MergedModel
        # elif net_model == 'tran3':
        #     from rl_multi_3d_trans.net_nn_tran3 import BetaActorMulti, CriticMulti, MergedModel
        # elif net_model == 'tran4':
        #     from rl_multi_3d_trans.net_nn_tran3 import BetaActorMulti, CriticMulti, MergedModel

        self.dir = dir
        self.logger = logger
        self.share_layer_flag = share_layer_flag
        shared_layers_actor = net_models[net_model].MergedModel(s1_dim=state_dim, s2_dim=s2_dim, net_width=net_width,
                                                                with_position=with_position, token_query=token_query,
                                                                num_enc=num_enc)
        if share_layer_flag:
            shared_layers_critic = shared_layers_actor
        else:
            shared_layers_critic = net_models[net_model].MergedModel(s1_dim=state_dim, s2_dim=s2_dim,
                                                                     net_width=net_width,
                                                                     with_position=with_position,
                                                                     token_query=token_query,
                                                                     num_enc=num_enc)
        self.dist = dist
        self.env_with_Dead = env_with_Dead
        self.action_dim = action_dim
        self.clip_rate = clip_rate
        self.gamma = gamma
        self.lambd = lambd
        self.clip_rate = clip_rate
        self.K_epochs = K_epochs
        self.data = {}
        self.l2_reg = l2_reg
        self.a_optim_batch_size = a_optim_batch_size
        self.c_optim_batch_size = c_optim_batch_size
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.writer = writer
        self.anneal_lr = anneal_lr
        self.totoal_steps = totoal_steps
        self.a_lr = a_lr
        self.c_lr = c_lr
        # if not test:
        self.actor = net_models[net_model].BetaActorMulti(state_dim, s2_dim, action_dim, net_width,
                                                          shared_layers_actor, beta_base).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.critic = net_models[net_model].CriticMulti(state_dim, s2_dim, net_width, shared_layers_critic).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

    def load_pretrained(self):
        pass

    def select_action(self, s1, s2):  # only used when interact with the env
        self.actor.eval()
        with torch.no_grad():
            # if not isinstance(state, np.ndarray):
            #     state = np.array(state)
            # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            # print(f"s1, {len(s1[-1])}")
            # print(f"s2, {len(s2[-1][-1])}")
            s1 = np.array(s1)
            s1 = torch.FloatTensor(s1).to(device)
            s2 = np.array(s2)
            s2 = torch.FloatTensor(s2).to(device)

            dist, alpha, beta, nan_event = self.actor.get_dist(s1, s2, self.logger)

            assert torch.all((0 <= alpha))
            assert torch.all((0 <= beta))
            if nan_event:
                self.save('nan')
                sys.exit()
            a = dist.sample()
            assert torch.all((0 <= a)) and torch.all(a <= 1)
            a = torch.clamp(a, 0, 1)
            logprob_a = dist.log_prob(a).cpu().numpy()
            return a.cpu().numpy(), logprob_a, alpha, beta

    def evaluate(self, s1, s2):  # only used when evaluate the policy.Making the performance more stable
        self.actor.eval()
        with torch.no_grad():
            s1 = np.array(s1)
            s1 = torch.FloatTensor(s1).to(device)
            s2 = np.array(s2)
            s2 = torch.FloatTensor(s2).to(device)
            if self.dist == 'Beta':
                a = self.actor.dist_mode(s1, s2)
            if self.dist == 'GS_ms':
                a, b = self.actor(s1)
            if self.dist == 'GS_m':
                a = self.actor(s1)
            return a.cpu().numpy(), 0.0

    def train(self, global_step, epoches=None):

        if self.anneal_lr:
            frac = 1.0 - global_step / self.totoal_steps
            alrnow = frac * self.a_lr
            clrnow = frac * self.c_lr
            self.actor_optimizer.param_groups[0]["lr"] = alrnow
            self.critic_optimizer.param_groups[0]["lr"] = clrnow

        self.entropy_coef *= self.entropy_coef_decay

        transitions = self.gae()
        dataset = MyDataset(transitions)

        dataloader = DataLoader(dataset, batch_size=self.a_optim_batch_size, shuffle=True, drop_last=True)
        # s1, s2, a, r, s1_prime, s2_prime, logprob_a, done_mask, dw_mask = self.make_batch()
        #
        # ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        # self.critic.eval()
        # with torch.no_grad():
        #     vs = self.critic(s1, s2)
        #     vs_ = self.critic(s1_prime, s2_prime)
        #
        #     '''dw for TD_target and Adv'''
        #     deltas = r + self.gamma * vs_ * (1 - dw_mask) - vs
        #
        #     deltas = deltas.cpu().flatten().numpy()
        #     adv = [0]
        #
        #     '''done for GAE'''
        #     for dlt, mask in zip(deltas[::-1], done_mask.cpu().flatten().numpy()[::-1]):
        #         advantage = dlt + self.gamma * self.lambd * adv[-1] * (1 - mask)
        #         adv.append(advantage)
        #     adv.reverse()
        #     adv = copy.deepcopy(adv[0:-1])
        #     adv = torch.tensor(adv).unsqueeze(1).float().to(device)
        #     td_target = adv + vs
        #     adv = (adv - adv.mean()) / (adv.std() + 1e-6)  # sometimes helps

        """Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
        # a_optim_iter_num = int(math.ceil(s1.shape[0] / self.a_optim_batch_size))
        # c_optim_iter_num = int(math.ceil(s1.shape[0] / self.c_optim_batch_size))

        clipfracs = []
        for i in range(epoches):

            # Shuffle the trajectory, Good for training
            # perm = np.arange(s1.shape[0])
            # np.random.shuffle(perm)
            # perm = torch.LongTensor(perm).to(device)
            # s1 = s1[perm].clone()
            # s2 = s2[perm].clone()
            # a = a[perm].clone()
            # td_target = td_target[perm].clone()
            # adv = adv[perm].clone()
            # logprob_a = logprob_a[perm].clone()

            '''update the actor-critic'''
            self.actor.train()
            self.critic.train()
            # for i in range(a_optim_iter_num):
            for batch in dataloader:
                s1 = batch['s1'].to(device)
                s2 = batch['s2'].to(device)
                a = batch['a'].to(device)
                # r = batch['r'].to(device)
                # s1_prime = batch['s1_prime'].to(device)
                # s2_prime = batch['s2_prime'].to(device)
                logprob_a = batch['logprob_a'].to(device)
                # done_mask = batch['done_mask'].to(device)
                # dw_mask = batch['dw_mask'].to(device)
                adv = batch['adv'].to(device)
                td_target = batch['td_target'].to(device)

                '''derive the actor loss'''
                # index = slice(i * self.a_optim_batch_size, min((i + 1) * self.a_optim_batch_size, s1.shape[0]))
                distribution, _, _, nan_event = self.actor.get_dist(s1, s2, self.logger)
                if nan_event:
                    self.save('nan')
                dist_entropy = distribution.entropy().sum(1, keepdim=True)
                logprob_a_now = distribution.log_prob(a)

                logratio = logprob_a_now.sum(1, keepdim=True) - logprob_a.sum(1, keepdim=True)
                ratio = torch.exp(logratio)  # a/b == exp(log(a)-log(b))

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [torch.mean((ratio - 1.0).abs() > self.clip_rate, dtype=torch.float32).item()]

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv
                pg_loss = -torch.min(surr1, surr2)
                a_loss = pg_loss - self.entropy_coef * dist_entropy

                '''derive the critic loss'''
                # index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, s1.shape[0]))
                c_loss = (self.critic(s1, s2) - td_target).pow(2).mean()
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                '''updata parameters'''
                self.actor_optimizer.zero_grad()
                a_loss.mean().backward(retain_graph=self.share_layer_flag)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                self.critic_optimizer.step()

        # y_pred, y_true = vs.cpu().numpy(), td_target.cpu().numpy()
        # var_y = np.var(y_true)
        # explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        # self.actor_cpu = copy.deepcopy(self.actor).to('cpu')
        # self.writer.add_scalar("charts/averaged_accumulated_reward", sum(accumulated_reward) / len(accumulated_reward),
        #                   global_step)
        self.writer.add_scalar("weights/critic_learning_rate", self.critic_optimizer.param_groups[0]["lr"], global_step)
        self.writer.add_scalar("losses/value_loss", c_loss.item(), global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.mean().item(), global_step)
        self.writer.add_scalar("losses/entropy", dist_entropy.mean().item(), global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # self.writer.add_scalar("losses/explained_variance", explained_var, global_step)

        # del s1, s2, a, r, s1_prime, s2_prime, logprob_a, done_mask, dw_mask
        # del vs, td_target
        del a_loss, c_loss, pg_loss, dist_entropy, old_approx_kl, approx_kl, logprob_a_now, logratio  # , perm
        del surr1, surr2
        torch.cuda.empty_cache()
        self.data = {}

    def make_batch(self, agent):
        s1_lst = []
        s2_lst = []
        a_lst = []
        r_lst = []
        s1_prime_lst = []
        s2_prime_lst = []
        logprob_a_lst = []
        done_lst = []
        dw_lst = []
        for transition in self.data[agent]:
            s1, s2, a, r, s1_prime, s2_prime, logprob_a, done, dw = transition
            s1_lst.append(s1)
            s2_lst.append(s2)
            a_lst.append(a)
            logprob_a_lst.append(logprob_a)
            r_lst.append([r])
            s1_prime_lst.append(s1_prime)
            s2_prime_lst.append(s2_prime)
            done_lst.append([done])
            dw_lst.append([dw])

        if not self.env_with_Dead:
            '''Important!!!'''
            # env_without_DeadAndWin: deltas = r + self.gamma * vs_ - vs
            # env_with_DeadAndWin: deltas = r + self.gamma * vs_ * (1 - dw) - vs
            dw_lst = (np.array(dw_lst) * False).tolist()

        # self.data = []  # Clean history trajectory

        '''list to tensor'''
        with torch.no_grad():
            s1, s2, a, r, s1_prime, s2_prime, logprob_a, done_mask, dw_mask = \
                torch.tensor(np.array(s1_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(s2_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(a_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(r_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(s1_prime_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(s2_prime_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(logprob_a_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(done_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(dw_lst), dtype=torch.float).to(device),
        return s1, s2, a, r, s1_prime, s2_prime, logprob_a, done_mask, dw_mask

    def gae(self, unification=True):
        transitions = []

        collect_adv = []
        for agent in self.data:
            s1, s2, _, r, s1_prime, s2_prime, _, done_mask, dw_mask = self.make_batch(agent)
            ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
            self.critic.eval()
            with torch.no_grad():
                vs = self.critic(s1, s2)
                vs_ = self.critic(s1_prime, s2_prime)
                '''dw for TD_target and Adv'''
                deltas = r + self.gamma * vs_ * (1 - dw_mask) - vs
                deltas = deltas.cpu().flatten().numpy()
                adv = [0]
                '''done for GAE'''
                for dlt, mask in zip(deltas[::-1], done_mask.cpu().flatten().numpy()[::-1]):
                    advantage = dlt + self.gamma * self.lambd * adv[-1] * (1 - mask)
                    adv.append(advantage)
                adv.reverse()
                adv = copy.deepcopy(adv[0:-1])
                collect_adv += adv
                td_target = np.array(adv) + np.array(vs.to('cpu').squeeze(1))
            for i, single_transition in enumerate(self.data[agent]):
                transitions.append(single_transition + [[td_target[i]], adv[i]])
        adv_mean = np.mean(collect_adv)
        adv_std = np.std(collect_adv)
        transitions = [tuple(tran[0:-1] + [[(tran[-1] - adv_mean) / (adv_std + 1e-6)]]) for tran in transitions]
        return transitions

    def put_data(self, agent, transition):
        if agent in self.data:
            self.data[agent].append(transition)
        else:
            self.data[agent] = [transition]

    def save(self, global_step,index=None):
        # global_step is usually interger, but also could be string for some events
        diff=f"_{index}" if index else ''
        if isinstance(global_step, str):
            global_step = global_step
            torch.save(self.critic.state_dict(), f"{self.dir}/ppo_critic_{global_step}{diff}.pth")
            torch.save(self.actor.state_dict(), f"{self.dir}/ppo_actor_{global_step}{diff}.pth")
        else:
            global_step /= 1e6
            torch.save(self.critic.state_dict(), f"{self.dir}/ppo_critic_{global_step}m{diff}.pth")
            torch.save(self.actor.state_dict(), f"{self.dir}/ppo_actor_{global_step}m{diff}.pth")


    def load(self, folder, global_step):
        if isinstance(global_step, float):
            global_step = str(global_step / 1000000) + 'm'
        if folder.startswith('/'):
            self.critic.load_state_dict(torch.load(f"{folder}/ppo_critic_{global_step}.pth"))
            self.actor.load_state_dict(torch.load(f"{folder}/ppo_actor_{global_step}.pth"))
        else:
            self.critic.load_state_dict(torch.load(f"./{folder}/ppo_critic_{global_step}.pth"))
            self.actor.load_state_dict(torch.load(f"./{folder}/ppo_actor_{global_step}.pth"))

    def load_and_copy(self, folder, global_step, a_lr, c_lr):
        if folder.startswith('/'):
            temp_critic = torch.load(f"{folder}/ppo_critic{global_step}.pth")
            temp_actor = torch.load(f"{folder}/ppo_actor{global_step}.pth")
        else:
            temp_critic = torch.load(f"./{folder}/ppo_critic{global_step}.pth")
            temp_actor = torch.load(f"./{folder}/ppo_actor{global_step}.pth")
        for name, param in self.critic.named_parameters():
            if name in temp_critic:
                param.data.copy_(temp_critic[name].data)
        for name, param in self.actor.named_parameters():
            if name in temp_actor:
                param.data.copy_(temp_actor[name].data)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

    def weights_track(self, global_step):
        total_sum = 0.0
        for param in self.actor.parameters():
            total_sum += torch.sum(param)
        self.writer.add_scalar("weights/actor_sum", total_sum, global_step)
        total_sum = 0.0
        for param in self.critic.parameters():
            total_sum += torch.sum(param)
        self.writer.add_scalar("weights/critic_sum", total_sum, global_step)
