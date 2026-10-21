from .node import MCTSNode, INF
from .config import MCTSConfig
from env.base_env import BaseGame, EPS

import numpy as np

class UCTMCTSConfig(MCTSConfig):
    def __init__(
        self,
        n_rollout:int = 1,
        *args, **kwargs
    ):
        MCTSConfig.__init__(self, *args, **kwargs)
        self.n_rollout = n_rollout


class UCTMCTS:
    def __init__(self, init_env:BaseGame, config: UCTMCTSConfig, root:MCTSNode=None):
        self.config = config
        self.root = root
        if root is None:
            self.init_tree(init_env)
        self.root.cut_parent()
    
    def init_tree(self, init_env:BaseGame):
        # initialize the tree with the current state
        # fork the environment to avoid side effects
        env = init_env.fork()
        self.root = MCTSNode(
            action=None, env=env, reward=0,
        )
    
    def get_subtree(self, action:int):
        # return a subtree with root as the child of the current root
        # the subtree represents the state after taking action
        if self.root.has_child(action):
            new_root = self.root.get_child(action)
            return UCTMCTS(new_root.env, self.config, new_root)
        else:
            return None
    
    def uct_action_select(self, node:MCTSNode) -> int:
        
        # select the best action based on UCB when expanding the tree
        # TODO：只在 action_mask 允许的动作中计算 UCT 并返回最优动作。
        # 提示：未访问的合法动作应优先被选中；已访问动作的利用项为
        # child_V_total / child_N_visit，探索项使用 config.C 和父节点总访问次数。
        # 非法动作绝对不能被返回。
        
        raise NotImplementedError("TODO: 完成 UCT 动作选择")

    def backup(self, node:MCTSNode, value:float) -> None:
        # backup the value of the leaf node to the root
        # update N_visit and V_total of each node in the path
        # 提示：传入的 value 是叶节点“当前行动方”视角的价值，而父节点边上的
        # 统计量属于在父节点选择该动作的一方。每向上经过一条边都要转换一次
        # 玩家视角，并让该边的访问次数恰好增加 1。
        
        # TODO：从 node 沿 parent 指针回传到根节点。
        # 提示：统计量保存在父节点中对应 action 的 child_N_visit 和
        # child_V_total 上；每向上经过一条边，价值取反一次。
        raise NotImplementedError("TODO: 完成 MCTS 价值回传")
            
    
    def rollout(self, node:MCTSNode) -> float:
        # simulate the game until the end
        # return the reward of the game
        # NOTE: the reward should be convert to the perspective of the current player!
        # 提示：必须先 fork()，再从合法动作中随机落子直到终局，不能修改树节点
        # 保存的环境。step() 的终局 reward 属于刚刚落子的玩家；返回前应将它
        # 转换成 rollout 开始时 node.env.current_player 的视角。和棋的 EPS
        # 是中性标记，无论落子奇偶都直接返回 EPS。
        
        # TODO：在 node.env.fork() 得到的副本上随机选择合法动作，直到终局。
        # 返回值必须转换为 rollout 开始时 node.env.current_player 的视角。
        # 注意：step() 的终局 reward 属于“刚刚落子的玩家”，但 EPS 直接保持不变。
        raise NotImplementedError("TODO: 完成随机模拟")
    
    def pick_leaf(self) -> MCTSNode:
        # select the leaf node to expand
        # the leaf node is the node that has not been expanded
        # create and return a new node if game is not ended
        # 提示：从根节点开始用 uct_action_select() 选择动作；若对应孩子已经
        # 存在就继续向下，否则用 add_child() 扩展并立即返回新节点。若到达
        # 终局节点，则直接返回该终局节点。
        
        # TODO：从根节点开始重复执行 UCT 选择。若子节点已存在则继续向下；
        # 若对应合法动作尚未扩展，调用 add_child(action) 并立即返回新节点。
        # 若到达终局节点，直接返回它。
        raise NotImplementedError("TODO: 完成叶节点选择与扩展")
    
    def get_policy(self, node:MCTSNode = None) -> np.ndarray:
        # return the policy of the tree(root) after the search
        # the policy comes from the visit count of each action
        # 提示：本题直接将合法动作的访问次数归一化为概率；非法动作概率为 0。
        # 本次作业不要求使用 config 中的 temperature 或噪声相关参数。
        
        # TODO：将合法动作的访问次数归一化为概率分布。
        # 提示：返回长度应为 node.n_action，非法动作概率必须为 0；
        # 当所有合法动作的访问数仍为 0 时，可在合法动作上返回均匀分布。
        raise NotImplementedError("TODO: 完成根节点策略计算")

    def search(self):
        # search the tree for n_search times
        # each time, pick a leaf node, rollout the game (if game is not ended)
        #   for n_rollout times, and backup the value.
        # return the policy of the tree after the search
        for _ in range(self.config.n_search):
            leaf = self.pick_leaf()
            value = 0
            if leaf.done:
                # 提示：leaf.reward 属于生成该叶节点时刚刚落子的玩家，而此时
                # leaf.env.current_player 已经切换为另一方；传给 backup 前需要
                # 转换为叶节点当前行动方视角。
                # TODO：将 leaf.reward 转换为叶节点当前行动方视角的 value。
                # 提示：和棋值 EPS 不需要改变正负。
                raise NotImplementedError("TODO: 处理终局叶节点价值")
            else:
                # 提示：独立执行 n_rollout 次 rollout，取算术平均作为本轮叶节点
                # 价值，然后只调用一次 backup。因此每轮搜索的访问次数只增加 1。
                # TODO：独立运行 n_rollout 次 rollout，将算术平均值保存到 value。
                # 注意：多次 rollout 后只在下方调用一次 backup。
                raise NotImplementedError("TODO: 计算多次 rollout 的平均价值")
            self.backup(leaf, value)

        return self.get_policy(self.root)
