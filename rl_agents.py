from overcooked_ai_py.agents.agent import Agent, AgentPair, AgentFromPolicy
from overcooked_ai_py.agents.agent import RandomAgent, GreedyHumanModel
from overcooked_ai_py.mdp.actions import Action
import util
import random


class CentralizedAgent(AgentPair):

    def observeTransition(self, state, action, nextState, reward):
        self.a0.update(state, action, nextState, reward)
        self.a1.update(state, action, nextState, reward)

    def joint_action(self, state):
        act0 = self.a0.action(state)
        act1 = self.a1.action(state)
        return (act0, act1)

#RL Agent represents potential RL movement between 
class RLAgent(Agent):

    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.9):
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)

        self.q_values = util.Counter()

    def getQValue(self, state, action):
        return self.q_values[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)

        if not actions:
          return 0.0

        q_values = []
        for a in actions:
          q_values.append(self.getQValue(state, a))

        #Get the value from all (state, action) combinations
        return max(q_values)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if not actions:
          return None

        best_actions = []
        best_q_value = -9999

        for a in actions:
            action_value = self.getQValue(state, a)
            if action_value > best_q_value:
                best_actions = [a]
                best_q_value = action_value
            if action_value == best_q_value:
                best_actions.append(a)

        return random.choice(best_actions)     

    def getLegalActions(self, state):
        return Action.ALL_ACTIONS

    def action(self, state):
        """
        Should return an action, and an action info dictionary.
        If collecting trajectories of the agent with OvercookedEnv, the action
        info data will be included in the trajectory data under `ep_infos`.

        This allows agents to optionally store useful information about them
        in the trajectory for further analysis.
        """
        legalActions = self.getLegalActions(state)

        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)

        best_action = self.computeActionFromQValues(state)
        
        return best_action
        
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
        """
        "*** YOUR CODE HERE ***"
        cur_q_val = self.getQValue(state, action)
        # print("Q state:")
        # print((state, action))
        # print('--')
        self.q_values[(state, action)] = cur_q_val + self.alpha * (reward + self.discount*self.getValue(nextState) - cur_q_val)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def actions(self, states, agent_indices):
        """
        A multi-state version of the action method. This enables for parallized
        implementations that can potentially give speedups in action prediction. 

        Args:
            states (list): list of OvercookedStates for which we want actions for
            agent_indices (list): list to inform which agent we are requesting the action for in each state

        Returns:
            [(action, action_info), (action, action_info), ...]: the actions and action infos for each state-agent_index pair
        """
        return NotImplementedError()
