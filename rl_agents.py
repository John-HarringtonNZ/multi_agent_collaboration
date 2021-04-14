from overcooked_ai_py.agents.agent import Agent, AgentPair, AgentFromPolicy
from overcooked_ai_py.agents.agent import RandomAgent, GreedyHumanModel
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import ReducedOvercookedState
import util, math
import random

#Base class for AgentPair with RL functionality (observations, and custom joint actions)
class RLAgentPair(AgentPair):

    def observeTransition(self, state, action, nextState, reward):
        return raiseNotDefined()

    #def joint_action(self, state):
    #    return raiseNotDefined()
    
#RLAgentPair with 2 independent agents 
class DecentralizedAgent(RLAgentPair):

    def observeTransition(self, state, action, nextState, reward_pair):
        self.a0.update(self.a0.process_state(state), action[0], self.a0.process_state(nextState), reward_pair[0])
        self.a1.update(self.a1.process_state(state), action[1], self.a0.process_state(nextState), reward_pair[1])

    def joint_action(self, state):
        act0 = self.a0.action(self.a0.process_state(state))
        act1 = self.a1.action(self.a1.process_state(state))
        return (act0, act1)

#RLAgentPair that acts as single agent, each joint action is really one. 
#self.a1 is dummy agent
#TODO: Complete
class CentralizedAgent(RLAgentPair):

    def stub():
        print('abc')
    #self.a0 is actual agent
    #self.a1 is dummy?

#RLAgentPair that allows communication between CommunicateAgents
#TODO:Clean up and complete
class CommunicationPair(RLAgentPair):
    #who talks to who

    #self.agents = {ind:agent}
    def __init__(self, *agents):
        super().__init__(*agents)
        self.agent_dict = {}
        for i, a in enumerate(self.agents):
            self.agent_dict[i] = a
            a.parent = self
        self.agents[0].set_other_agent_index(1)
        self.agents[1].set_other_agent_index(0)

    def observeTransition(self, state, action_pair, nextState, reward_pair):
        self.a0.update(state, action_pair[0], nextState, reward_pair[0])
        self.a1.update(state, action_pair[1], nextState, reward_pair[1])

    def joint_action(self, state):
        act0 = self.a0.action(self.a0.process_state(state))
        act1 = self.a1.action(self.a1.process_state(state))
        return (act0, act1)

    def request_communication(self, agent_index):
        return self.agents[agent_index].communicate()


#Agent with RL functionality, processes state for Agent use.
#TODO: Update with potential feature usage
class RLAgent(Agent):

    def __init__(self, alpha=0.05, epsilon=0.05, gamma=0.9, parent=None):
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.q_values = util.Counter()
        self.parent = parent

    def getQValue(self, state, action):
        return self.q_values[(state, action)]

    #str or features.
    def process_state(self, state):
        return ReducedOvercookedState(state)

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
        best_q_value = -math.inf

        for a in actions:
            action_value = self.getQValue(state, a)
            if action_value > best_q_value:
                best_actions = [a]
                best_q_value = action_value
            if action_value == best_q_value:
                best_actions.append(a)

        return random.choice(best_actions)     

    def getLegalActions(self, state):
        #TODO: Limit actions to reasonable/doable
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


#RLAgent that can request information as well as provide information
class CommunicateAgent(RLAgent):


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.other_agent_index = 0
        self.communicable_information = []

    def set_other_agent_index(self, index):
        self.other_agent_index = index

    #what i say
    def communicate(self):
        return self.communicable_information

    #request from given agent
    def request_info(self, agent_index):
        return self.parent.request_communication(agent_index)


class ApproximateQAgent(RLAgent):
    """
       ApproximateQLearningAgent, similar to PA3

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, idx, mdp, mlam, **args):
        super().__init__(**args)
        self.set_agent_index(idx)
        self.weights = util.Counter()
        self.mmlam = mlam
        self.mmdp = mdp #TODO: this should be accessible at self.mdp via inheritance, but isn't...

    def getWeights(self):
        return self.weights

    def process_state(self, state):
        return state

    #TODO: problem is this must be real state not process_state
    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features, _ = self.mmdp.featurize(self.agent_index, state, action, self.mmlam)
        keys = features.keys()
        qval = 0
        for key in keys:
            #print('feature', self.agent_index)
            #print(key)
            #print(features[key])
            qval += (self.weights[key] * features[key])
        #print("---")
        return qval

    #TODO: problem is this must be real states not process_state bc pass to getQValue
    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        cur_weights = self.getWeights()

        features, _ = self.mmdp.featurize(self.agent_index, state, action, self.mmlam)
        difference = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)

        for feat, val in features.items():
            self.getWeights()[feat] = cur_weights[feat] + (self.alpha * difference * val)

#Agent communicates next expected action
#Simply concats this to existing state
class BasicCommunicateAgent(CommunicateAgent):

    #str or features.
    def process_state(self, state):
        comm_info = self.request_info(self.other_agent_index)
        return hash(str(ReducedOvercookedState(state)) + str(comm_info))

    #Update agent and update communicable information with next expected action
    def update(self, state, action, nextState, reward):
        super().update(state, action, nextState, reward)
        self.communicable_information = self.action(nextState)
