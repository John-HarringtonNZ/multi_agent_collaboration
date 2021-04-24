from overcooked_ai_py.agents.agent import Agent, AgentPair, AgentFromPolicy
from overcooked_ai_py.agents.agent import RandomAgent, GreedyHumanModel, StayAgent
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

    def getNumQVals(self):
        return (self.a0.getNumQVals(), self.a1.getNumQVals())

#RLAgentPair that acts as single agent, each joint action is really one. 
#self.a1 is dummy agent
#TODO: Complete
class CentralizedAgentPair(RLAgentPair):

    def observeTransition(self, state, action, nextState, reward_pair):
        self.a0.update(self.a0.process_state(state), action, self.a0.process_state(nextState), reward_pair[0]+reward_pair[1])

    def joint_action(self, state):
        act0 = self.a0.action(self.a0.process_state(state))
        return act0


#RLAgentPair that allows communication between CommunicateAgents
#TODO:Clean up and complete
class CommunicationPair(RLAgentPair):

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
class RLAgent(Agent):

    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.9, parent=None):
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.q_values = util.Counter()
        self.parent = parent

    def reset_q_values(self):
        self.q_values = util.Counter()

    def getNumQVals(self):
        return len(self.q_values)

    def getQValue(self, state, action):
        #if self.q_values[(state, action)] > 0:
            #print('getting prior state info!')
        return self.q_values[(state, action)]

    #str or features.
    def process_state(self, state):
        return ReducedOvercookedState(state)

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action) where the
          max is over legal actions. If at terminal state,
          return a value of 0.0.
        """
        actions = self.getLegalActions(state)
        if not actions:
            # at terminal state
            return 0.0
        q_values = []
        for a in actions:
            #Get the value from all (state, action) combinations
            q_values.append(self.getQValue(state, a))
        return max(q_values)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state. If at terminal state,
          return None.
        """
        actions = self.getLegalActions(state)
        if not actions:
            # at terminal state
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
        legalActions = self.getLegalActions(state)
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        best_action = self.computeActionFromQValues(state)
        return best_action
        
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          Do Q-Value update here
        """
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


class CentralAgent(RLAgent):
    """
      Centralized agent that treats each agent as an arm of a single agent.    
    """

    #Returns joint action due to central agent aspect
    def getLegalActions(self, state):
        import itertools
        single_actions = super().getLegalActions(state) 
        return list(itertools.product(single_actions, single_actions))

#Abstract RLAgent that can request and provide information to other agents
class CommunicateAgent(RLAgent):
    """
      RLAgent that can request information as well as provide information
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.other_agent_index = 0
        self.communicable_information = []

    def set_other_agent_index(self, index):
        self.other_agent_index = index

    # What I say
    def communicate(self):
        return self.communicable_information

    # Request from given agent
    def request_info(self, agent_index):
        return self.parent.request_communication(agent_index)


#This agent is a dummy agent that works within the RLAgent Hierarchy
class StayRLAgent(RLAgent, StayAgent):
    """
      An RLAgent that only stays in same place
    """

    def action(self, state):
        return 'stay'


class ApproximateQAgent(RLAgent):
    """
      ApproximateQLearningAgent, similar to PA3
    """

    def __init__(self, idx, mlam, **args):
        super().__init__(**args)
        self.set_agent_index(idx)
        self.weights = util.Counter()
        self.mmlam = mlam

    def getWeights(self):
        return self.weights

    def process_state(self, state):
        return state

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector ;where * is the dotProduct operator
          Must be real state here, not process_state, because featurize needs the actual state to pull info from
        """
        features, _ = self.mdp.featurize(self.agent_index, state, action, self.mmlam)
        keys = features.keys()
        qval = 0
        for key in keys:
            qval += (self.weights[key] * features[key])
        return qval

    def update(self, state, action, nextState, reward):
        """
          Should update your weights based on transition
          Must be real states here, not process_state because pass to getQValue
        """
        cur_weights = self.getWeights()

        features, _ = self.mdp.featurize(self.agent_index, state, action, self.mmlam)
        difference = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)

        for feat, val in features.items():
            self.getWeights()[feat] = cur_weights[feat] + (self.alpha * difference * val)


#Implementation of Communication Agent, looks at other agents future actions (1 step ahead)
class BasicCommunicateAgent(CommunicateAgent):
    """
      Agent communicates next expected action
      (Simply concats this to existing state)
    """

    #str or features.
    def process_state(self, state):
        comm_info = self.request_info(self.other_agent_index)
        return hash(str(ReducedOvercookedState(state)) + str(comm_info))

    #Update agent and update communicable information with next expected action
    def update(self, state, action, nextState, reward):
        super().update(state, action, nextState, reward)
        self.communicable_information = self.action(nextState)
