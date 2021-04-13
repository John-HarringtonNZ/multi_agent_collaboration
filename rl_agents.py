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

    def observeTransition(self, state, action, nextState, reward):
        self.a0.update(self.a0.process_state(state), action, self.a0.process_state(nextState), reward)
        self.a1.update(self.a1.process_state(state), action, self.a0.process_state(nextState), reward)

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
class BasicCommunicationPair(RLAgentPair):
    #who talks to who

    #self.agents = {ind:agent}

    def observeTransition(self, state, action, nextState, reward):
        print('stub')
        #self.a0.update(state, action, nextState, reward)
        #self.a1.update(state, action, nextState, reward)

    def joint_action(self, state):
        #act0 = self.a0.action(state)
        #act1 = self.a1.action(state)
        #return (act0, act1)
        return raiseNotDefined()

    def request_communication(self, agent_index):
        return self.agents[agent_index].communicate()


#Agent with RL functionality, processes state for Agent use.
#TODO: Update with potential feature usage
class RLAgent(Agent):

    def __init__(self, alpha=0.05, epsilon=0.05, gamma=0.9):
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        #self.pair = parentpair
        #self.index
        self.q_values = util.Counter()

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
    #what i say
    def communicate():
        return raiseNotDefined()

    #request from given agent
    def request_info(self, agent_index):
        return raiseNotDefined()


class ApproximateQAgent(RLAgent):
    """
       ApproximateQLearningAgent, similar to PA3

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, mdp, mlam, **args):
        super().__init__(**args)
        self.weights = util.Counter()
        self.mmlam = mlam
        self.mmdp = mdp

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
        features, _ = self.mmdp.custom_featurize_state(state, self.mmlam)#self.featExtractor.getFeatures(state, action)
        print(features)
        keys = features.keys()
        qval = 0
        for key in keys:
            print('feature')
            print(key)
            print(features[key])
            qval += (self.weights[key] * features[key])
        return qval

    #TODO: problem is this must be real states not process_state bc pass to getQValue
    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        cur_weights = self.getWeights()

        features, _ = self.mmdp.custom_featurize_state(state, self.mmlam)#self.featExtractor.getFeatures(state, action)
        difference = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)

        for feat, val in features.items():
            self.getWeights()[feat] = cur_weights[feat] + (self.alpha * difference * val)
