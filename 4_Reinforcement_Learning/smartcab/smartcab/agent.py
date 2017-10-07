import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions
        self.stepSim = 0
        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        # dictionaries to map states and actions to values
        waypointValid = ['forward', 'left', 'right']
        lightValid = ['red', 'green']

        self.statesD = dict()
        i = 1
        for wayp in waypointValid:
            for lightt in lightValid:
                for oncom in waypointValid:
                    for leftt in waypointValid:
                        self.statesD.update({'state-'+str(i) : [wayp, lightt, oncom, leftt]})
                        i += 1

        print self.statesD

        self.actionsD = dict()
        i = 1
        for actt in self.valid_actions:
            self.actionsD.update({'action-'+str(i) : actt}) 
            i += 1

        print self.actionsD


    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice

        decayF = 0
        if decayF == 0:
            # linear decay
            self.epsilon -= 0.05
        elif decayF == 1:
            # exponential decay
            aDecay = 0.999
            self.epsilon = aDecay**self.stepSim
        elif decayF == 2:
            # quadratic inverse
            if self.stepSim == 0:
                self.epsilon = 1
            else:
                self.epsilon = self.stepSim**(-2)
        elif decayF == 3:
            # exponential decay 2
            aDecay = 0.9
            self.epsilon = math.exp(-aDecay*self.stepSim)
        elif decayF == 4:
            aDecay = 0.1
            self.epsilon = math.cos(aDecay*self.stepSim)

        self.stepSim += 1
        # Update additional class parameters as needed

        # If 'testing' is True, set epsilon and alpha to 0
        if testing == True:
            self.epsilon = 0.0
            self.alpha = 0.0
            self.stepSim = 0

        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        
        # NOTE : you are not allowed to engineer features outside of the inputs available.
        # Because the aim of this project is to teach Reinforcement Learning, we have placed 
        # constraints in order for you to learn how to adjust epsilon and alpha, and thus learn about the balance between exploration and exploitation.
        # With the hand-engineered features, this learning process gets entirely negated.
        
        # Set 'state' as a tuple of relevant data for the agent        
        state = [waypoint, inputs['light'], inputs['oncoming'], inputs['left']];

        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
        #state = tuple(state)
        vectorQ = self.Q[state]
        maxQ = -100000
        for act in vectorQ:
            if maxQ < vectorQ[act]:
                maxQ = vectorQ[act]
            #print act
            #print maxQ


        return maxQ 


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        if self.learning == True:
            state = tuple(state)
            # print state

            exist = False
            for statt in self.Q:
                if statt == state:
                    exist = True
                    break
            print exist
            if exist == False:
                self.Q.update({state :  \
                    { None : 0.0,        \
                    'forward':0.0,      \
                    'left':0.0,         \
                    'right':0.0}})

        #print self.Q
        #return None


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        #action = None

        actions = [None, 'left', 'right', 'forward'];
        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        # Otherwise, choose an action with the highest Q-value for the current state
        # Be sure that when choosing an action with highest Q-value that you randomly select between actions that "tie".
        
        if self.learning == 0:
            # random action
            action = actions[random.randint(0, 3)]
        else:
            # sample exploration-explotation distribution
            sampleEE = random.random()
            if sampleEE <= self.epsilon:
                # random action
                action = actions[random.randint(0, 3)]    
                
            else:
                # action based on highest Q-value
                state = tuple(state)
                maxQ = self.get_maxQ(state)
                
                vectorQ = self.Q[state]
                actMaxQ = list()
                for act in vectorQ:
                    if maxQ == vectorQ[act]:
                        actMaxQ.append(act)
                print actMaxQ
                # randomly get one from list of possible actions
                action = actMaxQ[random.randint(0, len(actMaxQ)-1)]

        print action
        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        if self.learning == True:
            state = tuple (state)
            maxQ = self.get_maxQ(state)
            oldQ = self.Q[state][action]
            print oldQ

            self.Q[state][action] = oldQ + self.alpha*(reward + maxQ - oldQ)

            print self.Q[state][action]


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn
        
        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """
    
    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(verbose=True)
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, learning=True, epsilon=1, alpha=0.3)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay=0.01, display=False, log_metrics=True, optimized=False)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test=10)


if __name__ == '__main__':
    run()
